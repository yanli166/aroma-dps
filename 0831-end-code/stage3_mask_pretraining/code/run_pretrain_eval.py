"""
Stage3 掩码预训练实验: Ring Masking 作为独立学习策略评估

固定网络结构 (Stage2 最佳配置)、数据划分和下游训练设置,
比较3种训练方式:
  1. direct_supervised:    直接监督训练 (无预训练, 对照)
  2. random_mask_pretrain: 普通随机掩码预训练 + 微调
  3. ring_mask_pretrain:   环结构化掩码预训练 + 微调

预训练: 掩码原子特征重建 (GraphMAE 风格), 使用所有 trainval 数据 (无标签)
微调:   加载预训练权重, 用标签数据训练 (与 direct_supervised 相同下游流程)

输出目录结构: output_dir/seed_X/TASK/TRAINING_MODE/
结果 dict: 'model'=backbone, 'config'=training_mode
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

# 添加 last_end_code 到 sys.path (供 import common/ models/ stage3_mask_pretraining)
LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED
from common.graph_data import load_adj_format
from common.train_eval import set_full_seed, train_custom_model, eval_custom_model, save_results
from models.ring_conditioned_gnn import build_model
from stage3_mask_pretraining.code.mask_pretrain import MaskedAutoencoder, pretrain

# 默认训练参数 (与 Stage2 一致, 保证下游可比)
DEFAULT_PARAMS = {
    'hidden_dim': 128,
    'n_conv_layers': 3,
    'n_hidden_layers': 2,
    'learning_rate': 0.001,
    'p_dropout': 0.2,
    'batch_size': 64,
    'weight_decay': 1e-5,
    'n_epochs': 200,
    'patience': 30,
}
PRETRAIN_EPOCHS = 100
NODE_VEC_LEN = 60
MAX_ATOMS = 75

# 3种训练方式
TRAINING_MODES = ['direct_supervised', 'random_mask_pretrain', 'ring_mask_pretrain']


def select_best_stage2_config(stage2_csv, task_name):
    """从 Stage2 汇总 CSV 中选取指定任务的最佳配置

    选择标准: test_r2 (若无则 cv_r2) 最高
    Returns:
        dict: {backbone, ring_flag, readout_mode}
    """
    df = pd.read_csv(stage2_csv)
    # 过滤任务
    if 'task' in df.columns:
        sub = df[df['task'] == task_name].copy()
    else:
        sub = df.copy()
    if len(sub) == 0:
        raise ValueError(
            f"Stage2 CSV 中未找到任务 {task_name}, "
            f"现有任务: {df['task'].unique().tolist() if 'task' in df.columns else '无 task 列'}"
        )

    # 选择指标列 (优先 test_r2, 回退 cv_r2)
    metric_col = None
    for c in ['test_r2', 'cv_r2', 'test_r2_mean', 'cv_r2_mean']:
        if c in sub.columns:
            metric_col = c
            break
    if metric_col is None:
        raise ValueError(f"Stage2 CSV 中未找到 test_r2/cv_r2 列, 现有列: {list(sub.columns)}")

    best_row = sub.loc[sub[metric_col].idxmax()]

    # 解析 backbone
    backbone = None
    for c in ['backbone', 'model']:
        if c in best_row.index and pd.notna(best_row[c]):
            backbone = best_row[c]
            break
    if backbone is None:
        raise ValueError("Stage2 CSV 中未找到 backbone/model 列")

    # 解析 ring_flag
    if 'ring_flag' in best_row.index and pd.notna(best_row.get('ring_flag')):
        ring_flag = int(best_row['ring_flag'])
    else:
        # 从 config 名称推断 (Membership/Joint=10, Base/LearnableReadout=0)
        config = str(best_row.get('config', '')).lower()
        ring_flag = 10 if ('membership' in config or 'joint' in config) else 0

    # 解析 readout_mode
    if 'readout_mode' in best_row.index and pd.notna(best_row.get('readout_mode')):
        readout_mode = str(best_row['readout_mode'])
    else:
        config = str(best_row.get('config', '')).lower()
        if 'attention' in config or 'learnable' in config or 'joint' in config:
            readout_mode = 'attention'
        else:
            readout_mode = 'fixed_avg'

    return {
        'backbone': str(backbone),
        'ring_flag': ring_flag,
        'readout_mode': readout_mode,
    }


def run_one_mode(task, training_mode, best_cfg, params, data, splits, device,
                 output_dir, seed, pretrain_epochs):
    """运行单种训练方式 (fold 内预训练 -> 微调 -> 5折CV + final模型 + 测试集评估)

    预训练严格限定在每个 fold 的训练分子内, 防止验证集信息泄漏。
    每个 fold 独立预训练 (仅用该 fold 的训练分子), final 模型用 final_train 预训练。
    """
    task_name = task['name']
    test_idx, cv_folds, final_train_idx, final_val_idx = splits
    n_total = data['n']
    node_vec_len = data['node_vec_len']
    backbone = best_cfg['backbone']
    ring_flag = best_cfg['ring_flag']
    readout_mode = best_cfg['readout_mode']
    hidden_dim = params['hidden_dim']
    n_conv = params['n_conv_layers']
    n_hidden = params['n_hidden_layers']
    p_dropout = params['p_dropout']
    bs = params['batch_size']

    mode_dir = os.path.join(output_dir, f"seed_{seed}", task_name, training_mode)
    os.makedirs(mode_dir, exist_ok=True)

    print(f"\n=== [{task_name}] {training_mode} "
          f"(backbone={backbone}, ring_flag={ring_flag}, readout={readout_mode}) ===")
    t0 = time.time()

    do_pretrain = training_mode != 'direct_supervised'
    mask_type = 'random' if training_mode == 'random_mask_pretrain' else 'ring'

    def _pretrain_on(train_idx_t):
        """在指定训练索引上做掩码预训练, 返回 encoder state_dict (clone)"""
        mae_model = MaskedAutoencoder(
            backbone=backbone, node_vec_len=node_vec_len, hidden_dim=hidden_dim,
            n_conv=n_conv, n_hidden=n_hidden, p_dropout=p_dropout,
            readout_mode=readout_mode,
        ).to(device)
        mae_model = pretrain(
            mae_model, data, train_idx_t, device, mask_type=mask_type,
            n_epochs=pretrain_epochs, batch_size=bs,
            lr=params['learning_rate'], weight_decay=params['weight_decay'], seed=seed,
        )
        state = {k: v.clone() for k, v in mae_model.encoder.state_dict().items()}
        del mae_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return state

    # ===== 5折 CV (每折独立预训练, 仅用 fold 训练分子) =====
    cv_rows = []
    for fold_i, (tr_idx, va_idx) in enumerate(cv_folds):
        set_full_seed(seed + fold_i)

        # fold 内预训练 (仅 random/ring 模式): 严格限定在训练分子内
        pretrained_state = None
        if do_pretrain:
            print(f"  [Fold {fold_i+1} 预训练] mask={mask_type}, epochs={pretrain_epochs}, "
                  f"samples={len(tr_idx)}", flush=True)
            pretrained_state = _pretrain_on(tr_idx)

        model = build_model(backbone, node_vec_len, hidden_dim, n_conv, n_hidden,
                            n_outputs=1, p_dropout=p_dropout, readout_mode=readout_mode).to(device)
        if pretrained_state is not None:
            model.load_state_dict(pretrained_state, strict=False)
        model = train_custom_model(model, params, data, tr_idx, va_idx, device,
                                   n_epochs=params['n_epochs'], patience=params['patience'],
                                   seed=seed + fold_i)
        m, _, _ = eval_custom_model(model, data, va_idx, bs, device)
        cv_rows.append({'fold': fold_i + 1, 'r2': m[0], 'mae': m[1], 'rmse': m[2]})
        print(f"  Fold {fold_i+1}: R2={m[0]:.4f} MAE={m[1]:.4f} RMSE={m[2]:.4f}")
        del model

    # ===== Final 模型 (final_train 预训练 -> final_train 微调) =====
    set_full_seed(seed)

    # final 模型也独立预训练 (仅用 final_train 分子)
    pretrained_state = None
    if do_pretrain:
        print(f"  [Final 预训练] mask={mask_type}, epochs={pretrain_epochs}, "
              f"samples={len(final_train_idx)}", flush=True)
        pretrained_state = _pretrain_on(final_train_idx)

    final_model = build_model(backbone, node_vec_len, hidden_dim, n_conv, n_hidden,
                             n_outputs=1, p_dropout=p_dropout, readout_mode=readout_mode).to(device)
    if pretrained_state is not None:
        final_model.load_state_dict(pretrained_state, strict=False)
    final_model = train_custom_model(final_model, params, data, final_train_idx, final_val_idx,
                                     device, n_epochs=params['n_epochs'],
                                     patience=params['patience'], seed=seed)
    train_m, _, _ = eval_custom_model(final_model, data, final_train_idx, bs, device)
    test_m, te_pred, te_true = eval_custom_model(final_model, data, test_idx, bs, device)
    elapsed = time.time() - t0

    extra = {
        'ring_flag': ring_flag, 'readout_mode': readout_mode,
        'seed': seed, 'backbone': backbone, 'training_mode': training_mode,
        'pretrain_epochs': pretrain_epochs if training_mode != 'direct_supervised' else 0,
    }
    summary = save_results(mode_dir, backbone, training_mode, task_name, n_total,
                           cv_rows, train_m, test_m, te_pred, te_true, elapsed, extra=extra)
    summary['training_mode'] = training_mode
    print(f"  [结果] Test: R2={test_m[0]:.4f} MAE={test_m[1]:.4f} RMSE={test_m[2]:.4f} "
          f"(耗时 {elapsed:.1f}s)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage3 掩码预训练实验")
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(LAST_END_ROOT, 'results', 'stage3_mask_pretraining'),
        help="输出目录 (默认 .../results/stage3_mask_pretraining)",
    )
    parser.add_argument(
        '--stage2_results', type=str,
        default=os.path.join(LAST_END_ROOT, 'results', 'stage2_ring_conditioning',
                             'all_stage2_summary.csv'),
        help="Stage2 汇总 CSV 路径",
    )
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS,
                        help="预训练轮数 (默认 100)")
    parser.add_argument('--tasks', nargs='+', default=[t['name'] for t in TASKS],
                        help="任务列表 (默认全部)")
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'],
                        help="微调训练轮数")
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'],
                        help="early stopping patience")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="随机种子")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU 编号")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"输出目录: {args.output_dir}")

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    os.makedirs(args.output_dir, exist_ok=True)

    all_summaries = []
    for task in TASKS:
        if task['name'] not in args.tasks:
            continue
        # 1. 读取 Stage2 最佳配置
        best_cfg = select_best_stage2_config(args.stage2_results, task['name'])
        print(f"\n[{task['name']}] Stage2 最佳配置: backbone={best_cfg['backbone']}, "
              f"ring_flag={best_cfg['ring_flag']}, readout={best_cfg['readout_mode']}")

        # 2. 加载数据 (用最佳 ring_flag 决定是否注入环成员关系)
        data = load_adj_format(task['dataset_path'], task['target_col'],
                               node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
                               ring_flag_value=best_cfg['ring_flag'], device=device)

        # 3. 规范划分 (与 Stage1/2 一致, 保证可比)
        # parent-molecule-level split: 同一分子的多个环级样本进入同一 split
        test_idx, cv_folds, final_train_idx, final_val_idx = canonical_splits(
            data['n'], seed=args.seed, groups=data['smiles'])
        # 转为 device 上的 tensor (train_custom_model/eval_custom_model 需要)
        test_t = torch.tensor(test_idx, dtype=torch.long, device=device)
        ftrain_t = torch.tensor(final_train_idx, dtype=torch.long, device=device)
        fval_t = torch.tensor(final_val_idx, dtype=torch.long, device=device)
        cv_t = [(torch.tensor(tr, dtype=torch.long, device=device),
                 torch.tensor(va, dtype=torch.long, device=device))
                for tr, va in cv_folds]
        splits = (test_t, cv_t, ftrain_t, fval_t)

        # 4. 3种训练方式分别运行
        for mode in TRAINING_MODES:
            summary = run_one_mode(task, mode, best_cfg, params, data, splits,
                                   device, args.output_dir, args.seed, args.pretrain_epochs)
            all_summaries.append(summary)

    # 汇总保存
    if all_summaries:
        agg_df = pd.DataFrame(all_summaries)
        agg_path = os.path.join(args.output_dir, f"all_stage3_summary_seed_{args.seed}.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"\n汇总保存至: {agg_path}")


if __name__ == '__main__':
    main()
