"""
[0831 协议重构] Stage 3: 掩码预训练 (结果写入 results_v2/stage3)

科学问题 (Fig.3 / 参考思路2.txt #14):
  - Ring masking 作为独立自监督策略是否提升目标环芳香性预测?
  - 固定 Membership backbone (ring_flag=10, readout=fixed_avg, ring_value=10), 比较
      direct_supervised      : 直接监督训练 (对照)
      random_mask_pretrain   : 随机掩码预训练 + 微调
      ring_mask_pretrain     : 环结构化掩码预训练 + 微调

协议铁律 (PROTOCOL_SPEC.md):
  1. 固定 80/20 holdout: common.protocol.get_final_splits (SPLIT_SEED=2026)。
     model seed 绝不进入 split; 同一任务下不同 training_mode/model seed 的 test 集逐字节相同。
  2. 分组: make_group_ids(data['smiles']) (canonical SMILES); 启动即 assert_no_leak。
  3. 模型选择只看 CV (val-MAE); final test 绝不用于选择。
  4. MAE 为主指标; 输出至少三份: cv_results.csv / final_test_results.csv / per_seed_results.csv。
  5. 完整性: completeness_check 校验理论组合无缺失。
  6. 结果写入 results_v2/stage3, 不覆盖旧 results/。

Dry-run 配置: 只跑 1 个 model seed = 11, 低 epoch (预训练 n_epochs≈10, 微调 n_epochs≈25, patience≈8),
             三个任务全跑 (HOMA / NICS_1zz / MBCO)。
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import RESULTS_V2_DIR, SPLITS_DIR
from common.protocol import (get_final_splits, make_group_ids, assert_no_leak,
                             completeness_check, SPLIT_SEED, MODEL_SEEDS)
from common.tasks import TASKS
from common.graph_data import load_adj_format
from common.train_eval import set_full_seed, train_custom_model, eval_custom_model
from models.ring_conditioned_gnn import build_model
from stage3_mask_pretraining.code.mask_pretrain import MaskedAutoencoder, pretrain

# ============== Membership backbone (Stage3 固定配置, 参考思路2.txt #14) ==============
MEMBERSHIP = {'ring_flag': 10, 'readout': 'fixed_avg', 'ring_value': 10}
DEFAULT_BACKBONE = 'MPNN'          # dry-run 建议 backbone (协议 line 35 / Stage2 一致)

# 3 种训练方式
TRAINING_MODES = ['direct_supervised', 'random_mask_pretrain', 'ring_mask_pretrain']

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 25, 'patience': 8,   # dry-run 低 epoch (微调)
}
PRETRAIN_EPOCHS = 10                # dry-run 低 epoch (预训练)
NODE_VEC_LEN, MAX_ATOMS = 60, 75


def make_final_train_val(dev_idx, groups, seed=SPLIT_SEED, val_ratio=0.125):
    """dev 内部 87.5/12.5 group split (用 split seed, 不用 model seed) 供 final 模型 early-stop。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_pos, va_pos = next(gss.split(np.asarray(dev_idx), groups=np.asarray(groups)[dev_idx]))
    return np.asarray(dev_idx[tr_pos]), np.asarray(dev_idx[va_pos])


def build_encoder(backbone, readout, params, device):
    return build_model(
        backbone, node_vec_len=NODE_VEC_LEN,
        hidden_dim=params['hidden_dim'],
        n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'],
        p_dropout=params['p_dropout'],
        readout_mode=readout,
    ).to(device)


def pretrain_encoder(backbone, readout, params, data, train_idx_t, device,
                     mask_type, seed, pretrain_epochs, bs):
    """在指定训练索引上做掩码预训练, 返回 encoder state_dict (clone)。"""
    mae_model = MaskedAutoencoder(
        backbone=backbone, node_vec_len=data['node_vec_len'],
        hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'], p_dropout=params['p_dropout'],
        readout_mode=readout,
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


def run_mode(task, mode, backbone, params, data, splits, groups, device,
             share_dir, seed, pretrain_epochs):
    """运行单种训练方式: (可选) 每折独立预训练 -> 微调 -> 5折CV + final 模型 -> 固定 test。

    预训练严格限定在每个 fold 的训练分子内 (防验证集泄漏)。
    返回 (per_seed_row, cv_rows, final_row)。
    """
    task_name = task['name']
    do_pretrain = mode != 'direct_supervised'
    mask_type = 'random' if mode == 'random_mask_pretrain' else 'ring'

    # ---------- 5 折 CV (每折独立预训练, 仅用 fold 训练分子) ----------
    cv_rows = []
    for fold_i, (tr, va) in enumerate(splits['folds']):
        tr_t = torch.tensor(tr, device=device)
        va_t = torch.tensor(va, device=device)
        set_full_seed(seed + fold_i)
        t0 = time.time()

        pretrained_state = None
        if do_pretrain:
            pretrained_state = pretrain_encoder(
                backbone, MEMBERSHIP['readout'], params, data, tr_t, device,
                mask_type, seed + fold_i, pretrain_epochs, params['batch_size'])

        model = build_encoder(backbone, MEMBERSHIP['readout'], params, device)
        if pretrained_state is not None:
            model.load_state_dict(pretrained_state, strict=False)
        model = train_custom_model(model, params, data, tr_t, va_t, device,
                                   n_epochs=params['n_epochs'], patience=params['patience'],
                                   seed=seed + fold_i)
        (r2, mae, rmse), _, _ = eval_custom_model(model, data, va_t, params['batch_size'], device)
        print(f"  Fold {fold_i+1} [{mode}]: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} "
              f"({time.time()-t0:.0f}s)")
        cv_rows.append({'fold': fold_i + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        del model
        torch.cuda.empty_cache()

    cv_mae = float(np.mean([r['mae'] for r in cv_rows]))
    cv_rmse = float(np.mean([r['rmse'] for r in cv_rows]))
    cv_r2 = float(np.mean([r['r2'] for r in cv_rows]))
    cv_r2_std = float(np.std([r['r2'] for r in cv_rows], ddof=1))
    print(f"  [CV {mode}] -> cv_mae={cv_mae:.4f} cv_rmse={cv_rmse:.4f} cv_r2={cv_r2:.4f}±{cv_r2_std:.4f}")

    # ---------- Final 模型 (dev 上 final_train 预训练(可选) + 微调) => 固定 test ----------
    final_tr, final_va = make_final_train_val(splits['train_idx'], make_group_ids(data['smiles']))
    final_tr_t = torch.tensor(final_tr, device=device)
    final_va_t = torch.tensor(final_va, device=device)
    test_t = torch.tensor(splits['test_idx'], device=device)

    set_full_seed(seed)
    t0 = time.time()
    pretrained_state = None
    if do_pretrain:
        pretrained_state = pretrain_encoder(
            backbone, MEMBERSHIP['readout'], params, data, final_tr_t, device,
            mask_type, seed, pretrain_epochs, params['batch_size'])
    final_model = build_encoder(backbone, MEMBERSHIP['readout'], params, device)
    if pretrained_state is not None:
        final_model.load_state_dict(pretrained_state, strict=False)
    final_model = train_custom_model(final_model, params, data, final_tr_t, final_va_t, device,
                                     n_epochs=params['n_epochs'], patience=params['patience'],
                                     seed=seed)
    train_time = time.time() - t0
    (tr_r2, tr_mae, tr_rmse), _, _ = eval_custom_model(final_model, data, final_tr_t, params['batch_size'], device)
    (te_r2, te_mae, te_rmse), te_pred, te_true = eval_custom_model(final_model, data, test_t, params['batch_size'], device)
    best_epoch = getattr(final_model, 'best_epoch', -1)
    del final_model
    torch.cuda.empty_cache()
    print(f"  [final test {mode}] test_mae={te_mae:.4f} test_r2={te_r2:.4f} "
          f"test_rmse={te_rmse:.4f} ({train_time:.0f}s)")

    per_seed_row = {
        'seed': seed, 'task': task_name, 'model': backbone, 'config': mode,
        'n': len(te_true),
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'best_epoch': best_epoch,
        'pretrain_epochs': pretrain_epochs if do_pretrain else 0,
        'ring_flag': MEMBERSHIP['ring_flag'], 'readout_mode': MEMBERSHIP['readout'],
        'run_status': 'ok', 'error_message': '',
    }
    final_row = {
        'task': task_name, 'model': backbone, 'config': mode, 'n': len(te_true),
        'cv_mae': cv_mae, 'dev_n': len(splits['train_idx']), 'test_n': len(splits['test_idx']),
        'seed': seed,
        'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae, 'train_rmse': tr_rmse,
        'train_time_sec': train_time,
        'run_status': 'ok',
    }
    return per_seed_row, cv_rows, final_row


def run_task(task, backbone, params, device, share_dir, seed, pretrain_epochs):
    """单个 task: 固定 split -> 3 种训练方式依次运行。返回 (per_rows, cv_rows, final_rows)。"""
    task_name = task['name']
    print(f"\n{'='*72}\n任务: {task_name} | backbone: {backbone} (Membership) | seed: {seed}\n{'='*72}")

    # 固定 split 只计算一次 (load ring_flag 不影响 split; index 对所有 mode 一致)
    data = load_adj_format(task['dataset_path'], task['target_col'], NODE_VEC_LEN, MAX_ATOMS,
                           ring_flag_value=MEMBERSHIP['ring_flag'], device=device)
    groups = make_group_ids(data['smiles'])
    splits = get_final_splits(data['n'], groups, persist=SPLITS_DIR)
    assert_no_leak(splits['train_idx'], splits['test_idx'], groups, f"{task_name} final 80/20")
    for k, (tr, va) in enumerate(splits['folds']):
        assert_no_leak(tr, va, groups, f"{task_name} fold{k+1}")

    # 每个 mode 保存自己的 cv/final 明细
    for mode in TRAINING_MODES:
        mode_dir = os.path.join(share_dir, task_name, mode)
        os.makedirs(mode_dir, exist_ok=True)
        pe_row, cv_rows, final_row = run_mode(
            task, mode, backbone, params, data, splits, groups, device,
            share_dir, seed, pretrain_epochs)
        pd.DataFrame(cv_rows).to_csv(os.path.join(mode_dir, 'cv_results.csv'), index=False)
        pd.DataFrame([final_row]).to_csv(
            os.path.join(mode_dir, 'final_test_results.csv'), index=False)
        pe_row_df = pd.DataFrame([pe_row])
        per_csv = os.path.join(mode_dir, 'per_seed_results.csv')
        if os.path.exists(per_csv):
            pe_row_df = pd.concat([pd.read_csv(per_csv), pe_row_df], ignore_index=True)
        pe_row_df.to_csv(per_csv, index=False)
        # 汇总到主目录 (merge 视图)
        pd.DataFrame([pe_row]).to_csv(os.path.join(share_dir, f'per_seed_{mode}.csv'),
                                      mode='a', header=not os.path.exists(
                                          os.path.join(share_dir, f'per_seed_{mode}.csv')),
                                      index=False)
    return groups


def main():
    parser = argparse.ArgumentParser(description='Stage 3: 掩码预训练 (Membership backbone, results_v2/stage3)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(RESULTS_V2_DIR, 'stage3'))
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE)
    parser.add_argument('--tasks', type=str, default='all',
                        help='逗号分隔 (HOMA,NICS_1zz,MBCO) 或 all')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'], help='微调 epoch')
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS, help='预训练 epoch')
    parser.add_argument('--seed', type=int, default=MODEL_SEEDS[0], help='model seed (dry-run=11)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.backbone not in ('GNN', 'GIN', 'GAT', 'MPNN', 'GraphSAGE'):
        raise ValueError(f"不支持的 backbone: {args.backbone}")

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == nxt) for nxt in args.tasks.split(',')]

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    seed = args.seed
    out_dir = args.output_dir
    run_dir = os.path.join(out_dir, f'seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")
    print(f"任务数: {len(task_list)}  训练方式: {TRAINING_MODES}  预训练 epochs={args.pretrain_epochs}")

    per_all, groups_cache = [], {}
    for task in task_list:
        run_task(task, args.backbone, params, device,
                 run_dir, seed, args.pretrain_epochs)

    # ---------- 汇总主目录三份 CSV (merge 各 mode 文件) ----------
    for mode in TRAINING_MODES:
        mode_csv = os.path.join(run_dir, f'per_seed_{mode}.csv')
        if os.path.exists(mode_csv):
            per_all.append(pd.read_csv(mode_csv))

    per_df = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    final_df = per_df[['task', 'model', 'config', 'n', 'cv_mae',
                       'test_r2', 'test_mae', 'test_rmse']].copy()

    per_csv = os.path.join(run_dir, 'per_seed_results.csv')
    cv_csv = os.path.join(run_dir, 'cv_results.csv')
    final_csv = os.path.join(run_dir, 'final_test_results.csv')
    per_df.to_csv(per_csv, index=False)
    final_df.to_csv(final_csv, index=False)
    # cv_results.csv: 只保留 5 折 CV 明细
    cv_folds_df = pd.concat([pd.read_csv(os.path.join(run_dir, t['name'], mode, 'cv_results.csv'))
                             .assign(task=t['name'], config=mode, model=args.backbone, seed=seed)
                             for mode in TRAINING_MODES for t in task_list], ignore_index=True)
    cv_folds_df.to_csv(cv_csv, index=False)

    # ---------- 完整性检查 (seed=11, training_modes x tasks) ----------
    expected_idx = pd.MultiIndex.from_product(
        [TRAINING_MODES, [t['name'] for t in task_list], [seed]],
        names=['config', 'task', 'seed'])
    actual_idx = pd.MultiIndex.from_frame(per_df[['config', 'task', 'seed']])
    ok = True
    try:
        completeness_check(actual_idx, expected_idx, f"Stage3 (seed={seed}) 完整组合")
        print(f"[完整性] Stage3 seed={seed} 缺失 0 个组合, 校验通过。")
    except RuntimeError as e:
        ok = False
        print(f"[完整性] 校验失败: {e}")

    # ---------- 汇总打印 ----------
    print(f"\n{'='*72}\nStage 3 掩码预训练汇总 (Membership backbone={args.backbone}, seed={seed})\n{'='*72}")
    print("--- 每算法 CV (模型选择依据: cv_mae) ---")
    print(final_df[['task', 'config', 'cv_mae']].to_string(index=False))
    print("\n--- Final test 结果 (仅报告, 不用于选择) ---")
    print(final_df[['task', 'config', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
    print(f"\n完整性通过: {ok}")
    print(f"结果已保存: {run_dir}")


if __name__ == '__main__':
    main()