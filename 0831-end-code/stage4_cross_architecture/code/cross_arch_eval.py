#!/usr/bin/env python3
"""
Stage 4: 跨架构实验 (Cross-Architecture Evaluation)

检验 ring-conditioned architecture 是否依赖于特定 molecular encoder。
将相同的 Target-Ring Membership Encoding 与 Ring-Level Readout 应用于
MPNN, D-MPNN, GIN, GAT 等代表性图神经网络, 在每种 backbone 内比较
基础模型与 ring-conditioned model 的性能变化。

对每个 backbone, 比较 2 种配置:
  - base:             ring_flag=0  + fixed_avg readout  (Stage1 配置, 无环条件化)
  - ring_conditioned: ring_flag=10 + attention readout  (Stage2 Joint 配置, 完整环条件化)

判断: 显式目标环条件化是否能作为跨网络架构的通用结构先验。

实验流程 (每个 backbone x 配置):
  1. 5 折交叉验证 (在 train_val 上)
  2. Final 模型训练 (final_train, 用 final_val 做 early stopping)
  3. 测试集评估 (test_idx)
  4. 保存 summary.csv / cv_results.csv / test_predictions.csv / parity_plot.png

输出目录结构:
  output_dir/seed_X/TASK/BACKBONE_CONFIG/
    例如: seed_42/HOMA/MPNN_base/
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

# 将 last_end_code 根目录加入 sys.path, 以支持 common / models 包导入
_LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAST_END_ROOT not in sys.path:
    sys.path.insert(0, _LAST_END_ROOT)

from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED, get_task
from common.graph_data import load_adj_format, load_pyg_format
from common.train_eval import (
    set_full_seed,
    train_custom_model,
    eval_custom_model,
    train_pyg_model,
    eval_pyg_model,
    save_results,
)
from models.ring_conditioned_gnn import build_model
from models.pyg_models import build_pyg_model


# ============== 配置定义 ==============
# 两种配置: base (无环条件化) vs ring_conditioned (完整环条件化)
CONFIGS = {
    'base':             {'ring_flag': 0,  'readout': 'fixed_avg'},
    'ring_conditioned': {'ring_flag': 10, 'readout': 'attention'},
}

# 使用自定义 RingConditionedGNN 的 backbone
BACKBONES_CUSTOM = ['MPNN', 'GIN', 'GAT']
# 使用 PyG 模型的 backbone
BACKBONES_PYG = ['DMPNN']

# 默认训练超参数
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

# 节点特征维度 / 最大原子数 (与各阶段保持一致)
NODE_VEC_LEN = 60
MAX_ATOMS = 75


def _to_tensor_idx(idx, device):
    """将 numpy 索引转为 device 上的 torch.long 张量 (供自定义模型张量索引使用)"""
    return torch.tensor(np.asarray(idx), dtype=torch.long, device=device)


# ============== 自定义 backbone (MPNN/GIN/GAT) ==============
def run_custom_backbone(backbone, config_name, task, params, seed, n_epochs, patience, device):
    """运行自定义 backbone (MPNN/GIN/GAT) 的单次配置实验

    流程: 5折CV + final模型 + 测试集评估
    返回包含 cv_rows / train_metrics / test_metrics / 预测值 / 训练耗时 的 dict
    """
    cfg = CONFIGS[config_name]
    ring_flag = cfg['ring_flag']
    readout_mode = cfg['readout']

    print(f"\n{'='*70}")
    print(f"[Custom] backbone={backbone} | config={config_name} | task={task['name']}")
    print(f"  ring_flag={ring_flag}, readout={readout_mode}")
    print(f"{'='*70}", flush=True)

    # 加载数据 (adj 格式, 直接放到 device 上)
    data = load_adj_format(
        task['dataset_path'], task['target_col'],
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device=device,
    )
    n_total = data['n']

    # 规范化划分 (与各阶段保持一致, 保证可比)
    # parent-molecule-level split: 同一分子的多个环级样本进入同一 split
    test_idx_np, cv_folds, final_train_idx_np, final_val_idx_np = canonical_splits(
        n_total, seed=seed, groups=data['smiles'])

    # ---------- 5 折交叉验证 ----------
    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        print(f"\n  --- CV fold {fold+1}/5 ---", flush=True)
        set_full_seed(seed + fold)
        model = build_model(
            backbone, node_vec_len=NODE_VEC_LEN,
            hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
            n_hidden=params['n_hidden_layers'], p_dropout=params['p_dropout'],
            readout_mode=readout_mode,
        ).to(device)
        train_idx = _to_tensor_idx(tr_idx, device)
        val_idx = _to_tensor_idx(va_idx, device)
        model = train_custom_model(
            model, params, data, train_idx, val_idx, device,
            n_epochs=n_epochs, patience=patience, seed=seed + fold,
        )
        metrics, _, _ = eval_custom_model(model, data, val_idx, params['batch_size'], device)
        r2, mae, rmse = metrics
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        print(f"  fold {fold+1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        del model

    # ---------- Final 模型 (final_train 训练, final_val early-stopping, test 评估) ----------
    print(f"\n  --- Final model (train={len(final_train_idx_np)}, val={len(final_val_idx_np)}, test={len(test_idx_np)}) ---", flush=True)
    set_full_seed(seed)
    final_model = build_model(
        backbone, node_vec_len=NODE_VEC_LEN,
        hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'], p_dropout=params['p_dropout'],
        readout_mode=readout_mode,
    ).to(device)
    final_train_idx = _to_tensor_idx(final_train_idx_np, device)
    final_val_idx = _to_tensor_idx(final_val_idx_np, device)
    t0 = time.time()
    final_model = train_custom_model(
        final_model, params, data, final_train_idx, final_val_idx, device,
        n_epochs=n_epochs, patience=patience, seed=seed,
    )
    train_time = time.time() - t0

    # 训练集 + 测试集评估
    train_metrics, _, _ = eval_custom_model(final_model, data, final_train_idx, params['batch_size'], device)
    test_idx = _to_tensor_idx(test_idx_np, device)
    test_metrics, te_pred, te_true = eval_custom_model(
        final_model, data, test_idx, params['batch_size'], device)
    del final_model

    print(f"\n  CV    R2={np.mean([r['r2'] for r in cv_rows]):.4f} +/- {np.std([r['r2'] for r in cv_rows], ddof=1):.4f}")
    print(f"  Train R2={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}")
    print(f"  Test  R2={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    return {
        'cv_rows': cv_rows,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'te_pred': te_pred,
        'te_true': te_true,
        'train_time_sec': train_time,
        'n_total': n_total,
    }


# ============== PyG backbone (DMPNN) ==============
def run_pyg_backbone(backbone, config_name, task, params, seed, n_epochs, patience, device):
    """运行 PyG backbone (DMPNN) 的单次配置实验

    流程: 5折CV + final模型 + 测试集评估
    返回包含 cv_rows / train_metrics / test_metrics / 预测值 / 训练耗时 的 dict
    """
    cfg = CONFIGS[config_name]
    ring_flag = cfg['ring_flag']
    readout_mode = cfg['readout']

    print(f"\n{'='*70}")
    print(f"[PyG] backbone={backbone} | config={config_name} | task={task['name']}")
    print(f"  ring_flag={ring_flag}, readout={readout_mode}")
    print(f"{'='*70}", flush=True)

    # 加载数据 (PyG 格式, CPU 上; 训练时 batch 再 to(device))
    data_list, _ = load_pyg_format(
        task['dataset_path'], task['target_col'],
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device='cpu',
    )
    n_total = len(data_list)
    in_channels = data_list[0].x.size(-1)

    # 规范化划分 (parent-molecule-level split)
    test_idx_np, cv_folds, final_train_idx_np, final_val_idx_np = canonical_splits(
        n_total, seed=seed, groups=[d.smiles for d in data_list])

    # ---------- 5 折交叉验证 ----------
    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        print(f"\n  --- CV fold {fold+1}/5 ---", flush=True)
        set_full_seed(seed + fold)
        model = build_pyg_model(
            backbone, in_channels=in_channels,
            hidden_dim=params['hidden_dim'], n_layers=params['n_conv_layers'],
            dropout=params['p_dropout'], edge_dim=4,
            readout_mode=readout_mode,
        ).to(device)
        train_idx = np.asarray(tr_idx)
        val_idx = np.asarray(va_idx)
        model = train_pyg_model(
            model, params, data_list, train_idx, val_idx, device,
            n_epochs=n_epochs, patience=patience, seed=seed + fold,
        )
        metrics, _, _ = eval_pyg_model(model, data_list, val_idx, params['batch_size'], device)
        r2, mae, rmse = metrics
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        print(f"  fold {fold+1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        del model

    # ---------- Final 模型 ----------
    print(f"\n  --- Final model (train={len(final_train_idx_np)}, val={len(final_val_idx_np)}, test={len(test_idx_np)}) ---", flush=True)
    set_full_seed(seed)
    final_model = build_pyg_model(
        backbone, in_channels=in_channels,
        hidden_dim=params['hidden_dim'], n_layers=params['n_conv_layers'],
        dropout=params['p_dropout'], edge_dim=4,
        readout_mode=readout_mode,
    ).to(device)
    final_train_idx = np.asarray(final_train_idx_np)
    final_val_idx = np.asarray(final_val_idx_np)
    t0 = time.time()
    final_model = train_pyg_model(
        final_model, params, data_list, final_train_idx, final_val_idx, device,
        n_epochs=n_epochs, patience=patience, seed=seed,
    )
    train_time = time.time() - t0

    train_metrics, _, _ = eval_pyg_model(final_model, data_list, final_train_idx, params['batch_size'], device)
    test_idx = np.asarray(test_idx_np)
    test_metrics, te_pred, te_true = eval_pyg_model(
        final_model, data_list, test_idx, params['batch_size'], device)
    del final_model

    print(f"\n  CV    R2={np.mean([r['r2'] for r in cv_rows]):.4f} +/- {np.std([r['r2'] for r in cv_rows], ddof=1):.4f}")
    print(f"  Train R2={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}")
    print(f"  Test  R2={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    return {
        'cv_rows': cv_rows,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'te_pred': te_pred,
        'te_true': te_true,
        'train_time_sec': train_time,
        'n_total': n_total,
    }


# ============== 主入口 ==============
def main():
    parser = argparse.ArgumentParser(description='Stage 4: 跨架构环条件化评估 (Cross-Architecture)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_LAST_END_ROOT, 'results', 'stage4_cross_arch'),
                        help='结果输出根目录')
    parser.add_argument('--backbones', type=str, default='MPNN,DMPNN,GIN,GAT',
                        help='逗号分隔的 backbone 列表 (可选: MPNN,DMPNN,GIN,GAT)')
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO',
                        help='逗号分隔的任务列表 (可选: HOMA,NICS_1zz,MBCO)')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'],
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'],
                        help='early stopping patience')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU 设备 id (无 GPU 时自动回退 CPU)')
    args = parser.parse_args()

    # 设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Stage 4 跨架构实验 | device={device}, seed={args.seed}")
    print(f"output_dir={args.output_dir}")
    print(f"配置对比: {list(CONFIGS.keys())}")

    # 解析 backbone / task 列表
    backbones = [b.strip() for b in args.backbones.split(',') if b.strip()]
    task_names = [t.strip() for t in args.tasks.split(',') if t.strip()]
    valid_backbones = BACKBONES_CUSTOM + BACKBONES_PYG
    for b in backbones:
        if b not in valid_backbones:
            print(f"[警告] 未知 backbone: {b} (可选: {valid_backbones}), 将跳过")

    # 构建超参数 dict (命令行覆盖默认值)
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    all_summaries = []

    for task_name in task_names:
        task = get_task(task_name)
        for backbone in backbones:
            # 判断 backbone 类型, 选择对应 runner
            if backbone in BACKBONES_PYG:
                runner = run_pyg_backbone
                bb_type = 'pyg'
            elif backbone in BACKBONES_CUSTOM:
                runner = run_custom_backbone
                bb_type = 'custom'
            else:
                continue

            for config_name in CONFIGS:
                # 输出目录: output_dir/seed_X/TASK/BACKBONE_CONFIG/
                out_dir = os.path.join(
                    args.output_dir, f'seed_{args.seed}', task_name, f'{backbone}_{config_name}'
                )
                os.makedirs(out_dir, exist_ok=True)

                # 运行单次实验
                res = runner(
                    backbone, config_name, task, params, args.seed,
                    params['n_epochs'], params['patience'], device,
                )

                # 保存结果 (summary.csv / cv_results.csv / test_predictions.csv / parity_plot.png)
                summary = save_results(
                    out_dir=out_dir,
                    model_name=backbone,
                    config_name=config_name,
                    task_name=task_name,
                    n_total=res['n_total'],
                    cv_rows=res['cv_rows'],
                    train_metrics=res['train_metrics'],
                    test_metrics=res['test_metrics'],
                    te_pred=res['te_pred'],
                    te_true=res['te_true'],
                    train_time_sec=res['train_time_sec'],
                    extra={
                        'backbone_type': bb_type,
                        'ring_flag': CONFIGS[config_name]['ring_flag'],
                        'readout_mode': CONFIGS[config_name]['readout'],
                        'seed': args.seed,
                    },
                )
                summary['seed'] = args.seed
                all_summaries.append(summary)

    # 汇总所有实验结果到单个 CSV (供后续跨架构对比分析)
    if all_summaries:
        import pandas as pd
        agg_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
        os.makedirs(agg_dir, exist_ok=True)
        agg_path = os.path.join(agg_dir, 'all_results.csv')
        pd.DataFrame(all_summaries).to_csv(agg_path, index=False)
        print(f"\n所有实验结果汇总已保存: {agg_path}")

    print("\nStage 4 跨架构实验完成。")


if __name__ == '__main__':
    main()
