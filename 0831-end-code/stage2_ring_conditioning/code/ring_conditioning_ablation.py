"""
[0831 重构] Stage 2: 环信息条件化 2×2 消融 (E*=median 协议 + feature_mode 双轨)

协议 (PROTOCOL_SPEC.md):
  1. 固定 20% holdout (split_seed=2026)
  2. 80% development 内 5-fold Group CV (每折早停, 记录 best_epoch)
  3. E* = median(best_epochs)
  4. 完整 dev 上按 E* 重训 final, fixed test 评估
  5. winner 仅由 CV MAE 选; final test 绝不上选择

4 configs (ring_flag x readout):
  - base:              ring_flag=0  + fixed_avg
  - membership:        ring_flag=10 + fixed_avg
  - learnable_readout: ring_flag=0  + attention
  - joint:             ring_flag=10 + attention

feature_mode 双轨 (default 'standard'; CLI 可切 'explicit_aromaticity_ablated')
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import RESULTS_V2_DIR, SPLITS_DIR
from common.protocol import (get_final_splits, make_group_ids, assert_no_leak,
                             completeness_check, SPLIT_SEED, MODEL_SEEDS)
from common.tasks import TASKS
from common.graph_data import load_adj_format
from common.train_eval import (set_full_seed, train_custom_model, eval_custom_model,
                                compute_estar, verify_split_invariants)
from common.features import get_feature_meta
from common.estar_pipeline import run_gnn_estar
from models.ring_conditioned_gnn import build_model

CONFIGS = {
    'base':              {'ring_flag': 0,  'readout': 'fixed_avg'},
    'membership':        {'ring_flag': 10, 'readout': 'fixed_avg'},
    'learnable_readout': {'ring_flag': 0,  'readout': 'attention'},
    'joint':             {'ring_flag': 10, 'readout': 'attention'},
}

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}
DEFAULT_BACKBONE = 'MPNN'
NODE_VEC_LEN, MAX_ATOMS = 60, 75


class FoldState:
    """per-task 数据/参数/设备 上下文, 避免 runner 接口带一堆上下文。"""
    cache = {}
    params = None
    device = None
    task = None
    _last_data = None  # 最近一次 train_cv/fit_final 用的 data, 供 _eval 精确取


def _train_cv(backbone, ring_flag, tr_idx, va_idx, n_epochs, patience, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', FoldState.task['name'], ring_flag, feature_mode)
    data = FoldState.cache.get(key)
    if data is None:
        data = load_adj_format(FoldState.task['dataset_path'],
                                FoldState.task['target_col'],
                                NODE_VEC_LEN, MAX_ATOMS,
                                ring_flag_value=ring_flag, device=FoldState.device,
                                feature_mode=feature_mode)
        FoldState.cache[key] = data
    cfg_name = _ringflag_to_cfg(ring_flag, 'fixed_avg')
    readout = CONFIGS[cfg_name]['readout']
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=FoldState.params['hidden_dim'],
                        n_conv=FoldState.params['n_conv_layers'],
                        n_hidden=FoldState.params['n_hidden_layers'],
                        p_dropout=FoldState.params['p_dropout'],
                        readout_mode=readout).to(FoldState.device)
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=FoldState.device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=FoldState.device)
    model = train_custom_model(model, FoldState.params, data, tr_t, va_t,
                                FoldState.device,
                                n_epochs=n_epochs, patience=patience, seed=seed)
    # 关键: 把本次跑用的 data 写入 _last_data, 避免 _eval 拿到别的 ring_flag 的 data
    FoldState._last_data = data
    return model


def _eval(model, idx):
    data = FoldState._last_data
    return eval_custom_model(model, data,
                              torch.tensor(idx, dtype=torch.long, device=FoldState.device),
                              FoldState.params['batch_size'],
                              FoldState.device)[0]


def _fit_final(backbone, ring_flag, train_idx, n_epochs, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', FoldState.task['name'], ring_flag, feature_mode)
    data = FoldState.cache[key]
    cfg_name = _ringflag_to_cfg(ring_flag, 'fixed_avg')
    readout = CONFIGS[cfg_name]['readout']
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=FoldState.params['hidden_dim'],
                        n_conv=FoldState.params['n_conv_layers'],
                        n_hidden=FoldState.params['n_hidden_layers'],
                        p_dropout=FoldState.params['p_dropout'],
                        readout_mode=readout).to(FoldState.device)
    tr_t = torch.tensor(np.asarray(train_idx), dtype=torch.long, device=FoldState.device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=FoldState.params['learning_rate'],
                                  weight_decay=FoldState.params['weight_decay'])
    loss_fn = torch.nn.MSELoss()
    bs = FoldState.params['batch_size']
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    ring_indices = data['ring_indices']; outputs = data['outputs']
    n = len(tr_t)
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n, device=FoldState.device)]
        for i in range(0, n, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = model(node_mats[bi], adj_mats[bi], ring_indices[bi]).reshape(-1)
            loss = loss_fn(pred, outputs[bi].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.best_epoch = n_epochs
    FoldState._last_data = data
    return model


def _ringflag_to_cfg(ring_flag, readout):
    for n, c in CONFIGS.items():
        if c['ring_flag'] == ring_flag and c['readout'] == readout:
            return n
    raise KeyError(f"未找到 ring_flag={ring_flag}, readout={readout}")


def run_task(task, backbone, params, device, n_epochs, patience, seed, feature_mode):
    task_name = task['name']
    print(f"\n{'='*72}\n任务: {task_name} | backbone: {backbone} | seed: {seed} | "
          f"feature_mode: {feature_mode}\n{'='*72}")

    # 固定 split (用 ring_flag=0 数据仅计算 group/smiles, 与 ring_flag 无关)
    data0 = load_adj_format(task['dataset_path'], task['target_col'],
                             NODE_VEC_LEN, MAX_ATOMS,
                             ring_flag_value=0, device=device, feature_mode=feature_mode)
    groups = make_group_ids(data0['smiles'])
    tag = f'stage2_{feature_mode}'
    splits = get_final_splits(data0['n'], groups, persist=SPLITS_DIR, tag=tag)
    verify_split_invariants(splits, groups, f"Stage2 {task_name} {feature_mode}")
    n_dev = len(splits['train_idx']); n_test = len(splits['test_idx'])
    n_total = n_dev + n_test

    FoldState.cache = {}; FoldState.params = params
    FoldState.device = device; FoldState.task = task

    per_seed_rows, cv_save_rows = [], []
    cv_map = {}
    for cfg in CONFIGS:
        ring_flag = CONFIGS[cfg]['ring_flag']
        res = run_gnn_estar(
            backbone=backbone, ring_flag=ring_flag,
            splits=splits, params=params,
            train_cv_fn=_train_cv, fit_final_fn=_fit_final,
            eval_fn=_eval, device=device, seed=seed,
            max_epochs=n_epochs, patience=patience, feature_mode=feature_mode,
        )
        cv_map[cfg] = res.cv_mae
        print(f"  [{cfg}] cv_mae={res.cv_mae:.4f} E*={res.estar} "
              f"best_epochs={res.best_epochs}")
        cv_save_rows.append({
            'task': task_name, 'model': backbone, 'config': cfg,
            'feature_mode': feature_mode, 'n': n_total,
            'cv_mae': res.cv_mae, 'cv_r2': res.cv_r2, 'cv_rmse': res.cv_rmse,
            'best_epochs': res.best_epochs, 'estar': res.estar,
            'n_ceiling_hits': res.n_ceiling_hits,
        })
        per_seed_rows.append({
            'seed': seed, 'task': task_name, 'model': backbone, 'config': cfg,
            'feature_mode': feature_mode, 'n': n_total,
            'cv_mae': res.cv_mae, 'cv_r2': res.cv_r2,
            'cv_r2_std': float(np.std([m for m in [res.cv_r2]], ddof=1)),
            'cv_rmse': res.cv_rmse,
            'best_epochs': res.best_epochs, 'estar': res.estar,
            'n_ceiling_hits': res.n_ceiling_hits,
            'test_r2': res.test_metrics[0],
            'test_mae': res.test_metrics[1],
            'test_rmse': res.test_metrics[2],
            'train_r2': res.train_metrics[0],
            'train_mae': res.train_metrics[1],
            'train_rmse': res.train_metrics[2],
            'train_time_sec': res.train_time_sec,
            'run_status': 'ok', 'error_message': '',
        })

    # ---- winner 仅按 CV MAE 选, final test 绝不上决策 ----
    winner = min(cv_map, key=cv_map.get)
    print(f"\n  [winner (按 cv_mae)] {task_name}: " +
          ", ".join(f"{c}={v:.4f}" for c, v in cv_map.items()) + f"  ->  {winner}")

    final_row = next(r for r in per_seed_rows if r['config'] == winner)
    final_summary = {
        'task': task_name, 'model': backbone, 'config': winner,
        'feature_mode': feature_mode, 'n': n_total,
        'cv_mae': cv_map[winner], 'dev_n': n_dev, 'test_n': n_test, 'seed': seed,
        'test_r2': final_row['test_r2'], 'test_mae': final_row['test_mae'],
        'test_rmse': final_row['test_rmse'],
        'train_r2': final_row['train_r2'], 'train_mae': final_row['train_mae'],
        'train_rmse': final_row['train_rmse'],
        'train_time_sec': final_row['train_time_sec'],
        'estar': final_row['estar'], 'best_epochs': final_row['best_epochs'],
        'run_status': 'ok',
    }
    return per_seed_rows, cv_save_rows, [final_summary], winner, cv_map


def main():
    parser = argparse.ArgumentParser(description='Stage 2: 环条件化 2×2 (E*=median)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(RESULTS_V2_DIR, 'stage2'))
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE)
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--seed', type=int, default=MODEL_SEEDS[0])
    parser.add_argument('--feature_mode', default='standard',
                        choices=['standard', 'explicit_aromaticity_ablated'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | feature_mode: {args.feature_mode}')

    if args.backbone not in ('GNN', 'GIN', 'GAT', 'MPNN', 'GraphSAGE'):
        raise ValueError(f"不支持的 backbone: {args.backbone}")

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == nxt) for nxt in args.tasks.split(',')]

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    feat_meta = get_feature_meta(args.feature_mode)
    seed = args.seed
    run_dir = os.path.join(args.output_dir, f'seed_{seed}', args.feature_mode)
    os.makedirs(run_dir, exist_ok=True)

    per_seed_all, cv_all, final_all = [], [], []
    cv_map_all = {}
    for task in task_list:
        per_seed_rows, cv_save_rows, final_rows, winner, cv_map = run_task(
            task, args.backbone, params, device,
            params['n_epochs'], params['patience'], seed, args.feature_mode)
        per_seed_all += per_seed_rows
        cv_all += cv_save_rows
        final_all += final_rows
        cv_map_all[task['name']] = cv_map

    per_df = pd.DataFrame(per_seed_all); cv_df = pd.DataFrame(cv_all)
    final_df = pd.DataFrame(final_all)
    per_df.to_csv(os.path.join(run_dir, 'per_seed_results.csv'), index=False)
    cv_df.to_csv(os.path.join(run_dir, 'cv_results.csv'), index=False)
    final_df.to_csv(os.path.join(run_dir, 'final_test_results.csv'), index=False)

    expected_idx = pd.MultiIndex.from_product(
        [list(CONFIGS.keys()), [t['name'] for t in task_list], [seed]],
        names=['config', 'task', 'seed'])
    completeness_check(pd.MultiIndex.from_frame(per_df[['config', 'task', 'seed']]),
                       expected_idx,
                       f"Stage2 (seed={seed}, feature_mode={args.feature_mode})")
    print(f"[完整性] Stage2 seed={seed} 缺失 0 个组合, 校验通过。")

    meta = {
        'protocol': 'PROTOCOL_SPEC Stage2 (P0-1 v2 + P0-2)',
        'split_seed': SPLIT_SEED, 'model_seed': seed,
        'feature_mode': args.feature_mode, 'feature_meta': feat_meta,
        'configs': CONFIGS, 'n_epochs_cv_max': params['n_epochs'],
        'patience': params['patience'],
        'protocol_pipeline': 'E* = median(best_epochs); final = retrain on full dev for E* epochs',
        'selection_rule': 'winner = min(cv_mae) over 4 configs; final test 不参与选择',
    }
    meta_path = os.path.join(run_dir, 'stage2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}\nStage 2 (seed={seed}, feature_mode={args.feature_mode}) 汇总\n{'='*72}")
    print("--- 逐配置 CV (模型选择依据) ---")
    print(cv_df[['task', 'model', 'config', 'cv_mae', 'estar']].to_string(index=False))
    print("\n--- Winner (按 cv_mae 选择) + final test ---")
    print(final_df[['task', 'model', 'config', 'cv_mae', 'estar',
                     'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
    print("\n各任务选出的 winner:")
    for t, m in cv_map_all.items():
        print(f"  {t:>8s}: winner={min(m, key=m.get)}  cv_mae_map=" +
              ", ".join(f"{c}={v:.4f}" for c, v in m.items()))
    print(f"\n结果: {run_dir}\nMeta: {meta_path}")


if __name__ == '__main__':
    main()
