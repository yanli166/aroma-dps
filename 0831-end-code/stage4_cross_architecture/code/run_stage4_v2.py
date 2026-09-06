"""
[0831 重构] Stage 4: 跨架构 Base vs Membership (E*=median + feature_mode 双轨)

协议 (PROTOCOL_SPEC.md):
  - 只比较 base vs membership (单变量, 便于归因)
  - 4 backbone: MPNN / GIN / GAT (自定义) + DMPNN (PyG)
  - 固定 split (split_seed=2026), E*=median 训练 final
  - feature_mode 双轨 (CLI 控制)

P0-2: feature_mode 双轨
P0-1: E*=median 协议 (max_epochs 改 200, patience 30; 不再 30/8)
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

from common.constants import RESULTS_V2_DIR
from common.tasks import TASKS
from common.protocol import (SPLIT_SEED, get_final_splits, make_group_ids, assert_no_leak,
                             completeness_check, MODEL_SEEDS)
from common.graph_data import load_adj_format, load_pyg_format
from common.train_eval import (set_full_seed, train_custom_model, eval_custom_model,
                                train_pyg_model, eval_pyg_model, save_results,
                                compute_estar, verify_split_invariants)
from common.features import get_feature_meta
from common.estar_pipeline import run_gnn_estar
from models.ring_conditioned_gnn import build_model
from models.pyg_models import build_pyg_model

CONFIGS = {
    'base':       {'ring_flag': 0,  'readout': 'fixed_avg'},
    'membership': {'ring_flag': 10, 'readout': 'fixed_avg'},
}

BACKBONES_CUSTOM = ['MPNN', 'GIN', 'GAT']
BACKBONES_PYG = ['DMPNN']
ALL_BACKBONES = sorted(set(BACKBONES_CUSTOM) | set(BACKBONES_PYG))

NODE_VEC_LEN = 60
MAX_ATOMS = 75

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}


class FoldCtx:
    cache = {}
    params = None
    device = None
    task = None
    _last_data = None


def _cv_train_custom(backbone, ring_flag, tr_idx, va_idx, n_epochs, patience, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', FoldCtx.task['name'], ring_flag, feature_mode)
    data = FoldCtx.cache.get(key)
    if data is None:
        data = load_adj_format(FoldCtx.task['dataset_path'],
                                FoldCtx.task['target_col'],
                                NODE_VEC_LEN, MAX_ATOMS,
                                ring_flag_value=ring_flag, device=FoldCtx.device,
                                feature_mode=feature_mode)
        FoldCtx.cache[key] = data
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=FoldCtx.params['hidden_dim'],
                        n_conv=FoldCtx.params['n_conv_layers'],
                        n_hidden=FoldCtx.params['n_hidden_layers'],
                        p_dropout=FoldCtx.params['p_dropout'],
                        readout_mode='fixed_avg').to(FoldCtx.device)
    tr_t = torch.tensor(np.asarray(tr_idx), dtype=torch.long, device=FoldCtx.device)
    va_t = torch.tensor(np.asarray(va_idx), dtype=torch.long, device=FoldCtx.device)
    model = train_custom_model(model, FoldCtx.params, data, tr_t, va_t, FoldCtx.device,
                                n_epochs=n_epochs, patience=patience, seed=seed)
    FoldCtx._last_data = data
    return model


def _cv_eval_custom(model, idx):
    data = FoldCtx._last_data
    return eval_custom_model(model, data,
                              torch.tensor(np.asarray(idx), dtype=torch.long,
                                            device=FoldCtx.device),
                              FoldCtx.params['batch_size'], FoldCtx.device)[0]


def _fit_final_custom(backbone, ring_flag, train_idx, n_epochs, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', FoldCtx.task['name'], ring_flag, feature_mode)
    data = FoldCtx.cache[key]
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=FoldCtx.params['hidden_dim'],
                        n_conv=FoldCtx.params['n_conv_layers'],
                        n_hidden=FoldCtx.params['n_hidden_layers'],
                        p_dropout=FoldCtx.params['p_dropout'],
                        readout_mode='fixed_avg').to(FoldCtx.device)
    tr_t = torch.tensor(np.asarray(train_idx), dtype=torch.long, device=FoldCtx.device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=FoldCtx.params['learning_rate'],
                                  weight_decay=FoldCtx.params['weight_decay'])
    loss_fn = torch.nn.MSELoss()
    bs = FoldCtx.params['batch_size']
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    ring_indices = data['ring_indices']; outputs = data['outputs']
    n = len(tr_t)
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n, device=FoldCtx.device)]
        for i in range(0, n, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = model(node_mats[bi], adj_mats[bi], ring_indices[bi]).reshape(-1)
            loss = loss_fn(pred, outputs[bi].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.best_epoch = n_epochs
    FoldCtx._last_data = data
    return model


def _cv_train_pyg(backbone, ring_flag, tr_idx, va_idx, n_epochs, patience, seed, feature_mode):
    set_full_seed(seed)
    key = ('pyg', FoldCtx.task['name'], ring_flag, feature_mode)
    data_list = FoldCtx.cache.get(key)
    if data_list is None:
        data_list, _ = load_pyg_format(FoldCtx.task['dataset_path'],
                                        FoldCtx.task['target_col'],
                                        NODE_VEC_LEN, MAX_ATOMS,
                                        ring_flag_value=ring_flag, device='cpu',
                                        feature_mode=feature_mode)
        FoldCtx.cache[key] = data_list
    in_channels = data_list[0].x.size(-1)
    model = build_pyg_model(backbone, in_channels=in_channels,
                            hidden_dim=FoldCtx.params['hidden_dim'],
                            n_layers=FoldCtx.params['n_conv_layers'],
                            dropout=FoldCtx.params['p_dropout'], edge_dim=4,
                            readout_mode='fixed_avg').to(FoldCtx.device)
    model = train_pyg_model(model, FoldCtx.params, data_list,
                             np.asarray(tr_idx), np.asarray(va_idx), FoldCtx.device,
                             n_epochs=n_epochs, patience=patience, seed=seed)
    return model, data_list


def _cv_eval_pyg(model, idx, data_list):
    return eval_pyg_model(model, data_list, np.asarray(idx),
                           FoldCtx.params['batch_size'], FoldCtx.device)[0]


def _fit_final_pyg(backbone, ring_flag, train_idx, n_epochs, seed, feature_mode):
    set_full_seed(seed)
    key = ('pyg', FoldCtx.task['name'], ring_flag, feature_mode)
    data_list = FoldCtx.cache[key]
    in_channels = data_list[0].x.size(-1)
    model = build_pyg_model(backbone, in_channels=in_channels,
                            hidden_dim=FoldCtx.params['hidden_dim'],
                            n_layers=FoldCtx.params['n_conv_layers'],
                            dropout=FoldCtx.params['p_dropout'], edge_dim=4,
                            readout_mode='fixed_avg').to(FoldCtx.device)
    from torch_geometric.loader import DataLoader
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=FoldCtx.params['learning_rate'],
                                  weight_decay=FoldCtx.params['weight_decay'])
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader([data_list[i] for i in train_idx],
                         batch_size=FoldCtx.params['batch_size'], shuffle=True)
    for ep in range(1, n_epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(FoldCtx.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ring_mask = getattr(batch, 'ring_mask', None)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr,
                         ring_mask=ring_mask).reshape(-1)
            loss = loss_fn(pred, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.best_epoch = n_epochs
    return model, data_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_root', type=str,
                    default=os.path.join(RESULTS_V2_DIR, 'stage4'))
    ap.add_argument('--backbones', type=str, default=','.join(ALL_BACKBONES))
    ap.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    ap.add_argument('--seed', type=int, default=MODEL_SEEDS[0])
    ap.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    ap.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    ap.add_argument('--feature_mode', default='standard',
                    choices=['standard', 'explicit_aromaticity_ablated'])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    backbones = [b.strip() for b in args.backbones.split(',') if b.strip()]
    invalid = [b for b in backbones if b not in ALL_BACKBONES]
    if invalid:
        raise SystemExit(f"[Stage4v2] 非法 backbone: {invalid}, 可选: {ALL_BACKBONES}")
    task_names = [t.strip() for t in args.tasks.split(',') if t.strip()]
    tasks = [next(t for t in TASKS if t['name'] == n) for n in task_names]

    feat_meta = get_feature_meta(args.feature_mode)
    print(f"[Stage4v2] device={device} seed={args.seed} "
          f"n_epochs={params['n_epochs']} patience={params['patience']} "
          f"feature_mode={args.feature_mode}")
    print(f"[Stage4v2] backbones={backbones} tasks={task_names}")
    print(f"[Stage4v2] feature_meta: {json.dumps(feat_meta, indent=2)}")

    seed = args.seed
    out_root = os.path.join(args.output_root, f'seed_{seed}', args.feature_mode)
    os.makedirs(out_root, exist_ok=True)

    per_seed_rows = []
    FoldCtx.params = params; FoldCtx.device = device

    for task in tasks:
        # 固定 split
        data0 = load_adj_format(task['dataset_path'], task['target_col'],
                                 NODE_VEC_LEN, MAX_ATOMS,
                                 ring_flag_value=0, device='cpu',
                                 feature_mode=args.feature_mode)
        groups = make_group_ids(list(data0['smiles']), use_inchikey=False)
        tag = f'stage4_{args.feature_mode}'
        splits = get_final_splits(data0['n'], groups, persist=None, tag=tag)
        verify_split_invariants(splits, groups, f"Stage4 {task['name']} {args.feature_mode}")
        n_total = data0['n']
        FoldCtx.task = task

        print(f"\n[Task={task['name']}] n={n_total} dev={len(splits['train_idx'])} "
              f"test={len(splits['test_idx'])} 泄漏检查通过", flush=True)

        for backbone in backbones:
            is_pyg = backbone in BACKBONES_PYG
            FoldCtx.cache = {}
            for config_name in CONFIGS:
                ring_flag = CONFIGS[config_name]['ring_flag']
                out_dir = os.path.join(out_root, task['name'], f'{backbone}_{config_name}')
                os.makedirs(out_dir, exist_ok=True)

                # ---- CV: 自定义与 PyG 分路径 ----
                cv_rows = []
                best_epochs = []
                FoldCtx.cache = {}
                if not is_pyg:
                    res = run_gnn_estar(
                        backbone=backbone, ring_flag=ring_flag,
                        splits=splits, params=params,
                        train_cv_fn=_cv_train_custom, fit_final_fn=_fit_final_custom,
                        eval_fn=_cv_eval_custom, device=device, seed=seed,
                        max_epochs=params['n_epochs'], patience=params['patience'],
                        feature_mode=args.feature_mode,
                    )
                    cv_mae, cv_r2, cv_rmse = res.cv_mae, res.cv_r2, res.cv_rmse
                    best_epochs = res.best_epochs
                    te_r2, te_mae, te_rmse = res.test_metrics
                    tr_mae = res.train_metrics[1]
                    train_time = res.train_time_sec
                    n_ceiling = res.n_ceiling_hits
                else:
                    # PyG 路径: 手动跑 CV + E* final
                    data_list = None
                    fold_metrics = []
                    fes = []
                    for k, (tr, va) in enumerate(splits['folds']):
                        model, dl = _cv_train_pyg(backbone, ring_flag, tr, va,
                                                    params['n_epochs'],
                                                    params['patience'], seed + k,
                                                    args.feature_mode)
                        r2, mae, rmse = _cv_eval_pyg(model, va, dl)
                        fold_metrics.append({'fold': k+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
                        fes.append(getattr(model, 'best_epoch', params['n_epochs']))
                        del model
                    cv_mae = float(np.mean([m['mae'] for m in fold_metrics]))
                    cv_r2 = float(np.mean([m['r2'] for m in fold_metrics]))
                    cv_rmse = float(np.mean([m['rmse'] for m in fold_metrics]))
                    best_epochs = fes
                    estar = int(np.median(fes))
                    n_ceiling = int(sum(1 for e in fes if e >= params['n_epochs'] - 1))

                    t0 = time.time()
                    fm, dl = _fit_final_pyg(backbone, ring_flag,
                                              splits['train_idx'], estar, seed,
                                              args.feature_mode)
                    train_time = time.time() - t0
                    tr_metrics = _cv_eval_pyg(fm, splits['train_idx'], dl)
                    te_metrics = _cv_eval_pyg(fm, splits['test_idx'], dl)
                    tr_mae = tr_metrics[1]
                    te_r2, te_mae, te_rmse = te_metrics

                print(f"  -> {backbone}/{config_name} cv_mae={cv_mae:.4f} "
                      f"E*={int(np.median(best_epochs))} "
                      f"test_mae={te_mae:.4f} ceiling={n_ceiling}/5")

                row = {
                    'seed': seed, 'task': task['name'], 'model': backbone,
                    'config': config_name, 'feature_mode': args.feature_mode,
                    'ring_flag': ring_flag, 'readout': 'fixed_avg',
                    'backbone_type': 'pyg' if is_pyg else 'custom',
                    'n': n_total, 'cv_mae': round(cv_mae, 6),
                    'cv_r2': round(cv_r2, 6), 'cv_rmse': round(cv_rmse, 6),
                    'best_epochs': best_epochs,
                    'estar': int(np.median(best_epochs)),
                    'n_ceiling_hits': n_ceiling,
                    'train_mae': round(tr_mae, 6),
                    'test_r2': round(te_r2, 6),
                    'test_mae': round(te_mae, 6),
                    'test_rmse': round(te_rmse, 6),
                    'train_time_sec': round(train_time, 2),
                    'run_status': 'ok', 'error_message': '',
                }
                per_seed_rows.append(row)
                _save_per_seed(per_seed_rows, out_root)

    _save_per_seed(per_seed_rows, out_root)
    meta = {
        'protocol': 'PROTOCOL_SPEC Stage4 (P0-1 v2 + P0-2)',
        'configs': CONFIGS, 'split_seed': SPLIT_SEED, 'model_seed': seed,
        'n_epochs_cv_max': params['n_epochs'], 'patience': params['patience'],
        'feature_mode': args.feature_mode, 'feature_meta': feat_meta,
        'protocol_pipeline': 'E* = median(best_epochs); final = retrain on full dev for E* epochs',
        'note': '仅 base vs membership (fixed_avg); PyG 走 DMPNN',
    }
    meta_path = os.path.join(out_root, 'stage4_v2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[Stage4v2] meta: {meta_path}\n[per_seed]: {out_root}/per_seed_results.csv")


def _save_per_seed(rows, output_root):
    df = pd.DataFrame(rows)
    cols = ['seed', 'task', 'model', 'config', 'feature_mode', 'n',
            'cv_mae', 'cv_r2', 'cv_rmse', 'train_mae',
            'test_mae', 'test_rmse', 'test_r2',
            'train_time_sec', 'run_status', 'error_message']
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ['ring_flag', 'readout', 'backbone_type', 'best_epochs',
              'estar', 'n_ceiling_hits']:
        if c not in df.columns:
            df[c] = pd.NA
    df.to_csv(os.path.join(output_root, 'per_seed_results.csv'), index=False)


if __name__ == '__main__':
    main()
