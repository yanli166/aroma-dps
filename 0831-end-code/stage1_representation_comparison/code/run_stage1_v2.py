"""
[0831 重构] Stage 1: 香草 GNN + 传统 ML 基线 (E*=median 协议)

协议 (PROTOCOL_SPEC.md):
  1. 固定 20% holdout (split_seed=2026, 与 model seed 解耦)
  2. 80% development 内 5-fold Group CV
  3. 每折按 val-MAE 早停, 记录 best_epoch
  4. E* = median(best_epochs)
  5. 在完整 80% development 上按 E* 重训 final, 在固定 20% test 上评估
  6. 不再使用 15 epoch / 5 patience 的 dry-run 默认限制
  7. feature_mode 双轨 (default 'standard'; CLI 可切 'explicit_aromaticity_ablated')
  8. 所有结果 CSV 含 feature_mode 字段 + feature metadata JSON
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
                             completeness_check, MODEL_SEEDS)
from common.tasks import TASKS, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from common.train_eval import (set_full_seed, train_custom_model, eval_custom_model,
                                compute_estar, verify_split_invariants)
from common.features import (build_fingerprint_matrix, get_feature_meta,
                             MACCS_DIM, MORGAN_BITS,
                             MOL_DESC_DIM_AROM_ABLATED, RING_DESC_DIM_AROM_ABLATED,
                             MOL_DESC_DIM, RING_DESC_DIM, FEAT_DIM, FEAT_DIM_AROM_ABLATED)
from common.estar_pipeline import run_gnn_estar
from models.ring_conditioned_gnn import build_model

GNNS = ['MPNN', 'GIN', 'GAT']
ML_MODELS = ['RF', 'MLP']
NODE_VEC_LEN, MAX_ATOMS = 60, 75
BIN_DIM = MACCS_DIM + MORGAN_BITS


def load_fixed_splits(task, device, feature_mode='standard', tag=None):
    data = load_adj_format(task['dataset_path'], task['target_col'], NODE_VEC_LEN, MAX_ATOMS,
                           ring_flag_value=0, device=device, feature_mode=feature_mode)
    groups = make_group_ids(data['smiles'])
    tag = tag or f'stage1_{feature_mode}'
    splits = get_final_splits(data['n'], groups, persist=SPLITS_DIR, tag=tag)
    verify_split_invariants(splits, groups, f"Stage1 {task['name']} {feature_mode}")
    return data, groups, splits


# ---------------- GNN runner for E* pipeline ----------------
def _gnn_train_cv(backbone, ring_flag, tr_idx, va_idx, n_epochs, patience, seed, feature_mode):
    """加载对应 ring_flag 数据并跑一折 CV 训练 (含 early-stop)。"""
    set_full_seed(seed)
    # NOTE: feature_mode 影响 node_mat 与 edge_attr, 由 load_adj_format 内部处理
    return _gnn_run_single_fold(backbone, ring_flag, tr_idx, va_idx,
                                 n_epochs, patience, seed, feature_mode)


def _gnn_run_single_fold(backbone, ring_flag, tr_idx, va_idx,
                          n_epochs, patience, seed, feature_mode):
    """一折 CV 训练 (辅助函数)。"""
    # 因为 device/smiles 全局唯一, 这里用缓存避免重 load
    key = ('adj', _gnn_run_single_fold._task['name'], ring_flag, feature_mode)
    data = _gnn_run_single_fold._cache.get(key)
    if data is None:
        data = load_adj_format(_gnn_run_single_fold._task['dataset_path'],
                                _gnn_run_single_fold._task['target_col'],
                                NODE_VEC_LEN, MAX_ATOMS,
                                ring_flag_value=ring_flag, device=_gnn_run_single_fold._device,
                                feature_mode=feature_mode)
        _gnn_run_single_fold._cache[key] = data
    params = _gnn_run_single_fold._params
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=params['hidden_dim'],
                        n_conv=params['n_conv_layers'],
                        n_hidden=params['n_hidden_layers'],
                        p_dropout=params['p_dropout'],
                        readout_mode='fixed_avg').to(_gnn_run_single_fold._device)
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=_gnn_run_single_fold._device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=_gnn_run_single_fold._device)
    model = train_custom_model(model, params, data, tr_t, va_t,
                                _gnn_run_single_fold._device,
                                n_epochs=n_epochs, patience=patience, seed=seed)
    _gnn_run_single_fold._last_data = data
    return model
_gnn_run_single_fold._cache = {}
_gnn_run_single_fold._params = None
_gnn_run_single_fold._device = None
_gnn_run_single_fold._task = None
_gnn_run_single_fold._last_data = None


def _gnn_eval(model, idx):
    data = _gnn_run_single_fold._last_data
    return eval_custom_model(model, data, torch.tensor(idx, dtype=torch.long,
                                                       device=_gnn_run_single_fold._device),
                              _gnn_run_single_fold._params['batch_size'],
                              _gnn_run_single_fold._device)[0]


def _gnn_fit_final(backbone, ring_flag, train_idx, n_epochs, seed, feature_mode):
    """固定 E* epochs 在完整 dev 上训练, 无 early-stop。"""
    set_full_seed(seed)
    key = ('adj', _gnn_run_single_fold._task['name'], ring_flag, feature_mode)
    data = _gnn_run_single_fold._cache[key]
    params = _gnn_run_single_fold._params
    device = _gnn_run_single_fold._device
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=params['hidden_dim'],
                        n_conv=params['n_conv_layers'],
                        n_hidden=params['n_hidden_layers'],
                        p_dropout=params['p_dropout'],
                        readout_mode='fixed_avg').to(device)
    tr_t = torch.tensor(np.asarray(train_idx), dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                  weight_decay=params['weight_decay'])
    loss_fn = torch.nn.MSELoss()
    bs = params['batch_size']
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    ring_indices = data['ring_indices']; outputs = data['outputs']
    n = len(tr_t)
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n, device=device)]
        for i in range(0, n, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = model(node_mats[bi], adj_mats[bi], ring_indices[bi]).reshape(-1)
            loss = loss_fn(pred, outputs[bi].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.best_epoch = n_epochs
    _gnn_run_single_fold._last_data = data
    return model


# ---------------- (a) 传统 ML ----------------
def build_feature_frame(data, feature_mode='standard'):
    clean_path = clean_dataset_csv(_TASK['dataset_path'], _TASK['target_col'])
    df = pd.read_csv(clean_path)
    df = df.iloc[:data['n']].copy()
    df['smiles'] = list(data['smiles'])
    X, valid = build_fingerprint_matrix(df['smiles'].tolist(), df=df,
                                         feature_mode=feature_mode)
    assert valid.all() and X.shape[0] == data['n'], "指纹与样本数不一致"
    return X, df


_TASK = {}


def ml_cv_and_final(mname, X, y, splits, feature_mode='standard'):
    X_bin = X[:, :BIN_DIM]
    X_con_dim = (FEAT_DIM_AROM_ABLATED if feature_mode == 'explicit_aromaticity_ablated' else FEAT_DIM) - BIN_DIM
    X_con = X[:, BIN_DIM:BIN_DIM + X_con_dim]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    if mname == 'RF':
        def model_z():
            return Pipeline([('model', RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=0, min_samples_leaf=2))])
    else:
        def model_z():
            ct = ColumnTransformer([
                ('bin', 'passthrough', np.arange(X_bin.shape[1])),
                ('con', StandardScaler(), np.arange(X_bin.shape[1], X_bin.shape[1] + X_con.shape[1]))])
            return Pipeline([('ct', ct),
                             ('model', MLPRegressor(hidden_layer_sizes=(256, 128),
                                                     max_iter=600,
                                                     random_state=0,
                                                     early_stopping=True))])

    cv_folds = []
    for fold, (tr, va) in enumerate(splits['folds']):
        m = model_z(); m.fit(X[tr], y[tr]); p = m.predict(X[va])
        r2, mae, rmse = compute_metrics(y[va], p)
        cv_folds.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        del m
    cv = {'cv_mae': float(np.mean([r['mae'] for r in cv_folds])),
          'cv_r2': float(np.mean([r['r2'] for r in cv_folds]))}
    m = model_z()
    m.fit(X[splits['train_idx']], y[splits['train_idx']])
    p = m.predict(X[splits['test_idx']])
    te_r2, te_mae, te_rmse = compute_metrics(y[splits['test_idx']], p)
    return cv, cv_folds, {'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse}


# ---------------- main ----------------
def main():
    global _TASK
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=os.path.join(RESULTS_V2_DIR, 'stage1'))
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--gnn_backbones', default=','.join(GNNS))
    ap.add_argument('--ml_models', default=','.join(ML_MODELS))
    ap.add_argument('--n_epochs', type=int, default=200,
                    help='CV max epochs; 取消 15 epoch dry-run 默认')
    ap.add_argument('--patience', type=int, default=30,
                    help='CV early-stop patience; 取消 5 patience dry-run 默认')
    ap.add_argument('--seed', type=int, default=MODEL_SEEDS[0])
    ap.add_argument('--feature_mode', default='standard',
                    choices=['standard', 'explicit_aromaticity_ablated'])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | feature_mode: {args.feature_mode} | '
          f'n_epochs={args.n_epochs} patience={args.patience}')

    backbones = args.gnn_backbones.split(',')
    ml_models = args.ml_models.split(',')
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == x) for x in args.tasks.split(',')]

    feat_meta = get_feature_meta(args.feature_mode)
    print(f"feature_meta: {json.dumps(feat_meta, indent=2, ensure_ascii=False)}")

    params = {'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
              'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
              'weight_decay': 1e-5, 'n_epochs': args.n_epochs,
              'patience': args.patience}

    seed = args.seed
    run_dir = os.path.join(args.output_dir, f'seed_{seed}', args.feature_mode)
    os.makedirs(run_dir, exist_ok=True)

    per_seed, cv_save, final_save = [], [], []

    for task in task_list:
        _TASK = task
        tname = task['name']
        print(f"\n{'='*72}\nTask: {tname} | seed={seed} | feature_mode={args.feature_mode}\n{'='*72}")

        data, groups, splits = load_fixed_splits(task, device, feature_mode=args.feature_mode)
        n_total = data['n']

        # (b) 香草 GNN
        _gnn_run_single_fold._cache = {}
        _gnn_run_single_fold._params = params
        _gnn_run_single_fold._device = device
        _gnn_run_single_fold._task = task
        for bb in backbones:
            res = run_gnn_estar(
                backbone=bb, ring_flag=0, splits=splits, params=params,
                train_cv_fn=_gnn_train_cv, fit_final_fn=_gnn_fit_final,
                eval_fn=_gnn_eval, device=device, seed=seed,
                max_epochs=params['n_epochs'], patience=params['patience'],
                feature_mode=args.feature_mode,
            )
            print(f"  GNN[{bb}] {tname}: cv_mae={res.cv_mae:.4f} "
                  f"E*={res.estar} best_epochs={res.best_epochs} "
                  f"test_mae={res.test_metrics[1]:.4f}")
            per_seed.append({
                'seed': seed, 'task': tname, 'model': f'GNN_{bb}',
                'config': 'base', 'feature_mode': args.feature_mode,
                'n': n_total,
                'cv_mae': res.cv_mae, 'cv_r2': res.cv_r2, 'cv_rmse': res.cv_rmse,
                'best_epochs': res.best_epochs, 'estar': res.estar,
                'n_ceiling_hits': res.n_ceiling_hits,
                'test_r2': res.test_metrics[0],
                'test_mae': res.test_metrics[1],
                'test_rmse': res.test_metrics[2],
                'train_mae': res.train_metrics[1],
                'train_time_sec': res.train_time_sec,
                'run_status': 'ok', 'error_message': '',
            })
            cv_save.append({
                'task': tname, 'model': f'GNN_{bb}', 'config': 'base',
                'feature_mode': args.feature_mode,
                'n': n_total, 'cv_mae': res.cv_mae,
                'cv_r2': res.cv_r2, 'cv_rmse': res.cv_rmse,
                'estar': res.estar, 'best_epochs': res.best_epochs,
            })
            final_save.append({
                'task': tname, 'model': f'GNN_{bb}', 'config': 'base',
                'feature_mode': args.feature_mode,
                'n': n_total, 'cv_mae': res.cv_mae,
                'test_mae': res.test_metrics[1],
                'test_rmse': res.test_metrics[2],
                'test_r2': res.test_metrics[0],
                'estar': res.estar, 'n_ceiling_hits': res.n_ceiling_hits,
                'best_epochs': res.best_epochs,
            })

        # (a) 传统 ML (CPU)
        y = data['outputs'].cpu().numpy()
        X, _df = build_feature_frame(data, feature_mode=args.feature_mode)
        for mm in ml_models:
            cv, cv_rows, fin = ml_cv_and_final(mm, X, y, splits, feature_mode=args.feature_mode)
            print(f"  ML[{mm}] {tname}: cv_mae={cv['cv_mae']:.4f} test_mae={fin['test_mae']:.4f}")
            per_seed.append({
                'seed': seed, 'task': tname, 'model': f'ML_{mm}',
                'config': 'base', 'feature_mode': args.feature_mode,
                'n': n_total,
                'cv_mae': cv['cv_mae'], 'cv_r2': cv['cv_r2'],
                'cv_rmse': float('nan'),
                'best_epochs': '[]', 'estar': 0, 'n_ceiling_hits': 0,
                'test_r2': fin['test_r2'],
                'test_mae': fin['test_mae'], 'test_rmse': fin['test_rmse'],
                'train_mae': float('nan'),
                'train_time_sec': float('nan'),
                'run_status': 'ok', 'error_message': '',
            })
            cv_save.append({
                'task': tname, 'model': f'ML_{mm}', 'config': 'base',
                'feature_mode': args.feature_mode,
                'n': n_total, 'cv_mae': cv['cv_mae'], 'cv_r2': cv['cv_r2'],
            })
            final_save.append({
                'task': tname, 'model': f'ML_{mm}', 'config': 'base',
                'feature_mode': args.feature_mode,
                'n': n_total, 'cv_mae': cv['cv_mae'],
                'test_mae': fin['test_mae'], 'test_rmse': fin['test_rmse'],
                'test_r2': fin['test_r2'],
            })

    per_df = pd.DataFrame(per_seed); cv_df = pd.DataFrame(cv_save); final_df = pd.DataFrame(final_save)
    per_df.to_csv(os.path.join(run_dir, 'per_seed_results.csv'), index=False)
    cv_df.to_csv(os.path.join(run_dir, 'cv_results.csv'), index=False)
    final_df.to_csv(os.path.join(run_dir, 'final_test_results.csv'), index=False)

    models = [f'GNN_{b}' for b in backbones] + [f'ML_{m}' for m in ml_models]
    exp = pd.MultiIndex.from_product([models, [t['name'] for t in task_list], [seed]],
                                     names=['model', 'task', 'seed'])
    completeness_check(pd.MultiIndex.from_frame(per_df[['model', 'task', 'seed']]), exp,
                       f'Stage1 (seed={seed}, feature_mode={args.feature_mode})')

    meta = {
        'protocol': 'PROTOCOL_SPEC Stage1 (P0-1 v2 + P0-2)',
        'split_seed': 2026, 'model_seed': seed,
        'feature_mode': args.feature_mode,
        'feature_meta': feat_meta,
        'n_epochs_cv_max': params['n_epochs'],
        'patience': params['patience'],
        'gnn_backbones': backbones, 'ml_models': ml_models,
        'protocol_pipeline': 'E* = median(best_epochs); final = retrain on full dev for E* epochs',
        'note': 'Stage1 是 base 基线; ring_flag=0 (no target-ring membership injection)',
    }
    meta_path = os.path.join(run_dir, 'stage1_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}\nStage1 汇总 (seed={seed}, feature_mode={args.feature_mode})\n{'='*72}")
    print(final_df[['task', 'model', 'feature_mode', 'cv_mae', 'estar',
                     'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
    print(f"\n[完整性] Stage1 校验通过 (缺失 0)")
    print(f"[feature_mode] {args.feature_mode}: {feat_meta['fingerprint_generation']}")
    print(f"[results] {run_dir}\n[meta] {meta_path}")


if __name__ == '__main__':
    main()
