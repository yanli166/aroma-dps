
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
"""
Anchor-based Δ-Learning Step 3-6: 三个核心模型 + Micro/Macro metrics

优化版:
  - Subtraction MPNN 只训练一次 per (task, seed), 缓存后复用于所有 fold/anchor
  - Siamese MPNN 每个 fold 训练一次 (因 train_idx 不同)
  - DeltaFP 每个 fold 训练一次

A. Stage-I absolute model subtraction
B. Difference fingerprint (XGBoost)
C. Siamese MPNN

Split: 5-fold GroupKFold (group=ring_name) + 5 seeds
Micro/Macro metrics: per-ring-type 等权平均
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import set_full_seed, DEFAULT_PARAMS, train_model, eval_model
from unified_models.mpnn.model import MPNNModel
from generalization_test.code.step5_siamese_mpnn import SiameseMPNN

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_anchor_v2')
PAIR_FILE = os.path.join(OUTPUT_DIR, 'anchor_pair_dataset.csv')

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10
SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5

TASK_COL_MAP = {
    'HOMA':      ('HOMA',    'homa_value'),
    'NICS_1zz':  ('NICS_ZZ', 'NICS_value'),
    'MBCO':      ('MBCO',    'mbco_value'),
}


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n < 2 or np.std(y_true) < 1e-8:
        r2 = float('nan')
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    try:
        sp, _ = spearmanr(y_true, y_pred)
        sp = sp if not np.isnan(sp) else 0.0
    except Exception:
        sp = 0.0
    try:
        kd, _ = kendalltau(y_true, y_pred)
        kd = kd if not np.isnan(kd) else 0.0
    except Exception:
        kd = 0.0
    if n < 2:
        pairwise_acc = float('nan')
    else:
        dt = y_true[:, None] - y_true[None, :]
        dp = y_pred[:, None] - y_pred[None, :]
        iu = np.triu_indices(n, k=1)
        signs = np.sign(dt[iu]) * np.sign(dp[iu])
        pairwise_acc = np.mean(signs > 0)
    return {
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'spearman': sp, 'kendall': kd, 'pairwise_acc': pairwise_acc,
        'n': n,
    }


def compute_macro_metrics(df_pred, ring_col='scaffold_id'):
    macro = {}
    per_ring = []
    for ring, sub in df_pred.groupby(ring_col):
        if len(sub) < 2:
            continue
        m = compute_metrics(sub['delta_A'].values, sub['pred_delta'].values)
        m['ring_type'] = ring
        per_ring.append(m)
    df_per = pd.DataFrame(per_ring)
    if len(df_per) == 0:
        return {k: float('nan') for k in ['r2', 'mae', 'rmse', 'spearman', 'kendall', 'pairwise_acc']}, df_per
    for col in ['r2', 'mae', 'rmse', 'spearman', 'kendall', 'pairwise_acc']:
        vals = df_per[col].dropna()
        macro[col] = vals.mean() if len(vals) > 0 else float('nan')
    return macro, df_per


# ============================================================
# Data loading helpers
# ============================================================

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def load_molecules_for_graph(df_pairs, role, target_col, device):
    smiles_col = f'smiles_{role}'
    a_col = f'A_{role}'
    ring_atoms_col = f'target_ring_{role}'

    tmp_path = f'/tmp/anchor_{role}_{os.getpid()}_{np.random.randint(1e6)}.csv'
    out_df = df_pairs[[smiles_col, a_col, 'ring_id', ring_atoms_col]].copy()
    out_df.columns = ['smiles', target_col, 'Ring_ID', 'Ring_Atoms']
    out_df['atom_on_ring'] = out_df['Ring_Atoms']
    out_df['New_ID'] = range(len(out_df))
    out_df.to_csv(tmp_path, index=False)

    data = load_adj_format(tmp_path, target_col, NVL, MAX_ATOMS,
                           ring_flag_value=RING_FLAG_VALUE, device=device)
    os.remove(tmp_path)
    return data


# ============================================================
# Baseline A: Absolute Model Subtraction (cached training)
# ============================================================

def train_mpnn_cached(task_info, device, seed, cache):
    """训练 MPNN 并缓存 (同一 task+seed 只训练一次)"""
    cache_key = (task_info['name'], seed)
    if cache_key in cache:
        return cache[cache_key]

    target_col = task_info['target_col']
    train_data = load_adj_format(task_info['dataset_path'], target_col, NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n_train = train_data['n']
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_train)
    n_final_train = int(0.875 * n_train)
    ti = torch.tensor(perm[:n_final_train], device=device)
    vi = torch.tensor(perm[n_final_train:], device=device)

    params = dict(DEFAULT_PARAMS)
    model = train_model('MPNN', params, train_data, ti, vi,
                        device, params['n_epochs'], params['patience'], seed)
    cache[cache_key] = model
    return model


def predict_subtraction(model, df_pairs, target_col, device):
    """用已训练的 MPNN 预测所有 pair 的 delta = A_sub_pred - A_ref_pred"""
    ref_data = load_molecules_for_graph(df_pairs, 'ref', target_col, device)
    sub_data = load_molecules_for_graph(df_pairs, 'sub', target_col, device)

    all_idx = torch.arange(ref_data['n'], device=device)
    _, _, _, pred_ref, _ = eval_model(model, ref_data, all_idx, batch_size=64, device=device)
    _, _, _, pred_sub, _ = eval_model(model, sub_data, all_idx, batch_size=64, device=device)
    pred_delta = pred_sub - pred_ref

    del ref_data, sub_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_delta


# ============================================================
# Baseline B: Difference Fingerprint
# ============================================================

def run_delta_fp(df_pairs, seed, train_idx, test_idx):
    import xgboost as xgb
    delta_fps = []
    for _, row in df_pairs.iterrows():
        fp_ref = smiles_to_morgan(row['smiles_ref'])
        fp_sub = smiles_to_morgan(row['smiles_sub'])
        delta_fps.append(fp_sub - fp_ref)
    X = np.array(delta_fps, dtype=np.float32)
    y = df_pairs['delta_A'].values.astype(np.float32)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=4
    )
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])

    df_pred = df_pairs.iloc[test_idx].copy().reset_index(drop=True)
    df_pred['pred_delta'] = y_pred
    return df_pred


# ============================================================
# Model C: Siamese MPNN
# ============================================================

def train_siamese_anchor(model, ref_data, sub_data, y_delta,
                         train_idx, val_idx, device, n_epochs=200, patience=30, seed=42):
    set_full_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    ref_nm, ref_am = ref_data['node_mats'], ref_data['adj_mats']
    sub_nm, sub_am = sub_data['node_mats'], sub_data['adj_mats']
    y_tensor = torch.tensor(y_delta, dtype=torch.float32, device=device)

    best_val, best_state, pcount = float('inf'), None, 0
    batch_size = 64

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        for i in range(0, len(train_idx), batch_size):
            batch_pairs = train_idx[perm[i:i + batch_size]]
            if len(batch_pairs) < 2:
                continue
            idx = torch.tensor(batch_pairs, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            preds = model(ref_nm[idx], ref_am[idx], sub_nm[idx], sub_am[idx])
            loss = loss_fn(preds, y_tensor[batch_pairs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(val_idx), batch_size):
                batch_pairs = val_idx[i:i + batch_size]
                if len(batch_pairs) < 2:
                    continue
                idx = torch.tensor(batch_pairs, device=device, dtype=torch.long)
                vp.append(model(ref_nm[idx], ref_am[idx], sub_nm[idx], sub_am[idx]))
            vp = torch.cat(vp) if vp else torch.zeros(0, device=device)
            vloss = loss_fn(vp, y_tensor[val_idx]).item() if len(vp) > 0 else float('inf')

        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_siamese_anchor(model, ref_data, sub_data, eval_idx, device, batch_size=64):
    model.eval()
    ref_nm, ref_am = ref_data['node_mats'], ref_data['adj_mats']
    sub_nm, sub_am = sub_data['node_mats'], sub_data['adj_mats']
    preds = []
    for i in range(0, len(eval_idx), batch_size):
        batch_pairs = eval_idx[i:i + batch_size]
        if len(batch_pairs) < 1:
            continue
        idx = torch.tensor(batch_pairs, device=device, dtype=torch.long)
        preds.append(model(ref_nm[idx], ref_am[idx], sub_nm[idx], sub_am[idx]))
    return torch.cat(preds).cpu().numpy() if preds else np.array([])


# ============================================================
# Split: 5-fold GroupKFold
# ============================================================

def make_splits(groups, n_folds=N_FOLDS, seed=42):
    gkf = GroupKFold(n_splits=n_folds)
    splits = []
    for train_idx, test_idx in gkf.split(np.arange(len(groups)), groups=groups):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_idx))
        n_val = max(2, int(0.125 * len(train_idx)))
        val_idx = train_idx[perm[:n_val]]
        train_idx_inner = train_idx[perm[n_val:]]
        splits.append((train_idx_inner, val_idx, test_idx))
    return splits


# ============================================================
# Main (optimized)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--models', type=str, default='subtraction,deltafp,siamese')
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    parser.add_argument('--anchors', type=str, default='F,Cl,OMe')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df_all = pd.read_csv(PAIR_FILE)
    print(f"Loaded {len(df_all)} anchor pairs")

    task_map = {t['name']: t for t in TASKS}
    models_to_run = args.models.split(',')
    tasks_to_run = args.tasks.split(',')
    anchors_to_run = args.anchors.split(',')

    all_results = []
    all_per_ring = []
    all_predictions = []
    mpnn_cache = {}  # (task_name, seed) → trained MPNN model

    for task_name in tasks_to_run:
        task = task_map[task_name]
        target_col = task['target_col']

        # 预加载每个 anchor 的图数据 (用于 siamese)
        anchor_data = {}
        if 'siamese' in models_to_run:
            for anchor in anchors_to_run:
                df_pairs = df_all[(df_all['task'] == task_name) & (df_all['anchor'] == anchor)].copy().reset_index(drop=True)
                ref_data = load_molecules_for_graph(df_pairs, 'ref', target_col, device)
                sub_data = load_molecules_for_graph(df_pairs, 'sub', target_col, device)
                anchor_data[anchor] = {'df': df_pairs, 'ref': ref_data, 'sub': sub_data}
                print(f"  [{task_name}|{anchor}] Loaded graphs: {ref_data['n']} pairs")

        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"Task={task_name} | seed={seed}")
            print(f"{'='*70}")

            set_full_seed(seed)

            # 训练 MPNN (缓存, 只训练一次)
            if 'subtraction' in models_to_run:
                print(f"  Training MPNN for subtraction baseline...")
                t0 = time.time()
                mpnn_model = train_mpnn_cached(task, device, seed, mpnn_cache)
                print(f"  MPNN trained (cached) in {time.time()-t0:.0f}s")

            for anchor in anchors_to_run:
                print(f"\n  --- Anchor={anchor} ---")
                df_pairs = anchor_data[anchor]['df'] if 'siamese' in models_to_run else \
                    df_all[(df_all['task'] == task_name) & (df_all['anchor'] == anchor)].copy().reset_index(drop=True)

                groups = df_pairs['scaffold_id'].values
                splits = make_splits(groups, N_FOLDS, seed)

                # Subtraction: 预测所有 pair 一次 (复用)
                pred_delta_sub = None
                if 'subtraction' in models_to_run:
                    pred_delta_sub = predict_subtraction(mpnn_model, df_pairs, target_col, device)

                for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits):
                    print(f"    Fold {fold_id+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

                    for model_name in models_to_run:
                        print(f"      [{model_name}]", end=' ', flush=True)
                        t0 = time.time()
                        try:
                            if model_name == 'subtraction':
                                df_pred = df_pairs.iloc[test_idx].copy().reset_index(drop=True)
                                df_pred['pred_delta'] = pred_delta_sub[test_idx]
                            elif model_name == 'deltafp':
                                df_pred = run_delta_fp(df_pairs, seed, train_idx, test_idx)
                            elif model_name == 'siamese':
                                ref_data = anchor_data[anchor]['ref']
                                sub_data = anchor_data[anchor]['sub']
                                set_full_seed(seed)
                                model = SiameseMPNN(
                                    node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2,
                                    p_dropout=0.2, mode='label'
                                )
                                y_delta = df_pairs['delta_A'].values.astype(np.float32)
                                model = train_siamese_anchor(
                                    model, ref_data, sub_data, y_delta,
                                    train_idx, val_idx, device, n_epochs=200, patience=30, seed=seed
                                )
                                pred_delta = predict_siamese_anchor(model, ref_data, sub_data, test_idx, device)
                                df_pred = df_pairs.iloc[test_idx].copy().reset_index(drop=True)
                                df_pred['pred_delta'] = pred_delta
                                del model
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                continue

                            micro = compute_metrics(df_pred['delta_A'].values, df_pred['pred_delta'].values)
                            macro, df_per_ring = compute_macro_metrics(df_pred)

                            elapsed = time.time() - t0
                            print(f"micro: R²={micro['r2']:.4f} MAE={micro['mae']:.4f} "
                                  f"Sp={micro['spearman']:.4f} Pw={micro['pairwise_acc']:.4f} "
                                  f"| macro: R²={macro['r2']:.4f} MAE={macro['mae']:.4f} "
                                  f"Sp={macro['spearman']:.4f} ({elapsed:.0f}s)")

                            res = {
                                'task': task_name, 'anchor': anchor, 'model': model_name,
                                'seed': seed, 'fold': fold_id,
                                'n_train': len(train_idx), 'n_test': len(test_idx),
                                'elapsed_sec': elapsed,
                            }
                            for k, v in micro.items():
                                res[f'micro_{k}'] = v
                            for k, v in macro.items():
                                res[f'macro_{k}'] = v
                            all_results.append(res)

                            df_per_ring['task'] = task_name
                            df_per_ring['anchor'] = anchor
                            df_per_ring['model'] = model_name
                            df_per_ring['seed'] = seed
                            df_per_ring['fold'] = fold_id
                            all_per_ring.append(df_per_ring)

                            df_pred['task'] = task_name
                            df_pred['anchor'] = anchor
                            df_pred['model'] = model_name
                            df_pred['seed'] = seed
                            df_pred['fold'] = fold_id
                            all_predictions.append(df_pred)

                        except Exception as e:
                            import traceback
                            print(f"ERROR: {e}")
                            traceback.print_exc()

        # 释放图数据
        anchor_data.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_raw.csv'), index=False)

    df_per = pd.concat(all_per_ring, ignore_index=True)
    df_per.to_csv(os.path.join(OUTPUT_DIR, 'per_ringtype_results.csv'), index=False)

    # 聚合 model_comparison
    metric_cols = [c for c in df_results.columns if c.startswith('micro_') or c.startswith('macro_')]
    agg_rows = []
    for (task, anchor, model), grp in df_results.groupby(['task', 'anchor', 'model']):
        row = {'task': task, 'anchor': anchor, 'model': model, 'n_runs': len(grp)}
        for c in metric_cols:
            vals = grp[c].dropna()
            row[f'{c}_mean'] = vals.mean() if len(vals) > 0 else float('nan')
            row[f'{c}_std'] = vals.std(ddof=1) if len(vals) > 1 else float('nan')
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

    # macro ringtype metrics
    macro_agg = df_per.groupby(['task', 'anchor', 'model', 'ring_type']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        n_samples=('n', 'mean'),
    ).reset_index()
    macro_agg.to_csv(os.path.join(OUTPUT_DIR, 'macro_ringtype_metrics.csv'), index=False)

    # 保存预测
    df_preds = pd.concat(all_predictions, ignore_index=True)
    df_preds.to_csv(os.path.join(OUTPUT_DIR, 'all_predictions.csv'), index=False)

    print(f"\n{'='*70}")
    print("Done. Files saved:")
    for f in ['model_comparison.csv', 'macro_ringtype_metrics.csv', 'per_ringtype_results.csv', 'all_predictions.csv']:
        print(f"  {OUTPUT_DIR}/{f}")


if __name__ == '__main__':
    main()
