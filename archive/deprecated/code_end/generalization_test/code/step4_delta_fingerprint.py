
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
Step 4: Baseline B — Difference Fingerprint

对每个 pair:
  FP_i = Morgan(smiles_i)
  FP_j = Morgan(smiles_j)
  delta_FP = FP_i - FP_j

使用 XGBoost / CatBoost 预测 delta_A。

要求: 与 Step 3 完全一致的 scaffold-grouped splits 和 seeds。

输出:
  results/lunci10_delta_learning/03_delta_fingerprint/
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
PAIR_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/01_pair_dataset')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/03_delta_fingerprint')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
TASKS = ['HOMA', 'NICS_1zz', 'MBCO']

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def smiles_to_morgan(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """SMILES → Morgan fingerprint numpy array"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_delta_fp(smiles_i, smiles_j):
    """计算 delta fingerprint = FP_i - FP_j"""
    fp_i = smiles_to_morgan(smiles_i)
    fp_j = smiles_to_morgan(smiles_j)
    return fp_i - fp_j


def compute_metrics_full(y_true, y_pred):
    """计算全部指标: R², MAE, RMSE, Spearman, Kendall, pairwise accuracy"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)

    # Pairwise direction accuracy
    correct = 0
    total = 0
    for td, pd in zip(y_true, y_pred):
        if td == 0 or pd == 0:
            continue
        if np.sign(td) == np.sign(pd):
            correct += 1
        total += 1
    pairwise_acc = correct / total if total > 0 else np.nan

    return {
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'spearman': rho, 'kendall': tau,
        'pairwise_acc': pairwise_acc,
    }


def main():
    print("=" * 70)
    print("Step 4: Baseline B — Difference Fingerprint")
    print("=" * 70)

    # 尝试导入 XGBoost 和 CatBoost
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False
        print("Warning: xgboost not available, will skip XGBoost")

    try:
        from catboost import CatBoostRegressor
        has_cat = True
    except ImportError:
        has_cat = False
        print("Warning: catboost not available, will skip CatBoost")

    if not has_xgb and not has_cat:
        print("Error: Neither xgboost nor catboost available!")
        return

    all_metrics = []

    for task_name in TASKS:
        print(f"\n{'='*70}")
        print(f"# {task_name}")
        print(f"{'='*70}")

        # 加载 pair dataset
        pair_file = os.path.join(PAIR_DIR, f'pair_dataset_{task_name.lower().replace("_1zz","")}.csv')
        df_pairs = pd.read_csv(pair_file)
        print(f"  Loaded {len(df_pairs)} pairs")

        # 计算 delta fingerprint
        print("  Computing delta fingerprints...")
        delta_fps = []
        for _, row in df_pairs.iterrows():
            dfp = compute_delta_fp(row['smiles_i'], row['smiles_j'])
            delta_fps.append(dfp)

        X = np.array(delta_fps, dtype=np.float32)
        y = df_pairs['delta_A'].values.astype(np.float32)
        groups = df_pairs['scaffold_id'].values  # scaffold-grouped split

        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  Unique scaffolds (groups): {len(np.unique(groups))}")

        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")

            # scaffold-grouped split: 80/20 train/test
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(gss.split(X, y, groups))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 5-fold CV within train for early stopping
            cv = GroupKFold(n_splits=5)

            for model_name, model_cls, params in [
                ('XGBoost', xgb.XGBRegressor if has_xgb else None,
                 {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': seed,
                  'n_jobs': 4} if has_xgb else None),
                ('CatBoost', CatBoostRegressor if has_cat else None,
                 {'iterations': 300, 'depth': 6, 'learning_rate': 0.05,
                  'random_seed': seed, 'verbose': 0} if has_cat else None),
            ]:
                if model_cls is None:
                    continue

                print(f"    Training {model_name}...")
                model = model_cls(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = compute_metrics_full(y_test, y_pred)
                metrics['task'] = task_name
                metrics['seed'] = seed
                metrics['model'] = model_name
                metrics['n_train'] = len(train_idx)
                metrics['n_test'] = len(test_idx)

                all_metrics.append(metrics)

                print(f"    {model_name}: R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | "
                      f"Spearman={metrics['spearman']:.4f} | Kendall={metrics['kendall']:.4f} | "
                      f"Pairwise={metrics['pairwise_acc']:.4f}")

    # 保存结果
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'delta_fp_per_seed.csv'), index=False)

    # 聚合
    agg = df_metrics.groupby(['task', 'model']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg.to_csv(os.path.join(OUTPUT_DIR, 'delta_fp_agg.csv'), index=False)

    # 打印汇总
    print(f"\n{'='*70}")
    print("Baseline B: Difference Fingerprint (mean±SD over 5 seeds)")
    print(f"{'='*70}")
    for _, row in agg.iterrows():
        print(f"\n  [{row['task']} / {row['model']}]")
        print(f"    Δ R²       = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
        print(f"    Δ MAE      = {row['mae_mean']:.4f} ± {row['mae_std']:.4f}")
        print(f"    Δ RMSE     = {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}")
        print(f"    Spearman ρ = {row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}")
        print(f"    Kendall τ  = {row['kendall_mean']:.4f} ± {row['kendall_std']:.4f}")
        print(f"    Pairwise   = {row['pairwise_mean']:.4f} ± {row['pairwise_std']:.4f}")

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
