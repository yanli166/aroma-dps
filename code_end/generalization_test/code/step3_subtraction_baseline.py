
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
Step 3: Baseline A — Absolute Model Subtraction

使用 Stage I 最优的单分子 MPNN (ring_flag=10), 不训练特殊 pair 模型。
分别预测 A_hat_i, A_hat_j, 然后 delta_hat = A_hat_i - A_hat_j。
评估 delta_hat vs delta_true。

5 seeds: 42, 123, 456, 789, 2024

输出:
  results/lunci10_delta_learning/02_subtraction_baseline/
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau
from itertools import combinations

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, DEFAULT_SEED, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, train_model, eval_model, set_full_seed

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/02_subtraction_baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIR_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/01_pair_dataset')

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10
SEEDS = [42, 123, 456, 789, 2024]

TASK_COL_MAP = {
    'HOMA':      ('HOMA',    'homa_value'),
    'NICS_1zz':  ('NICS_ZZ', 'NICS_value'),
    'MBCO':      ('MBCO',    'mbco_value'),
}


def load_lunci10_for_pred(task_name, task_info, device):
    """加载 lunci10 测试数据用于预测"""
    l10_col, target_col = TASK_COL_MAP[task_name]
    df_l10 = pd.read_csv(os.path.join(PROJ_ROOT, 'data1_end', 'lunci10-test.csv'),
                         encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=[l10_col, 'SMILES']).reset_index(drop=True)

    # 保存为临时 CSV 供 load_adj_format 使用 (需要 atom_on_ring 列)
    tmp_path = os.path.join('/tmp', f'lunci10_{task_name}_step3.csv')
    out_df = df_l10[['SMILES', l10_col, 'Ring_ID', 'Ring_Atoms', 'New_ID']].copy()
    out_df.columns = ['smiles', target_col, 'Ring_ID', 'Ring_Atoms', 'New_ID']
    out_df['atom_on_ring'] = df_l10['Ring_Atoms']
    out_df.to_csv(tmp_path, index=False)

    l10_data = load_adj_format(tmp_path, target_col, NVL, MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    return df_l10, l10_data


def train_and_predict(task_name, task_info, device, seed):
    """在训练集上训练 MPNN, 在 lunci10 上预测"""
    target_col = task_info['target_col']

    # 加载训练数据
    train_data = load_adj_format(task_info['dataset_path'], target_col, NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n_train = train_data['n']

    # 训练/验证划分: 87.5% train / 12.5% val
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_train)
    n_final_train = int(0.875 * n_train)
    train_idx = torch.tensor(perm[:n_final_train], device=device)
    val_idx = torch.tensor(perm[n_final_train:], device=device)

    # 训练
    params = dict(DEFAULT_PARAMS)
    t0 = time.time()
    model = train_model('MPNN', params, train_data, train_idx, val_idx,
                       device, params['n_epochs'], params['patience'], seed)
    train_time = time.time() - t0

    # 在 lunci10 上预测
    df_l10, l10_data = load_lunci10_for_pred(task_name, task_info, device)
    test_idx = torch.arange(l10_data['n'], device=device)
    te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, l10_data, test_idx,
                                                           params['batch_size'], device)

    print(f"  [{task_name} seed={seed}] Train+val | lunci10 R2={te_r2:.4f} "
          f"MAE={te_mae:.4f} ({train_time:.0f}s)")

    # 合并预测结果
    df_pred = df_l10.copy()
    df_pred['pred'] = te_pred
    df_pred['true'] = te_true

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df_pred, {'test_r2': te_r2, 'test_mae': te_mae, 'train_time': train_time}


def evaluate_delta_pairs(df_pairs, pred_map):
    """评估 delta_hat vs delta_true

    pred_map: {New_ID: {Ring_ID: pred_value}}
    """
    # 获取每对分子的预测值
    pred_i = []
    pred_j = []
    for _, row in df_pairs.iterrows():
        key_i = (row['new_id_i'], row['ring_id'])
        key_j = (row['new_id_j'], row['ring_id'])
        if key_i in pred_map and key_j in pred_map:
            pred_i.append(pred_map[key_i])
            pred_j.append(pred_map[key_j])
        else:
            pred_i.append(np.nan)
            pred_j.append(np.nan)

    df_pairs = df_pairs.copy()
    df_pairs['pred_i'] = pred_i
    df_pairs['pred_j'] = pred_j
    df_pairs['pred_delta'] = df_pairs['pred_i'] - df_pairs['pred_j']
    df_pairs['abs_error'] = (df_pairs['pred_delta'] - df_pairs['delta_A']).abs()

    # 移除 NaN
    df_valid = df_pairs.dropna(subset=['pred_i', 'pred_j']).reset_index(drop=True)

    if len(df_valid) == 0:
        return None, df_pairs

    true_delta = df_valid['delta_A'].values
    pred_delta = df_valid['pred_delta'].values

    # 指标
    r2, mae, rmse = compute_metrics(true_delta, pred_delta)
    rho, _ = spearmanr(true_delta, pred_delta)
    tau, _ = kendalltau(true_delta, pred_delta)

    # Pairwise direction accuracy
    correct = 0
    total = 0
    for td, pd in zip(true_delta, pred_delta):
        if td == 0 or pd == 0:
            continue
        if np.sign(td) == np.sign(pd):
            correct += 1
        total += 1
    pairwise_acc = correct / total if total > 0 else np.nan

    metrics = {
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'spearman': rho, 'kendall': tau,
        'pairwise_acc': pairwise_acc,
        'n_pairs': len(df_valid),
    }

    return metrics, df_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    task_map = {t['name']: t for t in TASKS}
    tasks = args.tasks.split(',')

    all_metrics = []
    all_pair_results = []

    for task_name in tasks:
        task = task_map[task_name]
        print(f"\n{'='*70}")
        print(f"# {task_name}")
        print(f"{'='*70}")

        # 加载 pair dataset
        pair_file = os.path.join(PAIR_DIR, f'pair_dataset_{task_name.lower().replace("_1zz","")}.csv')
        df_pairs = pd.read_csv(pair_file)
        print(f"  Loaded {len(df_pairs)} pairs from {pair_file}")

        seed_metrics = []
        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")
            df_pred, info = train_and_predict(task_name, task, device, seed)

            # 构建 pred_map: (New_ID, Ring_ID) → pred
            pred_map = {}
            for _, row in df_pred.iterrows():
                key = (row['New_ID'], row['Ring_ID'])
                pred_map[key] = row['pred']

            # 评估 pairs
            metrics, df_pair_results = evaluate_delta_pairs(df_pairs, pred_map)
            if metrics is None:
                print(f"    No valid pairs!")
                continue

            metrics['task'] = task_name
            metrics['seed'] = seed
            metrics['abs_r2'] = info['test_r2']
            metrics['abs_mae'] = info['test_mae']
            metrics['train_time'] = info['train_time']
            seed_metrics.append(metrics)

            df_pair_results['task'] = task_name
            df_pair_results['seed'] = seed
            all_pair_results.append(df_pair_results)

            print(f"    Δ R²={metrics['r2']:.4f} | Δ MAE={metrics['mae']:.4f} | "
                  f"Δ RMSE={metrics['rmse']:.4f} | Spearman={metrics['spearman']:.4f} | "
                  f"Kendall={metrics['kendall']:.4f} | Pairwise={metrics['pairwise_acc']:.4f}")

        all_metrics.extend(seed_metrics)

    # 保存结果
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'subtraction_per_seed.csv'), index=False)

    # 聚合
    agg = df_metrics.groupby('task').agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        abs_r2_mean=('abs_r2', 'mean'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg.to_csv(os.path.join(OUTPUT_DIR, 'subtraction_agg.csv'), index=False)

    # 保存 pair-level 结果 (最后一个 seed)
    if all_pair_results:
        df_pairs_all = pd.concat(all_pair_results, ignore_index=True)
        df_pairs_all.to_csv(os.path.join(OUTPUT_DIR, 'pair_level_predictions.csv'), index=False)

    # 打印汇总
    print(f"\n{'='*70}")
    print("Baseline A: Absolute Model Subtraction (mean±SD over 5 seeds)")
    print(f"{'='*70}")
    for _, row in agg.iterrows():
        print(f"\n  [{row['task']}]")
        print(f"    Absolute R² = {row['abs_r2_mean']:.4f}")
        print(f"    Δ R²       = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
        print(f"    Δ MAE      = {row['mae_mean']:.4f} ± {row['mae_std']:.4f}")
        print(f"    Δ RMSE     = {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}")
        print(f"    Spearman ρ = {row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}")
        print(f"    Kendall τ  = {row['kendall_mean']:.4f} ± {row['kendall_std']:.4f}")
        print(f"    Pairwise   = {row['pairwise_mean']:.4f} ± {row['pairwise_std']:.4f}")

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
