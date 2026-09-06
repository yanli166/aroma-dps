
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
lunci10 三层分析:
  A. Absolute prediction        — 能不能预测总体芳香性?
  B. Within-scaffold Δ prediction — 能不能捕捉 substituent-induced aromaticity shift?
  C. Ranking                    — 对于同一 scaffold, 哪个 substituent 增强/削弱芳香性最多?

预测来源: 在训练集 (collet_*_0716.csv) 上训练 MPNN (ring_flag=10, 即 ring membership 编码),
         在 lunci10-test.csv 上推理. 5 seeds (42/123/456/789/2024), 报告 mean±SD.

scaffold/substituent 元数据来源: lunci10/lunci10-begin.csv (ring_name, sub_name, ring_pos).
reference 策略 (Layer B):
  由于 lunci10 不含 unsubstituted parent (无 H/none 取代基条目),
  采用 scaffold+ring_pos 组内均值作为 reference baseline:
    Δ_true_i = A_true_i - mean(A_true | scaffold, ring_pos)
    Δ_pred_i = A_pred_i - mean(A_pred | scaffold, ring_pos)
  这消除 scaffold-level 偏移, 隔离 substituent-induced shift.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau
from itertools import combinations
from collections import defaultdict

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'
LUNCI10_META = '_PROJ_ROOT/lunci10/lunci10-begin.csv'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, DEFAULT_SEED, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, train_model, eval_model, set_full_seed

RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/lunci10_three_layer'
os.makedirs(RESULTS_DIR, exist_ok=True)

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10  # ring membership encoding (label)
SEEDS = [42, 123, 456, 789, 2024]

TASK_COL_MAP = {
    'HOMA':      ('HOMA',    'homa_value'),
    'NICS_1zz':  ('NICS_ZZ', 'NICS_value'),
    'MBCO':      ('MBCO',    'mbco_value'),
}


def load_lunci10_with_meta(task_name):
    """加载 lunci10 测试数据并合并 scaffold/substituent 元数据"""
    l10_col, target_col = TASK_COL_MAP[task_name]

    df_l10 = pd.read_csv(os.path.join(PROJ_ROOT, 'data1_end', 'lunci10-test.csv'),
                         encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=[l10_col, 'SMILES']).reset_index(drop=True)

    # 加载元数据 (scaffold / substituent)
    df_meta = pd.read_csv(LUNCI10_META)
    df_meta.columns = df_meta.columns.str.strip()

    # 合并: 通过 SMILES 匹配
    df_l10['SMILES_canon'] = df_l10['SMILES']  # lunci10 SMILES 已是 canonical
    df_meta['SMILES_canon'] = df_meta['SMILES']

    # 用 New_ID 合并更可靠 (lunci10-test 的 New_ID = begin.csv 的 no)
    df_l10_merged = df_l10.merge(
        df_meta[['no', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']],
        left_on='New_ID', right_on='no', how='left'
    )

    # 标记未匹配的行
    unmatched = df_l10_merged['ring_name'].isna().sum()
    if unmatched > 0:
        print(f"  [load_lunci10_with_meta] {task_name}: {unmatched}/{len(df_l10_merged)} 行未匹配元数据 (将标记为 'unknown')")

    df_l10_merged['ring_name'] = df_l10_merged['ring_name'].fillna('unknown')
    df_l10_merged['sub_name'] = df_l10_merged['sub_name'].fillna('unknown')
    df_l10_merged['ring_pos'] = df_l10_merged['ring_pos'].fillna(-1)
    df_l10_merged['sub_type'] = df_l10_merged['sub_type'].fillna('unknown')

    # target 值
    df_l10_merged['target'] = df_l10_merged[l10_col].astype(float)

    # 保存为临时 CSV 供 load_adj_format 使用 (需要 smiles 列)
    tmp_path = os.path.join('/tmp', f'lunci10_{task_name}_for_pred.csv')
    out_df = df_l10_merged[['SMILES', 'target', 'Ring_ID', 'Ring_Atoms',
                            'New_ID', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']].copy()
    out_df.columns = ['smiles', target_col, 'Ring_ID', 'Ring_Atoms',
                      'New_ID', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']
    out_df['atom_on_ring'] = df_l10_merged['Ring_Atoms']
    out_df.to_csv(tmp_path, index=False)

    return out_df, tmp_path, df_l10_merged, target_col


def train_and_predict(task_name, task_info, device, seed):
    """在训练集上训练 MPNN, 在 lunci10 上预测"""
    target_col = task_info['target_col']

    # 加载训练数据
    train_data = load_adj_format(task_info['dataset_path'], target_col, NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n_train = train_data['n']
    smiles_list = train_data['smiles']

    # 训练/验证划分: 87.5% train / 12.5% val (与 Layer 2 一致)
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

    # 在训练集上评估 (sanity check)
    tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, train_data, train_idx,
                                              params['batch_size'], device)

    # 加载 lunci10 数据并预测
    out_df, tmp_path, df_meta, _ = load_lunci10_with_meta(task_name)
    l10_data = load_adj_format(tmp_path, target_col, NVL, MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    test_idx = torch.arange(l10_data['n'], device=device)
    te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, l10_data, test_idx,
                                                           params['batch_size'], device)

    print(f"  [{task_name} seed={seed}] Train R2={tr_r2:.4f} | lunci10 R2={te_r2:.4f} "
          f"MAE={te_mae:.4f} RMSE={te_rmse:.4f} ({train_time:.0f}s)")

    # 合并预测结果与元数据
    df_pred = df_meta.copy()
    df_pred['pred'] = te_pred
    df_pred['true'] = te_true

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df_pred, {'train_r2': tr_r2, 'test_r2': te_r2, 'test_mae': te_mae,
                     'test_rmse': te_rmse, 'train_time': train_time, 'seed': seed}


# ── Layer A: Absolute prediction ──────────────────────────────────────────────

def layer_a_absolute(df_pred):
    """Layer A: 绝对预测 — 总体芳香性预测能力"""
    true = df_pred['true'].values
    pred = df_pred['pred'].values
    r2, mae, rmse = compute_metrics(true, pred)
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'n': len(true)}


# ── Layer B: Within-scaffold Δ prediction ──────────────────────────────────────

def layer_b_delta(df_pred):
    """Layer B: Within-scaffold Δ prediction

    Δ_i = A_i - mean(A | scaffold, ring_pos)

    评估: 模型预测的 Δ_pred 是否与真实 Δ_true 相关?
    """
    # 按 scaffold + ring_pos 分组, 计算 group mean
    group_col = ['ring_name', 'ring_pos']
    df = df_pred.copy()
    df['true_mean'] = df.groupby(group_col)['true'].transform('mean')
    df['pred_mean'] = df.groupby(group_col)['pred'].transform('mean')

    df['true_delta'] = df['true'] - df['true_mean']
    df['pred_delta'] = df['pred'] - df['pred_mean']

    # 过滤: 只保留 group size >= 3 的组 (至少 3 个 substituent 才有意义)
    group_sizes = df.groupby(group_col).size().reset_index(name='group_size')
    valid_groups = group_sizes[group_sizes['group_size'] >= 3]
    df_valid = df.merge(valid_groups[group_col + ['group_size']], on=group_col, how='inner')

    if len(df_valid) == 0:
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'n': 0,
                'n_groups': 0, 'correlation': np.nan}

    true_d = df_valid['true_delta'].values
    pred_d = df_valid['pred_delta'].values

    r2, mae, rmse = compute_metrics(true_d, pred_d)
    rho, _ = spearmanr(true_d, pred_d)

    n_groups = df_valid.groupby(group_col).ngroups

    return {
        'r2': r2, 'mae': mae, 'rmse': rmse, 'n': len(df_valid),
        'n_groups': n_groups, 'spearman_rho': rho,
        'true_delta_std': np.std(true_d, ddof=1),
        'pred_delta_std': np.std(pred_d, ddof=1),
        'df': df_valid,
    }


# ── Layer C: Ranking ───────────────────────────────────────────────────────────

def layer_c_ranking(df_pred):
    """Layer C: Within-scaffold ranking

    对每个 scaffold+ring_pos 组:
      - 计算 true ranking 和 pred ranking
      - Spearman ρ, Kendall τ, pairwise ranking accuracy
    """
    group_col = ['ring_name', 'ring_pos']
    df = df_pred.copy()

    results = []
    grouped = df.groupby(group_col)

    for (scaffold, ring_pos), group in grouped:
        if len(group) < 3:
            continue  # 至少 3 个 substituent 才能计算 ranking

        true_vals = group['true'].values
        pred_vals = group['pred'].values
        n = len(group)

        # Spearman ρ
        rho, p_rho = spearmanr(true_vals, pred_vals)

        # Kendall τ
        tau, p_tau = kendalltau(true_vals, pred_vals)

        # Pairwise ranking accuracy
        correct = 0
        total = 0
        for i, j in combinations(range(n), 2):
            true_diff = true_vals[i] - true_vals[j]
            pred_diff = pred_vals[i] - pred_vals[j]
            if true_diff == 0 or pred_diff == 0:
                continue
            true_sign = np.sign(true_diff)
            pred_sign = np.sign(pred_diff)
            if true_sign == pred_sign:
                correct += 1
            total += 1

        pairwise_acc = correct / total if total > 0 else np.nan

        results.append({
            'scaffold': scaffold,
            'ring_pos': ring_pos,
            'n_substituents': n,
            'spearman_rho': rho,
            'spearman_p': p_rho,
            'kendall_tau': tau,
            'kendall_p': p_tau,
            'pairwise_accuracy': pairwise_acc,
            'n_pairs': total,
        })

    df_rank = pd.DataFrame(results)
    return df_rank


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='lunci10 三层分析')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    task_map = {t['name']: t for t in TASKS}
    tasks = args.tasks.split(',')

    all_layer_a = []
    all_layer_b = []
    all_layer_c = []
    all_predictions = []

    for task_name in tasks:
        task = task_map[task_name]
        print(f"\n{'='*70}")
        print(f"# {task_name}")
        print(f"{'='*70}")

        seed_results_a = []
        seed_results_b = []
        seed_results_c = []
        seed_predictions = []

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            df_pred, metrics = train_and_predict(task_name, task, device, seed)

            # Layer A
            a = layer_a_absolute(df_pred)
            a['task'] = task_name
            a['seed'] = seed
            a['train_r2'] = metrics['train_r2']
            a['train_time'] = metrics['train_time']
            seed_results_a.append(a)

            # Layer B
            b = layer_b_delta(df_pred)
            b['task'] = task_name
            b['seed'] = seed
            seed_results_b.append(b)

            # Layer C
            c = layer_c_ranking(df_pred)
            c['task'] = task_name
            c['seed'] = seed
            seed_results_c.append(c)

            # 保存预测
            df_pred['task'] = task_name
            df_pred['seed'] = seed
            seed_predictions.append(df_pred.copy())

        # 聚合 Layer A
        df_a = pd.DataFrame([{k: v for k, v in r.items() if k != 'df'} for r in seed_results_a])
        all_layer_a.append(df_a)

        # 聚合 Layer B
        df_b = pd.DataFrame([{k: v for k, v in r.items() if k != 'df'} for r in seed_results_b])
        all_layer_b.append(df_b)

        # 聚合 Layer C
        df_c = pd.concat(seed_results_c, ignore_index=True)
        all_layer_c.append(df_c)

        # 保存预测
        df_preds = pd.concat(seed_predictions, ignore_index=True)
        all_predictions.append(df_preds)

        # 打印 per-task 汇总
        print(f"\n{'='*70}")
        print(f"# {task_name} 汇总 (mean±SD over {len(SEEDS)} seeds)")
        print(f"{'='*70}")

        print("\n[Layer A] Absolute prediction:")
        a_mean = df_a[['r2', 'mae', 'rmse']].mean()
        a_std = df_a[['r2', 'mae', 'rmse']].std()
        print(f"  R²  = {a_mean['r2']:.4f} ± {a_std['r2']:.4f}")
        print(f"  MAE = {a_mean['mae']:.4f} ± {a_std['mae']:.4f}")
        print(f"  RMSE= {a_mean['rmse']:.4f} ± {a_std['rmse']:.4f}")

        print("\n[Layer B] Within-scaffold Δ prediction:")
        b_mean = df_b[['r2', 'mae', 'rmse', 'spearman_rho']].mean()
        b_std = df_b[['r2', 'mae', 'rmse', 'spearman_rho']].std()
        print(f"  Δ R²  = {b_mean['r2']:.4f} ± {b_std['r2']:.4f}")
        print(f"  Δ MAE = {b_mean['mae']:.4f} ± {b_std['mae']:.4f}")
        print(f"  Δ RMSE= {b_mean['rmse']:.4f} ± {b_std['rmse']:.4f}")
        print(f"  Spearman ρ (Δ) = {b_mean['spearman_rho']:.4f} ± {b_std['spearman_rho']:.4f}")

        print("\n[Layer C] Within-scaffold ranking:")
        c_mean = df_c[['spearman_rho', 'kendall_tau', 'pairwise_accuracy']].mean()
        c_std = df_c[['spearman_rho', 'kendall_tau', 'pairwise_accuracy']].std()
        print(f"  Spearman ρ      = {c_mean['spearman_rho']:.4f} ± {c_std['spearman_rho']:.4f}")
        print(f"  Kendall τ       = {c_mean['kendall_tau']:.4f} ± {c_std['kendall_tau']:.4f}")
        print(f"  Pairwise acc.   = {c_mean['pairwise_accuracy']:.4f} ± {c_std['pairwise_accuracy']:.4f}")
        print(f"  (n_groups = {df_c.groupby(['task','seed']).ngroups} task×seed combos)")

    # 保存汇总
    df_all_a = pd.concat(all_layer_a, ignore_index=True)
    df_all_b = pd.concat(all_layer_b, ignore_index=True)
    df_all_c = pd.concat(all_layer_c, ignore_index=True)
    df_all_preds = pd.concat(all_predictions, ignore_index=True)

    df_all_a.to_csv(os.path.join(RESULTS_DIR, 'layer_a_absolute.csv'), index=False)
    df_all_b.to_csv(os.path.join(RESULTS_DIR, 'layer_b_delta.csv'), index=False)
    df_all_c.to_csv(os.path.join(RESULTS_DIR, 'layer_c_ranking.csv'), index=False)
    df_all_preds.to_csv(os.path.join(RESULTS_DIR, 'all_predictions.csv'), index=False)

    # 聚合表 (mean±SD over seeds)
    agg_a = df_all_a.groupby('task').agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        train_r2_mean=('train_r2', 'mean'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg_a.to_csv(os.path.join(RESULTS_DIR, 'layer_a_agg.csv'), index=False)

    agg_b = df_all_b.groupby('task').agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman_rho', 'mean'), spearman_std=('spearman_rho', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg_b.to_csv(os.path.join(RESULTS_DIR, 'layer_b_agg.csv'), index=False)

    agg_c = df_all_c.groupby('task').agg(
        spearman_mean=('spearman_rho', 'mean'), spearman_std=('spearman_rho', 'std'),
        kendall_mean=('kendall_tau', 'mean'), kendall_std=('kendall_tau', 'std'),
        pairwise_mean=('pairwise_accuracy', 'mean'), pairwise_std=('pairwise_accuracy', 'std'),
        n_groups=('scaffold', 'count'),
    ).reset_index()
    agg_c.to_csv(os.path.join(RESULTS_DIR, 'layer_c_agg.csv'), index=False)

    print(f"\n\n{'='*70}")
    print("所有结果保存至:")
    print(f"  {RESULTS_DIR}/layer_a_absolute.csv  (per seed)")
    print(f"  {RESULTS_DIR}/layer_a_agg.csv       (mean±SD)")
    print(f"  {RESULTS_DIR}/layer_b_delta.csv     (per seed)")
    print(f"  {RESULTS_DIR}/layer_b_agg.csv       (mean±SD)")
    print(f"  {RESULTS_DIR}/layer_c_ranking.csv   (per scaffold, per seed)")
    print(f"  {RESULTS_DIR}/layer_c_agg.csv       (mean±SD)")
    print(f"  {RESULTS_DIR}/all_predictions.csv   (all predictions)")


if __name__ == '__main__':
    main()
