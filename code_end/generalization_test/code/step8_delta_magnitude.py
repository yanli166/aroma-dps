
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
Step 8: Δ Magnitude / Signal-to-Noise Analysis

统计 |ΔHOMA|, |ΔNICS|, |ΔMBCO| 的分布。
将 pair 按真实 |Δ| 分为 small/medium/large (quantile-based)。
分别报告各区间: MAE, Spearman, pairwise accuracy。

回答: 模型是否主要在非常小的 substituent-induced aromaticity shift 上失败,
      而对较明显的变化具有更稳定的预测能力?

输出:
  results/lunci10_delta_learning/07_delta_magnitude/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
SUBTRACTION_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/02_subtraction_baseline')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/07_delta_magnitude')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = ['HOMA', 'NICS_1zz', 'MBCO']


def compute_metrics_bin(y_true, y_pred):
    """计算一个 bin 内的指标"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    if len(y_true) < 2:
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                'spearman': np.nan, 'kendall': np.nan, 'pairwise_acc': np.nan,
                'n': len(y_true)}

    r2 = r2_score(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred) if len(set(y_true)) > 1 else (np.nan, np.nan)
    tau, _ = kendalltau(y_true, y_pred) if len(set(y_true)) > 1 else (np.nan, np.nan)

    correct = 0
    total = 0
    for td, pd in zip(y_true, y_pred):
        if td == 0 or pd == 0:
            continue
        if np.sign(td) == np.sign(pd):
            correct += 1
        total += 1
    pairwise_acc = correct / total if total > 0 else np.nan

    return {'r2': r2, 'mae': mae, 'rmse': rmse,
            'spearman': rho, 'kendall': tau,
            'pairwise_acc': pairwise_acc, 'n': len(y_true)}


def main():
    print("=" * 70)
    print("Step 8: Δ Magnitude / Signal-to-Noise Analysis")
    print("=" * 70)

    all_results = []

    for task_name in TASKS:
        print(f"\n--- {task_name} ---")

        # 加载 pair-level predictions (from Step 3)
        pred_file = os.path.join(SUBTRACTION_DIR, 'pair_level_predictions.csv')
        if not os.path.exists(pred_file):
            print(f"  预测文件不存在: {pred_file}")
            continue

        df = pd.read_csv(pred_file)
        df = df[df['task'] == task_name].copy()

        if len(df) == 0:
            print(f"  No data for {task_name}")
            continue

        df['abs_delta_true'] = df['delta_A'].abs()

        # 按 |Δ| 的 quantile 分为 small/medium/large
        q33 = df['abs_delta_true'].quantile(0.33)
        q67 = df['abs_delta_true'].quantile(0.67)

        df['magnitude'] = 'medium'
        df.loc[df['abs_delta_true'] <= q33, 'magnitude'] = 'small'
        df.loc[df['abs_delta_true'] > q67, 'magnitude'] = 'large'

        print(f"  |Δ| quantiles: q33={q33:.4f}, q67={q67:.4f}")
        print(f"  Bin sizes: small={len(df[df['magnitude']=='small'])}, "
              f"medium={len(df[df['magnitude']=='medium'])}, "
              f"large={len(df[df['magnitude']=='large'])}")

        # 对每个 bin 计算 metrics (per seed, 然后聚合)
        for mag in ['small', 'medium', 'large']:
            subset = df[df['magnitude'] == mag]
            for seed in sorted(subset['seed'].unique()):
                seed_data = subset[subset['seed'] == seed]
                m = compute_metrics_bin(seed_data['delta_A'].values,
                                        seed_data['pred_delta'].values)
                m['task'] = task_name
                m['magnitude'] = mag
                m['seed'] = seed
                m['abs_delta_q33'] = q33
                m['abs_delta_q67'] = q67
                all_results.append(m)

                if seed == sorted(subset['seed'].unique())[0]:
                    print(f"  [{mag:6s}] n={m['n']:5d}, R²={m['r2']:.4f}, "
                          f"MAE={m['mae']:.4f}, Spearman={m['spearman']:.4f}, "
                          f"Pairwise={m['pairwise_acc']:.4f}")

        # 生成图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: |Δ| distribution
        ax = axes[0]
        ax.hist(df['abs_delta_true'], bins=50, edge='black', alpha=0.7)
        ax.axvline(x=q33, color='r', linestyle='--', label=f'q33={q33:.4f}')
        ax.axvline(x=q67, color='b', linestyle='--', label=f'q67={q67:.4f}')
        ax.set_xlabel('|Δ| (absolute delta)')
        ax.set_ylabel('Count')
        ax.set_title(f'{task_name}: |Δ| Distribution')
        ax.legend()

        # Plot 2: pairwise accuracy by magnitude
        ax = axes[1]
        df_plot = pd.DataFrame(all_results)
        df_task = df_plot[(df_plot['task'] == task_name)]
        mags = ['small', 'medium', 'large']
        means = [df_task[df_task['magnitude'] == m]['pairwise_acc'].mean() for m in mags]
        stds = [df_task[df_task['magnitude'] == m]['pairwise_acc'].std() for m in mags]
        ax.bar(mags, means, yerr=stds, capsize=5, alpha=0.7, color=['#2196F3', '#4CAF50', '#FF5722'])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.set_ylabel('Pairwise Accuracy')
        ax.set_title(f'{task_name}: Pairwise Accuracy by |Δ| Magnitude')
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'delta_magnitude_{task_name}.png'), dpi=150)
        plt.close()

    # 聚合
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'delta_magnitude_per_seed.csv'), index=False)

    agg = df_results.groupby(['task', 'magnitude']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        n_mean=('n', 'mean'),
    ).reset_index()
    agg.to_csv(os.path.join(OUTPUT_DIR, 'delta_magnitude_agg.csv'), index=False)

    print(f"\n{'='*70}")
    print("Δ Magnitude Analysis (mean±SD over 5 seeds)")
    print(f"{'='*70}")
    for _, r in agg.iterrows():
        print(f"  [{r['task']:10s} / {r['magnitude']:6s}] n={r['n_mean']:.0f} | "
              f"R²={r['r2_mean']:.4f}±{r['r2_std']:.4f} | "
              f"MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f} | "
              f"Spearman={r['spearman_mean']:.4f}±{r['spearman_std']:.4f} | "
              f"Pairwise={r['pairwise_mean']:.4f}±{r['pairwise_std']:.4f}")

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
