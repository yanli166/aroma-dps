
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
Step 7: Scaffold-level Bias Decomposition

解释为什么 absolute prediction 差, 但 Δ prediction 更好。

对于每个 scaffold+ring_pos:
  bias_g = mean(pred_A - true_A)
  true_centered_i = true_A_i - mean(true_A_group)
  pred_centered_i = pred_A_i - mean(pred_A_group)

计算:
  absolute_MAE_g, centered_MAE_g, bias_g

回答: lunci10 的 absolute OOD failure 是否主要来自 scaffold-level calibration offset?

输出:
  results/lunci10_delta_learning/06_bias_decomposition/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
L10_TEST = os.path.join(PROJ_ROOT, 'data1_end/lunci10-test.csv')
L10_META = '_PROJ_ROOT/lunci10/lunci10-begin.csv'
SUBTRACTION_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/02_subtraction_baseline')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/06_bias_decomposition')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}


def load_predictions_with_meta(task_name):
    """加载 lunci10 预测和元数据"""
    l10_col = TASKS[task_name]
    df_l10 = pd.read_csv(L10_TEST, encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=[l10_col, 'SMILES']).reset_index(drop=True)

    df_meta = pd.read_csv(L10_META)
    df_meta.columns = df_meta.columns.str.strip()

    df = df_l10.merge(
        df_meta[['no', 'ring_name', 'sub_name', 'ring_pos']],
        left_on='New_ID', right_on='no', how='left'
    )
    df['ring_name'] = df['ring_name'].fillna('unknown')
    df['ring_pos'] = df['ring_pos'].fillna(-1).astype(int)

    # 加载预测 (pair_level_predictions.csv 有 pred 和 true)
    pred_file = os.path.join(SUBTRACTION_DIR, 'pair_level_predictions.csv')
    if not os.path.exists(pred_file):
        # 从 log 解析
        return None, f"预测文件不存在: {pred_file}"

    df_pairs = pd.read_csv(pred_file)
    df_pairs = df_pairs[df_pairs['task'] == task_name]

    # 提取每个分子的预测
    pred_map = {}
    for _, row in df_pairs.iterrows():
        key_i = (row['new_id_i'], row['ring_id'])
        key_j = (row['new_id_j'], row['ring_id'])
        if key_i not in pred_map:
            pred_map[key_i] = row['pred_i']
        if key_j not in pred_map:
            pred_map[key_j] = row['pred_j']

    df['pred'] = df.apply(lambda r: pred_map.get((r['New_ID'], r['Ring_ID']), np.nan), axis=1)
    df['true'] = df[l10_col].astype(float)
    df = df.dropna(subset=['pred']).reset_index(drop=True)

    return df, None


def main():
    print("=" * 70)
    print("Step 7: Scaffold-level Bias Decomposition")
    print("=" * 70)

    all_results = []

    for task_name in TASKS:
        print(f"\n--- {task_name} ---")
        df, err = load_predictions_with_meta(task_name)
        if err:
            print(f"  {err}")
            continue

        group_cols = ['ring_name', 'ring_pos']

        # 计算每组的 bias 和 centered error
        df['error'] = df['pred'] - df['true']  # pred - true
        df['abs_error'] = df['error'].abs()

        # Group-level statistics
        group_stats = df.groupby(group_cols).agg(
            n_samples=('pred', 'count'),
            bias=('error', 'mean'),
            abs_mae=('abs_error', 'mean'),
            true_mean=('true', 'mean'),
            pred_mean=('pred', 'mean'),
            true_std=('true', 'std'),
            pred_std=('pred', 'std'),
        ).reset_index()

        # Centered errors (within-group)
        df['true_centered'] = df.groupby(group_cols)['true'].transform(lambda x: x - x.mean())
        df['pred_centered'] = df.groupby(group_cols)['pred'].transform(lambda x: x - x.mean())
        df['centered_error'] = (df['pred_centered'] - df['true_centered']).abs()

        group_stats['centered_mae'] = df.groupby(group_cols)['centered_error'].mean().values
        group_stats['bias_abs'] = group_stats['bias'].abs()

        # 总体统计
        overall_abs_mae = df['abs_error'].mean()
        overall_centered_mae = df['centered_error'].mean()
        overall_bias = df['error'].mean()

        print(f"  Overall absolute MAE:  {overall_abs_mae:.4f}")
        print(f"  Overall centered MAE:   {overall_centered_mae:.4f}")
        print(f"  Overall bias:           {overall_bias:.4f}")
        print(f"  Centering improvement:  {(overall_abs_mae - overall_centered_mae) / overall_abs_mae * 100:.1f}%")

        group_stats['task'] = task_name
        all_results.append(group_stats)

        # 生成图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: absolute_MAE vs centered_MAE per group
        ax = axes[0]
        ax.scatter(group_stats['abs_mae'], group_stats['centered_mae'], alpha=0.6, s=50)
        lim = max(group_stats['abs_mae'].max(), group_stats['centered_mae'].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'r--', alpha=0.5)
        ax.set_xlabel('Absolute MAE (per group)')
        ax.set_ylabel('Centered MAE (per group)')
        ax.set_title(f'{task_name}: Absolute vs Centered Error\n'
                     f'(Centering improves {task_name} by '
                     f'{(overall_abs_mae - overall_centered_mae) / overall_abs_mae * 100:.1f}%)')

        # Plot 2: bias distribution
        ax = axes[1]
        ax.hist(group_stats['bias'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Bias (mean(pred - true))')
        ax.set_ylabel('Count (groups)')
        ax.set_title(f'{task_name}: Scaffold-level Bias Distribution\n'
                     f'(mean bias={group_stats["bias"].mean():.4f}, '
                     f'std={group_stats["bias"].std():.4f})')

        # Plot 3: bias vs absolute error
        ax = axes[2]
        scatter = ax.scatter(group_stats['bias_abs'], group_stats['abs_mae'],
                             c=group_stats['centered_mae'], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax, label='Centered MAE')
        ax.set_xlabel('|Bias| (|mean(pred - true)|)')
        ax.set_ylabel('Absolute MAE')
        ax.set_title(f'{task_name}: |Bias| vs Absolute MAE')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'absolute_vs_centered_error_{task_name}.png'), dpi=150)
        plt.close()

        # 保存 per-group 结果
        group_stats.to_csv(
            os.path.join(OUTPUT_DIR, f'scaffold_bias_{task_name}.csv'),
            index=False
        )

    # 合并保存
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, 'scaffold_bias.csv'), index=False)

    # 生成 bias_distribution.png (所有任务合并)
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for task_name in TASKS:
            task_data = df_all[df_all['task'] == task_name]
            ax.hist(task_data['bias'], bins=20, alpha=0.5, label=task_name, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Bias (mean(pred - true)) per scaffold group')
        ax.set_ylabel('Count')
        ax.set_title('Scaffold-level Bias Distribution (All Tasks)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bias_distribution.png'), dpi=150)
        plt.close()

    # 生成报告
    report = "# Step 7: Scaffold-level Bias Decomposition\n\n"
    report += "## 核心问题\n\n"
    report += "> lunci10 的 absolute OOD failure 是否主要来自 scaffold-level calibration offset,\n"
    report += "> 而模型仍然保留 substituent-response information?\n\n"

    for task_name in TASKS:
        df_result = [r for r in all_results if r['task'].iloc[0] == task_name]
        if not df_result:
            continue
        gs = df_result[0]
        overall_abs = gs['abs_mae'].mean()
        overall_cen = gs['centered_mae'].mean()
        overall_bias_mean = gs['bias'].mean()
        overall_bias_std = gs['bias'].std()
        improvement = (overall_abs - overall_cen) / overall_abs * 100 if overall_abs > 0 else 0

        report += f"## {task_name}\\n\n"
        report += f"| 指标 | 值 |\n|------|----|\n"
        report += f"| Overall Absolute MAE | {overall_abs:.4f} |\n"
        report += f"| Overall Centered MAE | {overall_cen:.4f} |\n"
        report += f"| Centering Improvement | {improvement:.1f}% |\n"
        report += f"| Mean Bias | {overall_bias_mean:.4f} |\n"
        report += f"| Bias Std | {overall_bias_std:.4f} |\n"
        report += f"| N Groups | {len(gs)} |\n\n"

        if improvement > 50:
            report += f"**结论**: Centering 改善了 {improvement:.1f}% 的误差, 表明 absolute prediction 的误差\n"
            report += f"主要来自 scaffold-level calibration offset。模型仍然保留了 substituent-response information。\n\n"
        elif improvement > 20:
            report += f"**结论**: Centering 改善了 {improvement:.1f}% 的误差, 表明 scaffold-level bias 是\n"
            report += f"absolute prediction 误差的重要组成部分, 但不是唯一来源。\n\n"
        else:
            report += f"**结论**: Centering 仅改善了 {improvement:.1f}% 的误差, 表明 absolute prediction 的误差\n"
            report += f"不仅来自 scaffold-level bias, 还有其他来源 (如 substituent-level error)。\n\n"

    report += "## 文件列表\n\n"
    report += "- `scaffold_bias.csv` — 每个 scaffold+ring_pos 的 bias/MAE/centered_MAE\n"
    report += "- `scaffold_bias_{task}.csv` — 各任务单独的 per-group 结果\n"
    report += "- `absolute_vs_centered_error_{task}.png` — 绝对误差 vs centered 误差散点图\n"
    report += "- `bias_distribution.png` — bias 分布直方图\n"

    with open(os.path.join(OUTPUT_DIR, 'bias_decomposition_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
