
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
Step 9: Substituent / Position Error Analysis

分析模型误差与:
  - substituent identity/type
  - ring position
  - scaffold type
  - heteroatom composition
  - fused/non-fused

之间的关系。

不强行做复杂 Hammett 分析; 只有在 substituent 和位置具有明确 Hammett 可比性时再使用 σ 参数。

重点找: easy substituent perturbations vs hard substituent perturbations

输出:
  results/lunci10_delta_learning/08_error_analysis/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from rdkit import Chem

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
L10_TEST = os.path.join(PROJ_ROOT, 'data1_end/lunci10-test.csv')
L10_META = '_PROJ_ROOT/lunci10/lunci10-begin.csv'
SUBTRACTION_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/02_subtraction_baseline')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/08_error_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = ['HOMA', 'NICS_1zz', 'MBCO']


def get_heteroatom_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, 0
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    hetero = [a for a in atoms if a not in ('C', 'H')]
    return dict(Counter(hetero)), len(hetero)


def is_fused(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    if len(ri.AtomRings()) < 2:
        return False
    rings = ri.AtomRings()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if set(rings[i]) & set(rings[j]):
                return True
    return False


def main():
    print("=" * 70)
    print("Step 9: Substituent / Position Error Analysis")
    print("=" * 70)

    # 加载 pair-level predictions
    pred_file = os.path.join(SUBTRACTION_DIR, 'pair_level_predictions.csv')
    if not os.path.exists(pred_file):
        print(f"预测文件不存在: {pred_file}")
        return

    df_pred = pd.read_csv(pred_file)

    # 加载元数据
    df_meta = pd.read_csv(L10_META)
    df_meta.columns = df_meta.columns.str.strip()

    # 合并元数据到 pair predictions
    # pair dataset 已经有 sub_name_i, sub_name_j, ring_pos, scaffold_id
    # 直接使用

    all_results = []

    for task_name in TASKS:
        print(f"\n--- {task_name} ---")
        df = df_pred[df_pred['task'] == task_name].copy()
        if len(df) == 0:
            continue

        df['abs_error'] = (df['pred_delta'] - df['delta_A']).abs()

        # 1. By substituent type
        print(f"\n  [By substituent type]")
        sub_type_stats = df.groupby('sub_type_i').agg(
            n_pairs=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
            median_error=('abs_error', 'median'),
            mean_delta=('delta_A', 'mean'),
            mean_abs_delta=('delta_A', lambda x: x.abs().mean()),
        ).sort_values('mean_error', ascending=False).reset_index()
        print(sub_type_stats.to_string(index=False))
        sub_type_stats['task'] = task_name
        sub_type_stats['analysis'] = 'substituent_type'
        all_results.append(sub_type_stats)

        # 2. By ring position
        print(f"\n  [By ring position]")
        pos_stats = df.groupby('ring_pos').agg(
            n_pairs=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
            median_error=('abs_error', 'median'),
        ).reset_index()
        print(pos_stats.to_string(index=False))
        pos_stats['task'] = task_name
        pos_stats['analysis'] = 'ring_position'
        all_results.append(pos_stats)

        # 3. By scaffold type
        print(f"\n  [By scaffold type] (top 10 hardest)")
        scaffold_stats = df.groupby('scaffold_id').agg(
            n_pairs=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
            median_error=('abs_error', 'median'),
            mean_delta=('delta_A', 'mean'),
            mean_abs_delta=('delta_A', lambda x: x.abs().mean()),
        ).sort_values('mean_error', ascending=False).reset_index()
        print(scaffold_stats.head(10).to_string(index=False))
        scaffold_stats['task'] = task_name
        scaffold_stats['analysis'] = 'scaffold_type'
        all_results.append(scaffold_stats)

        # 4. By heteroatom composition
        print(f"\n  [By heteroatom composition]")
        hetero_data = []
        for _, row in df.iterrows():
            hetero_i, n_hetero_i = get_heteroatom_info(row['smiles_i'])
            hetero_j, n_hetero_j = get_heteroatom_info(row['smiles_j'])
            fused_i = is_fused(row['smiles_i'])
            fused_j = is_fused(row['smiles_j'])

            n_N = hetero_i.get('N', 0) + hetero_j.get('N', 0)
            n_O = hetero_i.get('O', 0) + hetero_j.get('O', 0)
            n_S = hetero_i.get('S', 0) + hetero_j.get('S', 0)
            n_hetero_total = n_hetero_i + n_hetero_j

            hetero_data.append({
                'abs_error': row['abs_error'],
                'delta_A': row['delta_A'],
                'n_N': n_N, 'n_O': n_O, 'n_S': n_S,
                'n_hetero': n_hetero_total,
                'fused': fused_i or fused_j,
                'has_N': n_N > 0, 'has_O': n_O > 0, 'has_S': n_S > 0,
            })

        df_hetero = pd.DataFrame(hetero_data)

        # Fused vs non-fused
        print(f"  Fused vs Non-fused:")
        for is_f, label in [(True, 'Fused'), (False, 'Non-fused')]:
            subset = df_hetero[df_hetero['fused'] == is_f]
            if len(subset) > 0:
                print(f"    {label:10s}: n={len(subset):5d}, MAE={subset['abs_error'].mean():.4f}, "
                      f"|Δ|={subset['delta_A'].abs().mean():.4f}")

        # Has N / O / S
        print(f"  Heteroatom presence:")
        for atom, col in [('N', 'has_N'), ('O', 'has_O'), ('S', 'has_S')]:
            for has, label in [(True, f'has_{atom}'), (False, f'no_{atom}')]:
                subset = df_hetero[df_hetero[col] == has]
                if len(subset) > 0:
                    print(f"    {label:10s}: n={len(subset):5d}, MAE={subset['abs_error'].mean():.4f}")

        hetero_stats = df_hetero.groupby('fused').agg(
            n_pairs=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
        ).reset_index()
        hetero_stats['task'] = task_name
        hetero_stats['analysis'] = 'fused'
        all_results.append(hetero_stats)

        # 5. Easy vs hard substituent perturbations
        print(f"\n  [Easy vs Hard substituent perturbations]")
        # 按 sub_name 对分组
        sub_name_stats = df.groupby(['sub_name_i', 'sub_name_j']).agg(
            n_pairs=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
            mean_delta=('delta_A', 'mean'),
            mean_abs_delta=('delta_A', lambda x: x.abs().mean()),
        ).reset_index()

        # 只保留 n_pairs >= 10 的
        sub_name_stats = sub_name_stats[sub_name_stats['n_pairs'] >= 10]
        sub_name_stats = sub_name_stats.sort_values('mean_error')

        print(f"  Top 5 EASIEST perturbations (lowest MAE):")
        print(sub_name_stats.head(5).to_string(index=False))
        print(f"\n  Top 5 HARDEST perturbations (highest MAE):")
        print(sub_name_stats.tail(5).to_string(index=False))

        sub_name_stats['task'] = task_name
        sub_name_stats['analysis'] = 'substituent_pair'
        all_results.append(sub_name_stats)

        # 生成图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Substituent type MAE
        ax = axes[0, 0]
        top_types = sub_type_stats.head(15)
        ax.barh(top_types['sub_type_i'], top_types['mean_error'], alpha=0.7)
        ax.set_xlabel('Mean Absolute Error')
        ax.set_title(f'{task_name}: Error by Substituent Type')

        # Scaffold MAE
        ax = axes[0, 1]
        top_scaffolds = scaffold_stats.head(15)
        ax.barh(top_scaffolds['scaffold_id'], top_scaffolds['mean_error'], alpha=0.7)
        ax.set_xlabel('Mean Absolute Error')
        ax.set_title(f'{task_name}: Error by Scaffold Type')

        # Ring position MAE
        ax = axes[1, 0]
        ax.bar(pos_stats['ring_pos'].astype(str), pos_stats['mean_error'], alpha=0.7)
        ax.set_xlabel('Ring Position')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'{task_name}: Error by Ring Position')

        # Fused vs Non-fused
        ax = axes[1, 1]
        fused_stats = df_hetero.groupby('fused').agg(
            n=('abs_error', 'count'),
            mean_error=('abs_error', 'mean'),
            std_error=('abs_error', 'std'),
        ).reset_index()
        ax.bar(['Non-fused', 'Fused'], fused_stats['mean_error'],
               yerr=fused_stats['std_error'], capsize=5, alpha=0.7)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'{task_name}: Fused vs Non-fused')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'error_analysis_{task_name}.png'), dpi=150)
        plt.close()

    # 保存所有结果
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, 'error_analysis.csv'), index=False)

    # 生成报告
    report = "# Step 9: Substituent / Position Error Analysis\n\n"

    for task_name in TASKS:
        task_dfs = [df for df in all_results if df.get('task', pd.Series()).iloc[0] == task_name]
        if not task_dfs:
            continue

        report += f"## {task_name}\n\n"

        # Substituent type
        sub_type_df = next((df for df in task_dfs if df['analysis'].iloc[0] == 'substituent_type'), None)
        if sub_type_df is not None:
            report += "### By Substituent Type\n\n"
            report += "| sub_type | n_pairs | mean_error | median_error | mean_delta | mean_abs_delta |\n"
            report += "|----------|---------|------------|-------------|-----------|----------------|\n"
            for _, r in sub_type_df.iterrows():
                report += f"| {r['sub_type_i']} | {r['n_pairs']} | {r['mean_error']:.4f} | {r['median_error']:.4f} | {r['mean_delta']:.4f} | {r['mean_abs_delta']:.4f} |\n"
            report += "\n"

        # Scaffold type (top 5 hardest)
        scaffold_df = next((df for df in task_dfs if df['analysis'].iloc[0] == 'scaffold_type'), None)
        if scaffold_df is not None:
            report += "### By Scaffold Type (Top 5 hardest)\n\n"
            report += "| scaffold | n_pairs | mean_error | mean_abs_delta |\n"
            report += "|----------|---------|------------|----------------|\n"
            for _, r in scaffold_df.head(5).iterrows():
                report += f"| {r['scaffold_id']} | {r['n_pairs']} | {r['mean_error']:.4f} | {r['mean_abs_delta']:.4f} |\n"
            report += "\n"

        # Easy vs Hard perturbations
        sub_pair_df = next((df for df in task_dfs if df['analysis'].iloc[0] == 'substituent_pair'), None)
        if sub_pair_df is not None:
            report += "### Easy vs Hard Substituent Perturbations\n\n"
            report += "**Top 5 Easiest:**\n\n"
            report += "| sub_i | sub_j | n | mean_error | mean_abs_delta |\n"
            report += "|-------|-------|---|-----------|----------------|\n"
            for _, r in sub_pair_df.head(5).iterrows():
                report += f"| {r['sub_name_i']} | {r['sub_name_j']} | {r['n_pairs']} | {r['mean_error']:.4f} | {r['mean_abs_delta']:.4f} |\n"
            report += "\n**Top 5 Hardest:**\n\n"
            report += "| sub_i | sub_j | n | mean_error | mean_abs_delta |\n"
            report += "|-------|-------|---|-----------|----------------|\n"
            for _, r in sub_pair_df.tail(5).iterrows():
                report += f"| {r['sub_name_i']} | {r['sub_name_j']} | {r['n_pairs']} | {r['mean_error']:.4f} | {r['mean_abs_delta']:.4f} |\n"
            report += "\n"

    report += "## 文件列表\n\n"
    report += "- `error_analysis.csv` — 所有分析结果\n"
    report += "- `error_analysis_{task}.png` — 各任务的误差分析图\n"

    with open(os.path.join(OUTPUT_DIR, 'error_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
