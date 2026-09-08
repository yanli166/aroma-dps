
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
Hard Ring-Type Table

找出最难的 10-20 种环类型, 分析为什么难。

指标:
  - 每种 ring type 的预测误差 (MAE / RMSE)
  - 在全数据集中的频率 (rare = hard)
  - heteroatom 组成
  - ring size
  - fused/non-fused
  - HOMA target 分布

输出:
  - hard_ring_types_table.csv
  - hard_ring_types.png (可视化)
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
sys.path.insert(0, PROJ_ROOT)
from generalization_test.code.ring_utils import identify_ring_type

L10_CSV = '_PROJ_ROOT/lunci10/lunci10-expanded-test.csv'
RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/learning_curve'
OUTPUT_DIR = '_PROJ_ROOT + "/code_end"/results/ood_difficulty_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_heteroatom_info(smiles):
    """获取杂原子信息"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, 0
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    hetero = [a for a in atoms if a != 'C' and a != 'H']
    return dict(Counter(hetero)), len(hetero)


def is_fused(smiles):
    """判断是否为稠合环"""
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


def get_ring_sizes(smiles):
    """获取环大小"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    ri = mol.GetRingInfo()
    return [len(r) for r in ri.AtomRings()]


def main():
    print("=" * 60)
    print("Hard Ring-Type Table")
    print("=" * 60)

    df_l10 = pd.read_csv(L10_CSV)
    task_cols = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}

    # 为每个 SMILES 计算环类型
    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}

    # 统计每种 ring type 的频率
    rt_counter = Counter(smi_to_rt.values())
    total_smiles = len(smiles_list)

    # 收集所有 fraction 的预测结果
    all_errors = []
    for task_name, target_col in task_cols.items():
        for frac in [5, 10, 20, 30, 50, 70]:
            pred_path = os.path.join(RESULTS_DIR, task_name, f'frac_{frac}', 'predictions.csv')
            if not os.path.exists(pred_path):
                continue
            df_pred = pd.read_csv(pred_path)
            # 需要关联 SMILES, 但 predictions.csv 只有 true/pred
            # 从日志或数据文件获取 SMILES
            # 实际上 l10_test CSV 保存了 SMILES, predictions.csv 没有
            # 我们需要从 l10_test CSV 获取
            test_csv = f'/tmp/lc_rerun/l10_test_{task_name}_{frac/100}.csv'
            if not os.path.exists(test_csv):
                # 尝试其他路径
                import glob
                test_csvs = glob.glob(f'/tmp/lc_rerun/l10_test_{task_name}_0.{frac}.csv')
                if test_csvs:
                    test_csv = test_csvs[0]
                else:
                    continue

            df_test = pd.read_csv(test_csv)
            if len(df_test) != len(df_pred):
                continue

            df_test['pred'] = df_pred['pred'].values
            tcol = {'HOMA': 'homa_value', 'NICS_1zz': 'NICS_value', 'MBCO': 'mbco_value'}[task_name]
            df_test['true_val'] = df_test[tcol]
            df_test['error'] = abs(df_test['pred'] - df_test['true_val'])
            df_test['ring_type'] = df_test['smiles'].map(smi_to_rt)
            df_test['task'] = task_name
            df_test['fraction'] = frac / 100.0

            all_errors.append(df_test[['smiles', 'ring_type', 'task', 'fraction',
                                        'true_val', 'pred', 'error']])

    if not all_errors:
        print("错误: 无预测结果, 请先运行 learning curve 实验")
        return

    df_errors = pd.concat(all_errors, ignore_index=True)
    print(f"\n总预测记录: {len(df_errors)}")

    # 按 ring type 聚合误差
    rt_stats = df_errors.groupby(['ring_type', 'task']).agg(
        n_samples=('error', 'count'),
        mean_error=('error', 'mean'),
        median_error=('error', 'median'),
        max_error=('error', 'max'),
        target_mean=('true_val', 'mean'),
        target_std=('true_val', 'std'),
    ).reset_index()

    # 添加频率信息
    rt_stats['freq_in_full'] = rt_stats['ring_type'].map(rt_counter)
    rt_stats['freq_pct'] = rt_stats['freq_in_full'] / total_smiles * 100

    # 添加结构信息 (取每个 ring type 的第一个 SMILES)
    rt_struct = {}
    for rt in rt_stats['ring_type'].unique():
        smis = [s for s, r in smi_to_rt.items() if r == rt]
        if smis:
            smi = smis[0]
            hetero, n_hetero = get_heteroatom_info(smi)
            rt_struct[rt] = {
                'example_smiles': smi,
                'n_heteroatoms': n_hetero,
                'hetero_composition': str(hetero),
                'fused': is_fused(smi),
                'ring_sizes': str(get_ring_sizes(smi)),
            }

    for rt, info in rt_struct.items():
        for k, v in info.items():
            mask = rt_stats['ring_type'] == rt
            rt_stats.loc[mask, k] = v

    # 按 HOMA 误差排序 (最难的在前)
    rt_homa = rt_stats[rt_stats['task'] == 'HOMA'].sort_values('mean_error', ascending=False)
    rt_homa.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_HOMA.csv'), index=False)

    rt_nics = rt_stats[rt_stats['task'] == 'NICS_1zz'].sort_values('mean_error', ascending=False)
    rt_nics.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_NICS.csv'), index=False)

    rt_mbco = rt_stats[rt_stats['task'] == 'MBCO'].sort_values('mean_error', ascending=False)
    rt_mbco.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_MBCO.csv'), index=False)

    rt_stats.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_all.csv'), index=False)

    # 打印 Top 15 最难 ring types (HOMA)
    print("\n" + "=" * 100)
    print("Top 15 最难 Ring Types (HOMA, 按 mean error 排序)")
    print("=" * 100)
    print(rt_homa[['ring_type', 'n_samples', 'mean_error', 'freq_in_full',
                     'freq_pct', 'n_heteroatoms', 'fused', 'target_mean']].head(15).to_string(index=False))

    # 可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    for i, (task_name, rt_data) in enumerate([('HOMA', rt_homa), ('NICS_1zz', rt_nics), ('MBCO', rt_mbco)]):
        ax = axes[i]
        top = rt_data.head(15)
        y_pos = np.arange(len(top))
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
        ax.barh(y_pos, top['mean_error'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{rt[:20]}" for rt in top['ring_type']], fontsize=8)
        ax.set_xlabel('Mean Absolute Error')
        ax.set_title(f'Top 15 Hard Ring Types ({task_name})')
        ax.invert_yaxis()

        # 标注频率
        for j, (_, row) in enumerate(top.iterrows()):
            ax.text(row['mean_error'] + 0.01, j, f"n={int(row['n_samples'])}, freq={row['freq_pct']:.1f}%",
                    va='center', fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'hard_ring_types.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n图表保存: {fig_path}")


if __name__ == '__main__':
    main()
