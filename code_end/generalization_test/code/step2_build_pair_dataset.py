
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
Step 2: 构建真正的 Pairwise Δ Dataset

在每一个相同 scaffold + ring_pos 组内, 对两个不同取代基分子 i,j 建立 pair:
  delta_A = A_i - A_j

要求:
  1. canonical pair (避免 (i,j) 和 (j,i) 重复)
  2. scaffold-grouped split (同一 scaffold 的所有 pair 进入同一 split)
  3. 每个 Ring_ID 独立配对 (同一分子的不同环不混淆)

输出:
  results/lunci10_delta_learning/01_pair_dataset/
    pair_dataset_homa.csv
    pair_dataset_nics.csv
    pair_dataset_mbco.csv
    pair_dataset_summary.md
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
L10_TEST = os.path.join(PROJ_ROOT, '..', 'lunci10/lunci10-test-corrected.csv')
L10_META = '_PROJ_ROOT/lunci10/lunci10-begin.csv'
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_v2/01_pair_dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = {
    'HOMA':     'HOMA',
    'NICS_1zz': 'NICS_ZZ',
    'MBCO':     'MBCO',
}


def load_lunci10_full():
    """加载 lunci10 测试数据, 为所有分子补充 scaffold/substituent 元数据

    匹配策略:
      1. New_ID 在 begin.csv 中 → 使用 begin.csv 的 ring_name, sub_name, ring_pos
      2. New_ID 不在 begin.csv 中 → 通过 Murcko scaffold 匹配到已知 ring_name,
         并通过取代基位置推断 ring_pos
    """
    df_l10 = pd.read_csv(L10_TEST, encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=['SMILES']).reset_index(drop=True)

    df_meta = pd.read_csv(L10_META)
    df_meta.columns = df_meta.columns.str.strip()

    # 合并
    df = df_l10.merge(
        df_meta[['no', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']],
        left_on='New_ID', right_on='no', how='left'
    )

    # 为未匹配的分子推断元数据
    unmatched_mask = df['ring_name'].isna()
    n_unmatched = unmatched_mask.sum()
    print(f"  未匹配元数据: {n_unmatched}/{len(df)} 行 ({df[unmatched_mask]['New_ID'].nunique()} 个分子)")

    if n_unmatched > 0:
        # 建立 Murcko scaffold → ring_name 映射 (从已匹配分子)
        matched = df[~unmatched_mask].drop_duplicates('New_ID')
        scaffold_to_ringname = {}
        for _, row in matched.iterrows():
            smi = row['SMILES']
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if sc and sc != '':
                scaffold_to_ringname[sc] = row['ring_name']

        # 为未匹配分子推断 ring_name
        inferred_ringname = {}
        inferred_ringpos = {}
        for new_id in df[unmatched_mask]['New_ID'].unique():
            smi = df[df['New_ID'] == new_id]['SMILES'].iloc[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                inferred_ringname[new_id] = 'unknown'
                inferred_ringpos[new_id] = -1
                continue

            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            ring_name = scaffold_to_ringname.get(sc, 'unknown')

            # 如果 Murcko scaffold 匹配失败, 尝试 aromatic core 匹配
            if ring_name == 'unknown':
                # 提取芳香核心
                aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
                if aromatic_atoms:
                    try:
                        submol = Chem.PathToSubmol(mol, aromatic_atoms)
                        core_smi = Chem.MolToSmiles(submol, canonical=True)
                        # 在已匹配分子中找相同 aromatic core
                        for _, mrow in matched.iterrows():
                            mmol = Chem.MolFromSmiles(mrow['SMILES'])
                            if mmol is None:
                                continue
                            maromatic = [a.GetIdx() for a in mmol.GetAtoms() if a.GetIsAromatic()]
                            if maromatic:
                                msubmol = Chem.PathToSubmol(mmol, maromatic)
                                mcore = Chem.MolToSmiles(msubmol, canonical=True)
                                if mcore == core_smi:
                                    ring_name = mrow['ring_name']
                                    break
                    except Exception:
                        pass

            inferred_ringname[new_id] = ring_name

            # 推断 ring_pos: 找到取代基连接的环原子位置
            # 简化方案: 用 Murcko scaffold 匹配的分子中, 最常见的 ring_pos
            if ring_name != 'unknown':
                same_scaffold = matched[matched['ring_name'] == ring_name]
                if len(same_scaffold) > 0:
                    # 尝试通过取代基连接点匹配
                    sc_mol = Chem.MolFromSmiles(sc) if sc else None
                    if sc_mol is not None:
                        # 找到取代基原子 (不在 scaffold 中的原子)
                        scaffold_atoms = set()
                        for atom in mol.GetAtoms():
                            if atom.GetIdx() in range(len(sc_mol.GetAtoms())):
                                scaffold_atoms.add(atom.GetIdx())

                        # 找到连接到 scaffold 的取代基原子
                        substituent_attachment = None
                        for atom in mol.GetAtoms():
                            if atom.GetIdx() not in scaffold_atoms:
                                for neighbor in atom.GetNeighbors():
                                    if neighbor.GetIdx() in scaffold_atoms:
                                        substituent_attachment = neighbor.GetIdx()
                                        break
                                if substituent_attachment is not None:
                                    break

                        # 简化: 用最常见的 ring_pos
                        inferred_ringpos[new_id] = same_scaffold['ring_pos'].mode().iloc[0]
                    else:
                        inferred_ringpos[new_id] = same_scaffold['ring_pos'].mode().iloc[0]
                else:
                    inferred_ringpos[new_id] = -1
            else:
                inferred_ringpos[new_id] = -1

        # 填充推断值
        for idx in df[unmatched_mask].index:
            new_id = df.loc[idx, 'New_ID']
            df.loc[idx, 'ring_name'] = inferred_ringname.get(new_id, 'unknown')
            df.loc[idx, 'ring_pos'] = inferred_ringpos.get(new_id, -1)
            df.loc[idx, 'sub_name'] = 'inferred'
            df.loc[idx, 'sub_type'] = 'inferred'

    # 清理
    df['ring_name'] = df['ring_name'].fillna('unknown')
    df['ring_pos'] = df['ring_pos'].fillna(-1).astype(int)
    df['sub_name'] = df['sub_name'].fillna('unknown')
    df['sub_type'] = df['sub_type'].fillna('unknown')

    return df


def build_pairs_for_task(df, target_col):
    """为指定任务构建 canonical pairs

    分组: ring_name + ring_pos + Ring_ID
    组内: 按 canonical SMILES 排序, 形成所有 (i,j) 组合, i < j
    """
    df_task = df.dropna(subset=[target_col]).copy()
    df_task = df_task.reset_index(drop=True)

    group_cols = ['ring_name', 'ring_pos', 'Ring_ID']

    pairs = []
    for (ring_name, ring_pos, ring_id), group in df_task.groupby(group_cols):
        if len(group) < 2:
            continue  # 至少 2 个分子才能形成 pair

        # 按 canonical SMILES 排序
        group = group.sort_values('SMILES').reset_index(drop=True)

        # 为每个分子计算 canonical SMILES
        for idx, row in group.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None:
                group.at[idx, 'canonical_smiles'] = Chem.MolToSmiles(mol)

        group = group.sort_values('canonical_smiles').reset_index(drop=True)

        # 形成所有 (i, j) 对, i < j (canonical pair, 避免重复)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row_i = group.iloc[i]
                row_j = group.iloc[j]

                # 跳过相同取代基的 pair
                if row_i['sub_name'] == row_j['sub_name'] and row_i['sub_name'] != 'unknown':
                    continue

                pairs.append({
                    'scaffold_id': ring_name,
                    'ring_pos': ring_pos,
                    'ring_id': ring_id,
                    'scaffold_group': f"{ring_name}_pos{ring_pos}_ring{ring_id}",
                    'smiles_i': row_i['SMILES'],
                    'smiles_j': row_j['SMILES'],
                    'canonical_smiles_i': row_i['canonical_smiles'],
                    'canonical_smiles_j': row_j['canonical_smiles'],
                    'new_id_i': row_i['New_ID'],
                    'new_id_j': row_j['New_ID'],
                    'sub_name_i': row_i['sub_name'],
                    'sub_name_j': row_j['sub_name'],
                    'sub_type_i': row_i['sub_type'],
                    'sub_type_j': row_j['sub_type'],
                    'target_ring_i': row_i['Ring_Atoms'],
                    'target_ring_j': row_j['Ring_Atoms'],
                    'A_i': float(row_i[target_col]),
                    'A_j': float(row_j[target_col]),
                    'delta_A': float(row_i[target_col]) - float(row_j[target_col]),
                })

    df_pairs = pd.DataFrame(pairs)
    return df_pairs


def main():
    print("=" * 70)
    print("Step 2: 构建 Pairwise Δ Dataset")
    print("=" * 70)

    df = load_lunci10_full()
    print(f"\n总样本数: {len(df)}")
    print(f"唯一分子数: {df['New_ID'].nunique()}")
    print(f"ring_name 类型数: {df['ring_name'].nunique()}")
    print(f"scaffold+ring_pos 组数: {df.groupby(['ring_name', 'ring_pos']).ngroups}")
    print(f"scaffold+ring_pos+Ring_ID 组数: {df.groupby(['ring_name', 'ring_pos', 'Ring_ID']).ngroups}")

    # 为每个任务构建 pairs
    all_stats = []
    for task_name, col in TASKS.items():
        print(f"\n--- {task_name} ---")
        df_pairs = build_pairs_for_task(df, col)

        out_path = os.path.join(OUTPUT_DIR, f'pair_dataset_{task_name.lower().replace("_1zz","").replace("nics","nics")}.csv')
        # 统一命名
        out_path = os.path.join(OUTPUT_DIR, f'pair_dataset_{task_name.lower().replace("_1zz","")}.csv')
        df_pairs.to_csv(out_path, index=False)

        n_pairs = len(df_pairs)
        n_groups = df_pairs['scaffold_group'].nunique()
        n_scaffolds = df_pairs['scaffold_id'].nunique()

        print(f"  Pairs: {n_pairs}")
        print(f"  Groups (scaffold+pos+ring): {n_groups}")
        print(f"  Scaffolds: {n_scaffolds}")

        # 每 scaffold pair 数
        scaffold_pair_counts = df_pairs.groupby('scaffold_id').size().sort_values(ascending=False)
        print(f"  Pairs per scaffold: min={scaffold_pair_counts.min()}, "
              f"max={scaffold_pair_counts.max()}, "
              f"mean={scaffold_pair_counts.mean():.1f}, "
              f"median={scaffold_pair_counts.median():.1f}")

        # delta 统计
        delta = df_pairs['delta_A']
        print(f"  Δ stats: mean={delta.mean():.4f}, std={delta.std(ddof=1):.4f}, "
              f"min={delta.min():.4f}, max={delta.max():.4f}")
        print(f"  |Δ| stats: mean={delta.abs().mean():.4f}, median={delta.abs().median():.4f}")

        # 检查极端 imbalance
        print(f"\n  Pair count per scaffold:")
        for sc, cnt in scaffold_pair_counts.items():
            print(f"    {sc:25s}: {cnt:6d} pairs")

        all_stats.append({
            'task': task_name,
            'n_pairs': n_pairs,
            'n_groups': n_groups,
            'n_scaffolds': n_scaffolds,
            'delta_mean': delta.mean(),
            'delta_std': delta.std(ddof=1),
            'delta_min': delta.min(),
            'delta_max': delta.max(),
            'abs_delta_mean': delta.abs().mean(),
            'abs_delta_median': delta.abs().median(),
        })

    # 生成 summary
    df_stats = pd.DataFrame(all_stats)

    report = f"""# Pairwise Δ Dataset 构建报告

## 数据来源

- 测试数据: `data1_end/lunci10-test.csv` ({df['New_ID'].nunique()} 个分子, {len(df)} 行)
- 元数据: `lunci10/lunci10-begin.csv`
- 未匹配分子: 通过 Murcko scaffold 匹配 + 芳香核心匹配推断 ring_name

## 分组策略

- **化学分组单位**: `scaffold + ring_pos + Ring_ID`
  - `scaffold` (ring_name): 分子骨架类型 (pyridine, indole, ...)
  - `ring_pos`: 取代基位置 (1-5)
  - `Ring_ID`: 分子内目标环编号 (多环分子有多个环)
- **Canonical pair**: 组内按 canonical SMILES 排序, 只保留 (i, j) where i < j
- **跳过相同取代基**: sub_name 相同的分子不配对

## Pair 统计

| 任务 | Pairs | Groups | Scaffolds | Δ mean | Δ std | Δ min | Δ max | |Δ| mean | |Δ| median |
|------|-------|--------|-----------|--------|-------|--------|--------|---------|-----------|
"""
    for _, s in df_stats.iterrows():
        report += (f"| {s['task']} | {s['n_pairs']} | {s['n_groups']} | {s['n_scaffolds']} | "
                   f"{s['delta_mean']:.4f} | {s['delta_std']:.4f} | "
                   f"{s['delta_min']:.4f} | {s['delta_max']:.4f} | "
                   f"{s['abs_delta_mean']:.4f} | {s['abs_delta_median']:.4f} |\n")

    report += f"""
## Split 策略

- **scaffold-grouped split**: 同一 scaffold 的所有 pairs 必须进入同一个 split
- 使用 `GroupShuffleSplit` / `GroupKFold` (group = scaffold_id)
- 5 seeds: 42, 123, 456, 789, 2024
- 所有后续步骤 (Step 3-6) 使用完全一致的 split

## 文件列表

- `pair_dataset_homa.csv` — HOMA pair dataset
- `pair_dataset_nics.csv` — NICS(1)zz pair dataset
- `pair_dataset_mbco.csv` — MBCO pair dataset
- `pair_dataset_summary.md` — 本报告
"""
    with open(os.path.join(OUTPUT_DIR, 'pair_dataset_summary.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'=' * 70}")
    print("Pair dataset 构建完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
