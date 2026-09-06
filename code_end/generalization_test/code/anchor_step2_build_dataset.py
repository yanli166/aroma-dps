
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
Anchor-based Δ-Learning Step 2: 构建 anchor pair dataset

对每个 (ring_name, ring_pos, Ring_ID) group 和每个 anchor (F/Cl/OMe):
  - 找到 anchor 分子 (reference)
  - 对组内其他取代基分子 (substituent_i), 构建 pair: (reference, substituent_i)
  - delta_A = A(substituent_i) - A(reference)

输出:
  results/lunci10_anchor_delta_final/anchor_pair_dataset.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJ_ROOT = '_PROJ_ROOT'
LUNCI10_BEGIN = os.path.join(PROJ_ROOT, 'lunci10/lunci10-begin.csv')
LUNCI10_TEST = os.path.join(PROJ_ROOT, 'lunci10/lunci10-test-corrected.csv')

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'code_end/results/lunci10_anchor_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANCHORS = ['F', 'Cl', 'OMe']

TASK_COL_MAP = {
    'HOMA':      'HOMA',
    'NICS_1zz':  'NICS_ZZ',
    'MBCO':      'MBCO',
}


def load_and_merge():
    """加载 lunci10 测试数据并合并元数据"""
    df_test = pd.read_csv(LUNCI10_TEST, encoding='utf-8-sig')
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.loc[:, ~df_test.columns.str.startswith('Unnamed')]
    df_test = df_test.dropna(subset=['SMILES']).reset_index(drop=True)

    df_meta = pd.read_csv(LUNCI10_BEGIN, encoding='utf-8-sig')
    df_meta.columns = df_meta.columns.str.strip()

    # 合并
    df = df_test.merge(
        df_meta[['no', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']],
        left_on='New_ID', right_on='no', how='left'
    )

    # 为未匹配的分子推断元数据 (使用 Murcko scaffold)
    unmatched_mask = df['ring_name'].isna()
    if unmatched_mask.sum() > 0:
        matched = df[~unmatched_mask].drop_duplicates('New_ID')
        scaffold_to_ringname = {}
        for _, row in matched.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is None:
                continue
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if sc:
                scaffold_to_ringname[sc] = row['ring_name']

        for idx in df[unmatched_mask].index:
            smi = df.loc[idx, 'SMILES']
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                df.loc[idx, 'ring_name'] = 'unknown'
                df.loc[idx, 'ring_pos'] = -1
                df.loc[idx, 'sub_name'] = 'unknown'
                df.loc[idx, 'sub_type'] = 'unknown'
                continue
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            rn = scaffold_to_ringname.get(sc, 'unknown')
            df.loc[idx, 'ring_name'] = rn
            df.loc[idx, 'ring_pos'] = -1 if rn == 'unknown' else matched[matched['ring_name'] == rn]['ring_pos'].mode().iloc[0]
            df.loc[idx, 'sub_name'] = 'inferred'
            df.loc[idx, 'sub_type'] = 'inferred'

    df['ring_name'] = df['ring_name'].fillna('unknown')
    df['ring_pos'] = df['ring_pos'].fillna(-1).astype(int)
    df['sub_name'] = df['sub_name'].fillna('unknown')
    df['sub_type'] = df['sub_type'].fillna('unknown')
    return df


def build_anchor_pairs(df, anchor, task_col):
    """为指定 anchor 和 task 构建 anchor pairs

    分组: ring_name + ring_pos + Ring_ID
    组内: 找到 anchor 分子 (reference), 对其他取代基分子构建 pair
    """
    df_task = df.dropna(subset=[task_col]).copy()
    df_task = df_task.reset_index(drop=True)

    group_cols = ['ring_name', 'ring_pos', 'Ring_ID']
    pairs = []

    for (ring_name, ring_pos, ring_id), group in df_task.groupby(group_cols):
        # 找到 anchor 分子
        ref_mask = group['sub_name'] == anchor
        if not ref_mask.any():
            continue  # 该组没有 anchor, 跳过

        # 取第一个 anchor 分子作为 reference
        ref_row = group[ref_mask].iloc[0]
        ref_A = float(ref_row[task_col])

        # 对组内其他取代基分子构建 pair
        for _, sub_row in group[~ref_mask].iterrows():
            if sub_row['sub_name'] == anchor:
                continue
            if sub_row['sub_name'] in ['unknown', 'inferred']:
                continue
            sub_A = float(sub_row[task_col])
            pairs.append({
                'anchor': anchor,
                'ring_name': ring_name,
                'ring_pos': ring_pos,
                'ring_id': ring_id,
                'scaffold_group': f"{ring_name}_pos{ring_pos}_ring{ring_id}",
                'scaffold_id': ring_name,  # 用于 grouped split
                'smiles_ref': ref_row['SMILES'],
                'smiles_sub': sub_row['SMILES'],
                'new_id_ref': ref_row['New_ID'],
                'new_id_sub': sub_row['New_ID'],
                'sub_name_ref': anchor,
                'sub_name_sub': sub_row['sub_name'],
                'sub_type_sub': sub_row['sub_type'],
                'target_ring_ref': ref_row['Ring_Atoms'],
                'target_ring_sub': sub_row['Ring_Atoms'],
                'A_ref': ref_A,
                'A_sub': sub_A,
                'delta_A': sub_A - ref_A,
                'task': task_col,
            })

    return pd.DataFrame(pairs)


def main():
    print("=" * 70)
    print("Anchor-based Pair Dataset Construction")
    print("=" * 70)

    df = load_and_merge()
    print(f"Loaded {len(df)} ring-level samples ({df['New_ID'].nunique()} molecules)")
    print(f"Anchors: {ANCHORS}")
    print()

    all_pairs = []
    for task_name, task_col in TASK_COL_MAP.items():
        for anchor in ANCHORS:
            df_pairs = build_anchor_pairs(df, anchor, task_col)
            df_pairs['task'] = task_name
            all_pairs.append(df_pairs)
            print(f"  [{task_name} | anchor={anchor}] {len(df_pairs)} pairs, "
                  f"{df_pairs['scaffold_id'].nunique()} ring types")

    df_all = pd.concat(all_pairs, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, 'anchor_pair_dataset.csv')
    df_all.to_csv(out_path, index=False)
    print()
    print(f"Total pairs: {len(df_all)}")
    print(f"Saved: {out_path}")

    # 统计
    print()
    print("=== Pair 统计 (by task × anchor) ===")
    summary = df_all.groupby(['task', 'anchor']).agg(
        n_pairs=('delta_A', 'count'),
        n_ring_types=('scaffold_id', 'nunique'),
        delta_mean=('delta_A', 'mean'),
        delta_std=('delta_A', 'std'),
        delta_min=('delta_A', 'min'),
        delta_max=('delta_A', 'max'),
        abs_delta_mean=('delta_A', lambda x: np.abs(x).mean()),
        abs_delta_median=('delta_A', lambda x: np.abs(x).median()),
    ).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, 'anchor_pair_summary.csv'), index=False)


if __name__ == '__main__':
    main()
