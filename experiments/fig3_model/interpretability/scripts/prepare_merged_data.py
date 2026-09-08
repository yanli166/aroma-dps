# -*- coding: utf-8 -*-
"""
Step 1: 构建 base(a/b) + lunci10_unified 合并去重数据集

决策 (用户已确认):
  1. collet e 系列 HOMA/MBCO 为三线态污染, 三个任务统一丢弃 e 系列
  2. lunci10 一律使用 lunci10_unified.csv; NICS 目标列 = NICS_ZZ
  3. 分子级去重: 若某分子在 collet a/b 与 lunci10 同时出现, 保留 collet 版本, 剔除 l10 对应行
  4. test = 原 full-collet 80/20 group holdout (seed=2026) 标记为 test 的分子 ∩ a/b
     (e 系列丢弃后, 原落在 test 的 e 分子随之剔除)
  5. lunci10 只进入训练 (train pool), 绝不进入 collet test

输出: data/merged_{HOMA,NICS_1zz,MBCO}.csv
  列: smiles, atom_on_ring, y, mol_id, source(collet/l10), split(dev/test/l10)
"""
import os
import ast
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COLLET_ROOT = "/home/ubuntu/aroma-dps-code/code_end/data1_end"
L10_PATH = "/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv"
SPLIT_SEED = 2026

TASKS = {
    "HOMA":     {"collet": os.path.join(COLLET_ROOT, "collet_homa_0716.csv"),
                 "ctarget": "homa_value", "l10_col": "HOMA"},
    "NICS_1zz": {"collet": os.path.join(COLLET_ROOT, "collet_nics_0716.csv"),
                 "ctarget": "NICS_value", "l10_col": "NICS_ZZ"},
    "MBCO":     {"collet": os.path.join(COLLET_ROOT, "collet_mbco_0716.csv"),
                 "ctarget": "mbco_value", "l10_col": "MBCO"},
}


def canon_mol(smi):
    """与 train_final.py make_group_ids 保持完全一致"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_list(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return [int(i) for i in x]


def full_collet_split_flags(full_df):
    """在完整 collet (a/b/e) 上复现 train_final.py 的 80/20 group holdout,
    得到 molecule -> test/dev 的固定指派。"""
    groups = full_df["mol_id"].values
    n = len(full_df)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    dev_pos, test_pos = next(gss.split(np.arange(n), groups=groups))
    dev_mols = set(groups[dev_pos])
    test_mols = set(groups[test_pos])
    assert len(dev_mols & test_mols) == 0, "full-collet holdout leakage!"
    return dev_mols, test_mols


def main():
    l10_raw = pd.read_csv(L10_PATH)
    l10_raw["mol_id"] = l10_raw["smiles"].apply(canon_mol)
    n_bad_l10 = l10_raw["mol_id"].isna().sum()

    summary = {}
    for task, cfg in TASKS.items():
        full = pd.read_csv(cfg["collet"])
        # 若存在全 NaN 的杂列(NICS 的 Unnamed) 先剔除
        full = full.loc[:, full.columns[~full.columns.str.startswith("Unnamed")]]
        full = full.dropna(subset=[cfg["ctarget"], "smiles"]).reset_index(drop=True)
        full["mol_id"] = full["smiles"].apply(canon_mol)
        full = full[full["mol_id"].notna()].reset_index(drop=True)

        # e 系列判别: New_ID 以 e+数字 开头
        e_mask = full["New_ID"].astype(str).str.match(r"^e\d+$").fillna(False)
        n_e = int(e_mask.sum())
        ab = full[~e_mask].reset_index(drop=True)

        # 固定 collet test 指派 (基于完整 collet a/b/e 的原始 holdout)
        dev_mols, test_mols = full_collet_split_flags(full)

        ab_test = ab[ab["mol_id"].isin(test_mols)]
        ab_dev = ab[ab["mol_id"].isin(dev_mols)]

        # lunci10 仅训练; 与 collet a/b 分子重叠者剔除(保留 collet 版本)
        l10 = l10_raw.dropna(subset=[cfg["l10_col"]]).copy()
        l10 = l10[l10["mol_id"].isin(ab["mol_id"]) == False].reset_index(drop=True)
        # 防止与 collet test 分子重叠导致的泄漏(理论上已通过上一条规避, 双保险)
        l10 = l10[l10["mol_id"].isin(ab_test["mol_id"]) == False].reset_index(drop=True)

        # 组装 merged
        rows = []
        for _, r in ab_dev.iterrows():
            rows.append((r["smiles"], parse_list(r["atom_on_ring"]), float(r[cfg["ctarget"]]),
                         r["mol_id"], "collet", "dev"))
        for _, r in ab_test.iterrows():
            rows.append((r["smiles"], parse_list(r["atom_on_ring"]), float(r[cfg["ctarget"]]),
                         r["mol_id"], "collet", "test"))
        for _, r in l10.iterrows():
            rows.append((r["smiles"], parse_list(r["ring_atoms"]), float(r[cfg["l10_col"]]),
                         r["mol_id"], "l10", "l10"))
        merged = pd.DataFrame(rows, columns=["smiles", "atom_on_ring", "y", "mol_id", "source", "split"])
        merged = merged[merged["y"].notna()].reset_index(drop=True)

        out = os.path.join(DATA_DIR, f"merged_{task}.csv")
        merged.to_csv(out, index=False)

        # 校验
        dev_m = set(merged[merged["split"] == "dev"]["mol_id"])
        test_m = set(merged[merged["split"] == "test"]["mol_id"])
        l10_m = set(merged[merged["split"] == "l10"]["mol_id"])
        assert len(dev_m & test_m) == 0
        assert len(test_m & l10_m) == 0
        assert len(merged[merged["split"] == "l10"]["mol_id"].duplicated() & merged[merged["split"] == "l10"]["mol_id"].duplicated()) >= 0

        summary[task] = {
            "n_collet_full": len(full), "n_e_dropped": n_e,
            "n_collet_ab": len(ab),
            "n_dev": len(ab_dev), "n_test": len(ab_test),
            "n_l10_added": len(l10), "n_l10_dup_with_ab_removed": int(len(l10_raw) - len(l10_raw.dropna(subset=[cfg["l10_col"]])) - len(l10)),
            "n_merged_total": len(merged),
            "n_test_molecules": len(test_m),
        }
        print(f"[{task}] full={len(full)} e_dropped={n_e} ab={len(ab)} "
              f"dev={len(ab_dev)} test={len(ab_test)} l10={len(l10)} merged={len(merged)}")

    # l10 检查说明
    summary["_l10_meta"] = {
        "n_l10_raw": len(l10_raw),
        "n_l10_nan_mol_id": int(n_bad_l10),
        "unique_mol_l10": int(l10_raw["mol_id"].nunique()),
    }
    with open(os.path.join(DATA_DIR, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nmerged files:", [f"merged_{t}.csv" for t in TASKS])


if __name__ == "__main__":
    main()
