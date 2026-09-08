# -*- coding: utf-8 -*-
"""
Step A: HOMA 新拆分 — 验证集 = lunci10 (分子级 20% 留出)

规则 (用户确认):
  - 训练 = collet a/b 全部 + lunci10 80% 分子 (分子级去重)
  - 验证 = lunci10 20% 分子 (组级留出, 同分子多环记录不跨 train/val, 验证分子绝不进训练)
  - e 系列仍全部丢弃 (污染)

输出: data/homa_l10val.csv (split: train / l10_val)
      data/homa_l10val_manifest.json
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

COLLET_HOMA = "/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_homa_0716.csv"
L10_PATH = "/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv"
SPLIT_SEED = 2026
L10_TEST_FRAC = 0.20


def canon(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_list(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return [int(i) for i in x]


def main():
    collet = pd.read_csv(COLLET_HOMA)
    collet["mol_id"] = collet["smiles"].apply(canon)
    collet["prefix"] = collet["New_ID"].astype(str).str.extract(r"^([a-zA-Z]+)")
    ab = collet[collet["prefix"].isin(["a", "b"])].copy()
    ab = ab[ab["mol_id"].notna()].reset_index(drop=True)
    print(f"collet a/b: {len(ab)}")

    l10 = pd.read_csv(L10_PATH)
    l10["mol_id"] = l10["smiles"].apply(canon)
    l10 = l10[l10["mol_id"].notna() & l10["HOMA"].notna()].reset_index(drop=True)
    # 剔除与 collet a/b 重叠的 l10 分子 (保留 collet 版本)
    l10 = l10[~l10["mol_id"].isin(set(ab["mol_id"]))].reset_index(drop=True)
    print(f"l10 (HOMA 非空, 去 collet 重叠): {len(l10)}")

    # l10 分子级 80/20
    uniq_mols = l10.drop_duplicates("mol_id").reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=L10_TEST_FRAC, random_state=SPLIT_SEED)
    tr_pos, va_pos = next(gss.split(np.arange(len(uniq_mols)),
                                    groups=uniq_mols["mol_id"].values))
    tr_mols = set(uniq_mols["mol_id"].iloc[tr_pos])
    va_mols = set(uniq_mols["mol_id"].iloc[va_pos])
    assert not (tr_mols & va_mols)
    l10_tr = l10[l10["mol_id"].isin(tr_mols)]
    l10_va = l10[l10["mol_id"].isin(va_mols)]
    print(f"l10 分子: train {len(tr_mols)} / val {len(va_mols)}; "
          f"l10 记录: train {len(l10_tr)} / val {len(l10_va)}")

    rows = []
    for _, r in ab.iterrows():
        rows.append((r["smiles"], parse_list(r["atom_on_ring"]), float(r["homa_value"]),
                     r["mol_id"], "collet", "train"))
    for _, r in l10_tr.iterrows():
        rows.append((r["smiles"], parse_list(r["ring_atoms"]), float(r["HOMA"]),
                     r["mol_id"], "l10", "train"))
    for _, r in l10_va.iterrows():
        rows.append((r["smiles"], parse_list(r["ring_atoms"]), float(r["HOMA"]),
                     r["mol_id"], "l10", "l10_val"))
    df = pd.DataFrame(rows, columns=["smiles", "atom_on_ring", "y", "mol_id", "source", "split"])
    df = df[df["y"].notna()].reset_index(drop=True)

    tr_m = set(df[df["split"] == "train"]["mol_id"])
    va_m = set(df[df["split"] == "l10_val"]["mol_id"])
    assert not (tr_m & va_m), "train/val 分子泄漏!"
    assert len(df[df["split"] == "l10_val"]["mol_id"].drop_duplicates()) == len(va_m)

    out = os.path.join(DATA_DIR, "homa_l10val.csv")
    df.to_csv(out, index=False)

    manifest = {
        "collet_ab_train": int(len(ab)),
        "l10_train_molecules": int(len(tr_mols)),
        "l10_val_molecules": int(len(va_mols)),
        "l10_train_records": int(len(l10_tr)),
        "l10_val_records": int(len(l10_va)),
        "total_train_records": int(len(df[df["split"] == "train"])),
        "total_val_records": int(len(df[df["split"] == "l10_val"])),
        "split_seed": SPLIT_SEED, "l10_test_frac": L10_TEST_FRAC,
    }
    with open(os.path.join(DATA_DIR, "homa_l10val_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] {json.dumps(manifest, indent=2)}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
