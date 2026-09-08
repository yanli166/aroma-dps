# -*- coding: utf-8 -*-
"""
Step A (generic): NICS_1zz / MBCO 新拆分 — 验证集 = lunci10 20% 分子 (复用 HOMA l10 留出)

规则 (与 HOMA_l10val 完全一致, 用户确认):
  - 训练 = collet a/b 全部 + lunci10 80% 分子 (分子级去重)
  - 验证 = lunci10 20% 分子; 为保证跨任务验证分子一致, 直接复用 HOMA l10val
    的分子级 train/val 划分 (data/homa_l10val.csv 的 source=='l10' 子集),
    因此 NICS/MBCO 的验证分子与 HOMA 完全相同 (满足 scaffold/molecule 一致性约束)。
  - 只有 HOMA 数据集中不存在的 l10 额外分子才允许进入 train 或 (若与 collet 重叠) 丢弃。
  - e 系列仍全部丢弃 (HOMA/MBCO 污染; NICS 亦一并不用, 保证三任务数据口径一致)。

用法: python3 prepare_task_l10val.py --task NICS_1zz
输出: data/{task}_l10val.csv + data/{task}_l10val_manifest.json
"""
import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
HOMA_L10VAL_CSV = os.path.join(DATA_DIR, "homa_l10val.csv")
L10_PATH = "/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv"
COLLET_DIR = "/home/ubuntu/aroma-dps-code/code_end/data1_end"

# task -> (collet csv, collet label col, l10 label col, output task 名)
TASKS = {
    "NICS_1zz": ("collet_nics_0716.csv", "NICS_value", "NICS_ZZ"),
    "MBCO": ("collet_mbco_0716.csv", "mbco_value", "MBCO"),
}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    args = ap.parse_args()
    collet_csv, collet_col, l10_col = TASKS[args.task]
    task = args.task

    # 复用 HOMA 的 l10 分子级划分 (保证跨任务一致)
    homa = pd.read_csv(HOMA_L10VAL_CSV)
    homa_l10 = homa[homa["source"] == "l10"]
    val_mols = set(homa_l10[homa_l10["split"] == "l10_val"]["mol_id"])
    train_mols = set(homa_l10[homa_l10["split"] == "train"]["mol_id"])
    print(f"HOMA l10 分子划分: train {len(train_mols)} / val {len(val_mols)}")

    collet = pd.read_csv(os.path.join(COLLET_DIR, collet_csv))
    collet["mol_id"] = collet["smiles"].apply(canon)
    collet["prefix"] = collet["New_ID"].astype(str).str.extract(r"^([a-zA-Z]+)")
    ab = collet[collet["prefix"].isin(["a", "b"])].copy()
    ab = ab[ab["mol_id"].notna() & ab[collet_col].notna()].reset_index(drop=True)
    print(f"collet a/b ({task}): {len(ab)}")

    l10 = pd.read_csv(L10_PATH)
    l10["mol_id"] = l10["smiles"].apply(canon)
    l10 = l10[l10["mol_id"].notna() & l10[l10_col].notna()].reset_index(drop=True)
    ab_mols = set(ab["mol_id"])
    # 丢弃规则: 仅 HOMA 池之外的 l10 分子若与 collet ab 重叠才丢弃 (collet 版本优先);
    # HOMA 池内的分子 (train/val 划分已定) 一律保留, 保证跨任务 val 分子完全一致
    keep = l10["mol_id"].isin(train_mols | val_mols)
    overlap_extra = (~keep) & l10["mol_id"].isin(ab_mols)
    print(f"l10 ({task}, 非空, {len(l10)} 条) 中与 collet 重叠的额外分子丢弃: "
          f"{int(overlap_extra.sum())} 条")
    l10 = l10[~(overlap_extra)].reset_index(drop=True)
    l10_tr = l10[l10["mol_id"].isin(train_mols)]
    l10_va = l10[l10["mol_id"].isin(val_mols)]
    extra_tr = l10[~l10["mol_id"].isin(train_mols | val_mols)]
    # HOMA 池之外的 l10 分子归入 train (不改变任何 val 分子)
    if len(extra_tr):
        print(f"  HOMA 池之外 l10 分子 (归 train): {extra_tr['mol_id'].nunique()} "
              f"分子 / {len(extra_tr)} 条")
    print(f"l10 ({task}) 记录: train {len(l10_tr)} / val {len(l10_va)}; "
          f"train 分子 {l10_tr['mol_id'].nunique()} / val 分子 {l10_va['mol_id'].nunique()}")

    rows = []
    for _, r in ab.iterrows():
        rows.append((r["smiles"], parse_list(r["atom_on_ring"]), float(r[collet_col]),
                     r["mol_id"], "collet", "train"))
    for _, r in l10_tr.iterrows():
        rows.append((r["smiles"], parse_list(r["ring_atoms"]), float(r[l10_col]),
                     r["mol_id"], "l10", "train"))
    for _, r in l10_va.iterrows():
        rows.append((r["smiles"], parse_list(r["ring_atoms"]), float(r[l10_col]),
                     r["mol_id"], "l10", "l10_val"))
    df = pd.DataFrame(rows, columns=["smiles", "atom_on_ring", "y", "mol_id",
                                     "source", "split"])
    df = df[df["y"].notna()].reset_index(drop=True)

    va_m = set(df[df["split"] == "l10_val"]["mol_id"])
    assert not (set(df[df["split"] == "train"]["mol_id"]) & va_m), "train/val 分子泄漏!"
    assert va_m <= val_mols, "val 分子必须全部来自 HOMA val 集合!"

    out = os.path.join(DATA_DIR, f"{task.lower()}_l10val.csv")
    df.to_csv(out, index=False)
    manifest = {
        "task": task,
        "collet_ab_train": int(len(ab)),
        "l10_train_molecules": int(l10_tr["mol_id"].nunique()),
        "l10_val_molecules": int(l10_va["mol_id"].nunique()),
        "l10_val_molecules_in_homa": int(len(va_m)),
        "l10_train_records": int(len(l10_tr)),
        "l10_val_records": int(len(l10_va)),
        "extra_l10_train_records": int(len(extra_tr)),
        "total_train_records": int(len(df[df["split"] == "train"])),
        "total_val_records": int(len(df[df["split"] == "l10_val"])),
    }
    with open(os.path.join(DATA_DIR, f"{task.lower()}_l10val_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] {json.dumps(manifest, indent=2)}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
