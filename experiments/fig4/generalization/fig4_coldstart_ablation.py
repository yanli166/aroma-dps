#!/usr/bin/env python3
"""Cold-start ablation: does removing the 'e' block (wrong values) from base
improve the 0%-exposure (base-only) locked-test generalization of Fig.4c?

Only frac=0.0 cold-start is retrained for each task, identical to fig4_retrain_v2.py
exp4 (same locked test molecules, same train/val split seed, same removal of
l10-overlapping molecules), EXCEPT the entire New_ID 'e*' block is dropped
from the base training set.

Control (e-block kept) numbers are taken from the completed quick run:
  HOMA 0.0684 | NICS_ZZ -0.3745 | MBCO 0.6926  (locked test 423 rec / 272 mol)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/aroma-dps-code")
sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
# import pipeline internals (module top-level parse_args tolerates plain argv)
from fig4_retrain_v2 import (TASK_CONFIG, OUT_DIR, load_base_dataset,
                             load_lunci10_dataset, build_and_cache, derive_rf10,
                             train_model, predict_model, metrics, value_array)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[coldstart_ablation] device={device}")

l10 = load_lunci10_dataset()
l10_cache = build_and_cache(l10, "lunci10")

# locked test molecules identical to quick run (seed-42 based, 272 mol)
lock_ids = pd.read_csv(OUT_DIR / "04_lunci10_adaptation" / "locked_test_ids.csv")
locked_mols = set(lock_ids["canonical_smiles"].unique())
lock_idx = l10.index[l10["canonical_smiles"].isin(locked_mols)].tolist()
print(f"[locked test] rec={len(lock_idx)} mol={len(locked_mols)}")

l10_all_mols = set(l10["canonical_smiles"])
te_val_cache = {}
for t in TASK_CONFIG:
    te_val_cache[t] = l10.loc[lock_idx, f"{t}_value"].to_numpy(dtype=np.float32)

CONTROL = {"HOMA": 0.0684, "NICS_ZZ": -0.3745, "MBCO": 0.6926}

rows = []
for task in TASK_CONFIG:
    df = load_base_dataset(task)                       # full base, aligned to existing cache
    base_cache = build_and_cache(df, f"base_{task}")   # loads existing npz (e-block included)
    node0 = base_cache["node_mats"]; adj0 = base_cache["adj_mats"]
    ring0 = base_cache["ring_indices"]; vals0 = value_array(df, task)

    eflag = df["New_ID"].astype(str).str.lower().str.startswith("e").to_numpy()
    print(f"\n[{task}] base rows={len(df)}  e-block rows={int(eflag.sum())}")

    # keep = NOT e-block AND NOT overlapping any l10 molecule (same rule as exp4)
    keep = (~eflag) & (~df["canonical_smiles"].isin(l10_all_mols).to_numpy())
    node = node0[keep]; adj = adj0[keep]; ring = ring0[keep]; vals = vals0[keep]
    mol_list = df.loc[keep, "canonical_smiles"].tolist()
    if TASK_CONFIG[task]["rf_val"] == 10:
        node = derive_rf10(node)
    print(f"  train rows (base\\e\\l10-overlap) = {len(mol_list)}")

    # molecule-level train/val split — identical scheme/seed to exp4 (rseed=2026, +500)
    mol_arr = np.array(mol_list)
    um = np.unique(mol_arr)
    rg2 = np.random.RandomState(2026 + 500)
    pm2 = rg2.permutation(len(um))
    n_va = max(20, int(0.125 * len(um)))
    va_ids = set(um[pm2[:n_va]].tolist())
    tr_idx = np.where(np.isin(mol_arr, um[pm2[n_va:]]))[0].tolist()
    va_idx = np.where(np.isin(mol_arr, list(va_ids)))[0].tolist()
    print(f"  tr={len(tr_idx)} va={len(va_idx)}")

    model, bmae = train_model(node, adj, ring, vals, tr_idx, va_idx, task, 2026)

    te_node = l10_cache["node_mats"][lock_idx]
    te_adj = l10_cache["adj_mats"][lock_idx]
    te_ring = l10_cache["ring_indices"][lock_idx]
    if TASK_CONFIG[task]["rf_val"] == 10:
        te_node = derive_rf10(te_node)
    yp = predict_model(model, te_node, te_adj, te_ring, list(range(len(lock_idx))))
    m = metrics(te_val_cache[task], yp)
    row = {"task": task, "variant": "base_wo_e", "n_train": len(mol_list),
           "control_R2_quick": CONTROL[task], "R2": m["R2"], "MAE": m["MAE"], "RMSE": m["RMSE"],
           "delta_R2": round(m["R2"] - CONTROL[task], 4), "best_val_mae": bmae}
    rows.append(row)
    print(f"  -> R2={m['R2']:.4f} (control={CONTROL[task]:.4f}) MAE={m['MAE']:.4f} "
          f"delta={row['delta_R2']:+.4f}")

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "06_tables" / "coldstart_ablation_base_wo_e.csv", index=False)
print("\nSaved:", OUT_DIR / "06_tables/coldstart_ablation_base_wo_e.csv")
print(out.round(4).to_string(index=False))
