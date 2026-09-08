#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Spectator ring prediction.

Predict aromaticity (HOMA / NICS / nMCBO) for the 5,722 Tier=R spectator rings
using the frozen best_model_package.  Uses fix_ring_indices.py to ensure
canonical-model atom indices, then calls the existing fixed prediction pipeline.

Outputs:
  fig6_analysis_v2/PANEL_B_TARGET_SPECTATOR/spectator_ring_property_predictions.csv
  fig6_analysis_v2/PANEL_B_TARGET_SPECTATOR/spectator_ring_pair_aromaticity_predictions.csv
"""
from __future__ import annotations
import json, os, sys, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "best_model_package"))

os.environ["TQDM_DISABLE"] = "1"

ALLRINGS = ROOT / "dearom_ring_pairs_A"
OUT_B = ROOT / "fig6_analysis_v2" / "PANEL_B_TARGET_SPECTATOR"
OUT_B.mkdir(parents=True, exist_ok=True)

# ---------- 1. build samples CSV from ring_pairs_ml ----------
print("[1/4] building spectator ring ML samples from ring_pairs_ml.csv ...")
rp = pd.read_csv(ALLRINGS / "ring_pairs_ml.csv", low_memory=False)
spec = rp[rp["stage2_tier"] == "R"].copy()
print(f"  spectator pairs: {len(spec)}")

rows = []
for _, r in spec.iterrows():
    r_map_nums = str(r["target_ring_map_numbers"])
    try:
        target_map_set = set(int(x) for x in r_map_nums.replace(";", " ").split() if x.strip().isdigit())
    except Exception:
        target_map_set = set()
    target_indices = [int(x) for x in str(r["reactant_ring_atom_indices"]).replace(";", " ").split() if x.strip().isdigit()]
    target_indices_p = [int(x) for x in str(r["product_ring_atom_indices"]).replace(";", " ").split() if x.strip().isdigit()]
    # Reactant sample
    rows.append({
        "sample_id": f"{r['pair_id']}__R",
        "pair_id": r["pair_id"],
        "reaction_id": r["reaction_id"],
        "side": "reactant",
        "smiles": r["reactant_component_smiles"],
        "smiles_mapped": r["reactant_component_smiles_mapped"],
        "target_atom_indices": ";".join(map(str, target_indices)),
        "target_mask": 0,  # filled later by fix_ring_indices
        "target_ring_map_numbers": r_map_nums,
    })
    # Product sample
    rows.append({
        "sample_id": f"{r['pair_id']}__P",
        "pair_id": r["pair_id"],
        "reaction_id": r["reaction_id"],
        "side": "product",
        "smiles": r["product_component_smiles"],
        "smiles_mapped": r["product_component_smiles_mapped"],
        "target_atom_indices": ";".join(map(str, target_indices_p)),
        "target_mask": 0,
        "target_ring_map_numbers": r_map_nums,
    })
samples = pd.DataFrame(rows)
samples_path = OUT_B / "spectator_ring_property_ml_samples.csv"
samples.to_csv(samples_path, index=False)
print(f"  wrote {samples_path}")

# ---------- 2. fix indices ----------
print("[2/4] running fix_ring_indices.py ...")
import subprocess
fixed_path = OUT_B / "spectator_ring_property_ml_samples_fixed.csv"
res = subprocess.run(
    ["python", str(ROOT / "fix_ring_indices.py"),
     "--input", str(samples_path),
     "--output", str(fixed_path)],
    capture_output=True, text=True)
print(res.stdout[-500:])
if res.returncode != 0:
    print(res.stderr[-500:])
    raise SystemExit("fix_ring_indices failed")

# ---------- 3. predict ----------
print("[3/4] running predict_ring_properties_fixed.py ...")
pred_path = OUT_B / "spectator_ring_property_predictions.csv"
pairs_path = OUT_B / "spectator_ring_pair_aromaticity_predictions.csv"
res = subprocess.run(
    ["python", str(ROOT / "predict_ring_properties_fixed.py"),
     "--input", str(fixed_path),
     "--model-package", "/home/ubuntu/aroma-dps-code/best_model_package",
     "--output", str(pred_path),
     "--pairs-output", str(pairs_path)],
    capture_output=True, text=True)
print(res.stdout[-500:])
if res.returncode != 0:
    print(res.stderr[-500:])
    raise SystemExit("predict failed")

# ---------- 4. summarize ----------
pred = pd.read_csv(pred_path, low_memory=False)
pairs = pd.read_csv(pairs_path, low_memory=False)
n_pass = (pred["prediction_status"] == "PASS").sum()
n_err = (pred["prediction_status"] != "PASS").sum()
ok_pairs = pred[pred["prediction_status"] == "PASS"].groupby("pair_id").size()
n_pairs_both = int((ok_pairs >= 2).sum())
summary = {
    "n_spectator_pairs_input": int(len(spec)),
    "n_samples_total": int(len(pred)),
    "n_samples_pass": int(n_pass),
    "n_samples_error": int(n_err),
    "n_pairs_with_both_RP_pass": n_pairs_both,
}
(OUT_B / "spectator_prediction_summary.json").write_text(json.dumps(summary, indent=2))
print("[4/4] summary:", summary)
print("Done")
