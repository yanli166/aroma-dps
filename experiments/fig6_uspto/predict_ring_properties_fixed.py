#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch aromaticity prediction using the corrected model atom indices.

Requires the output of fix_ring_indices.py.

Predictions are written at the molecule-ring sample level and optionally aggregated
into one row per reactant/product ring pair.

Expected model package:
  /home/ubuntu/aroma-dps-code/best_model_package/
with:
  from predict import AromaticityPredictor
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path
from typing import List, Any

import pandas as pd


def parse_indices(s: Any) -> List[int]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return [int(x) for x in str(s).split(";") if x.strip()]


def normalize_prediction(res):
    """
    Handles either dict-like predictor outputs or simple sequences.
    Edit keys here if your local predictor uses slightly different names.
    """
    if isinstance(res, dict):
        lower = {str(k).lower(): v for k, v in res.items()}

        def get(*names):
            for n in names:
                if n.lower() in lower:
                    return lower[n.lower()]
            return None

        return {
            "HOMA_pred": get("HOMA", "homa"),
            "NICS_1zz_pred": get("NICS_1zz", "NICS1zz", "NICS", "nics_1zz"),
            "nMCBO_pred": get("nMCBO", "MCBO", "MBCO", "nmcbO", "mcbo"),
        }

    if isinstance(res, (list, tuple)) and len(res) >= 3:
        return {
            "HOMA_pred": res[0],
            "NICS_1zz_pred": res[1],
            "nMCBO_pred": res[2],
        }

    raise ValueError(f"Unrecognized prediction output: {res!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-package", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pairs-output", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.model_package)
    from predict import AromaticityPredictor

    df = pd.read_csv(args.input, low_memory=False)
    if args.limit:
        df = df.head(args.limit).copy()

    pass_mask = df["index_status"].astype(str).str.startswith("PASS")
    if not pass_mask.all():
        print(
            f"WARNING: excluding {(~pass_mask).sum()} samples that failed index validation."
        )
        df = df[pass_mask].copy()

    predictor = AromaticityPredictor(args.model_package)

    records = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        smi = row["smiles_model"]
        idx = parse_indices(row["target_atom_indices_model"])

        rec = row.to_dict()
        try:
            raw = predictor.predict(smi, idx)
            rec.update(normalize_prediction(raw))
            rec["prediction_status"] = "PASS"
            rec["prediction_error"] = ""
        except Exception as exc:
            rec.update({
                "HOMA_pred": None,
                "NICS_1zz_pred": None,
                "nMCBO_pred": None,
                "prediction_status": "ERROR",
                "prediction_error": f"{type(exc).__name__}:{exc}",
            })

        records.append(rec)

        if i % args.checkpoint_every == 0:
            pd.DataFrame(records).to_csv(args.output, index=False)
            print(f"predicted {i:,}/{len(df):,}", flush=True)

    pred = pd.DataFrame(records)
    pred.to_csv(args.output, index=False)

    # One row per pair_id with reactant/product and reaction-level deltas.
    if args.pairs_output and "pair_id" in pred.columns and "side" in pred.columns:
        good = pred[pred["prediction_status"] == "PASS"].copy()

        cols = ["pair_id", "side", "HOMA_pred", "NICS_1zz_pred", "nMCBO_pred"]
        wide = good[cols].pivot_table(
            index="pair_id",
            columns="side",
            values=["HOMA_pred", "NICS_1zz_pred", "nMCBO_pred"],
            aggfunc="first"
        )

        wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
        wide = wide.reset_index()

        # Expected side labels from ring_property_ml_samples.csv are reactant/product.
        if {
            "HOMA_pred_reactant", "HOMA_pred_product"
        }.issubset(wide.columns):
            wide["Delta_HOMA"] = (
                wide["HOMA_pred_reactant"] - wide["HOMA_pred_product"]
            )

        if {
            "nMCBO_pred_reactant", "nMCBO_pred_product"
        }.issubset(wide.columns):
            wide["Delta_nMCBO"] = (
                wide["nMCBO_pred_reactant"] - wide["nMCBO_pred_product"]
            )

        if {
            "NICS_1zz_pred_reactant", "NICS_1zz_pred_product"
        }.issubset(wide.columns):
            # Positive = aromaticity loss:
            # (-NICS_R) - (-NICS_P) = NICS_P - NICS_R
            wide["Delta_NICS_star"] = (
                wide["NICS_1zz_pred_product"] - wide["NICS_1zz_pred_reactant"]
            )

        wide.to_csv(args.pairs_output, index=False)

    report = {
        "n_input_validated_samples": int(len(df)),
        "n_prediction_pass": int((pred["prediction_status"] == "PASS").sum()),
        "n_prediction_error": int((pred["prediction_status"] != "PASS").sum()),
        "IMPORTANT": (
            "Predictions use smiles_model + target_atom_indices_model, "
            "never the original target_atom_indices."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
