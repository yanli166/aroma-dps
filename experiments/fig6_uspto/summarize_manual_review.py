#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize manual Stage-2 review labels.

Expected input: manual_review_sample.csv created by dearom_stage2_exact.py.
Fill:
  human_label = TRUE / FALSE / UNCERTAIN
Optionally add human_note.

Outputs per-tier precision estimates and a confusion-style summary.
"""
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="manual_precision_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "human_label" not in df.columns:
        raise ValueError("Missing human_label column")

    df["human_label"] = df["human_label"].astype(str).str.strip().str.upper()
    valid = df[df["human_label"].isin(["TRUE", "FALSE", "UNCERTAIN"])].copy()

    rows = []
    for tier, g in valid.groupby("stage2_tier"):
        n = len(g)
        nt = int((g["human_label"] == "TRUE").sum())
        nf = int((g["human_label"] == "FALSE").sum())
        nu = int((g["human_label"] == "UNCERTAIN").sum())
        denom = nt + nf
        precision_excl_uncertain = nt / denom if denom else float("nan")
        rows.append({
            "stage2_tier": tier,
            "n_reviewed": n,
            "true": nt,
            "false": nf,
            "uncertain": nu,
            "precision_excluding_uncertain": precision_excl_uncertain,
        })

    out = pd.DataFrame(rows).sort_values("stage2_tier")
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
