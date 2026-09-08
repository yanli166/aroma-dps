#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch atom-map USPTO reactions with RXNMapper.

Install (if RDKit is already installed):
    pip install rxnmapper

Usage:
    python map_uspto_rxnmapper.py --input USPTO_50K.tsv --output USPTO_50K_mapped.csv --batch-size 64

Output columns include:
    id, class, reactions, mapped_reactions, mapping_confidence, mapping_status
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    need = {"id", "class", "reactions"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = load(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    mapper = RXNMapper()
    mapped, confs, status = [], [], []
    rxns = df["reactions"].astype(str).tolist()

    for start in tqdm(range(0, len(rxns), args.batch_size), desc="Atom mapping"):
        batch = rxns[start:start + args.batch_size]
        try:
            results = mapper.get_attention_guided_atom_maps(batch)
            for r in results:
                mapped.append(r.get("mapped_rxn", ""))
                confs.append(float(r.get("confidence", 0.0)))
                status.append("ok")
        except Exception:
            # Fall back to one-by-one so one malformed reaction does not kill a batch.
            for rxn in batch:
                try:
                    r = mapper.get_attention_guided_atom_maps([rxn])[0]
                    mapped.append(r.get("mapped_rxn", ""))
                    confs.append(float(r.get("confidence", 0.0)))
                    status.append("ok")
                except Exception as e:
                    mapped.append("")
                    confs.append(0.0)
                    status.append(f"failed:{type(e).__name__}")

    out = df.copy()
    out["mapped_reactions"] = mapped
    out["mapping_confidence"] = confs
    out["mapping_status"] = status
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out["mapping_status"].value_counts(dropna=False))
    print(out["mapping_confidence"].describe())


if __name__ == "__main__":
    main()
