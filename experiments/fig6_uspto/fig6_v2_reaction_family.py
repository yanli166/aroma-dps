#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Reaction-family (topology) classification.

Conservative rule-based classification using atom-mapped target-ring reaction-center topology.
NO mechanism claims.

Output under fig6_analysis_v2/REACTION_FAMILY_OPTIONAL/
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
TIERA = ROOT / "dearom_ring_pairs_A_tierA"
OUT = ROOT / "fig6_analysis_v2" / "REACTION_FAMILY_OPTIONAL"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def classify_reaction_family(row):
    """
    Conservative classification using atom-mapped target-ring reaction-center topology:
      - reduction-like        : no new heavy-atom bond added to target ring; bond order decreases
      - single-addition-like  : exactly 1 new heavy-atom bond to target ring
      - multi-addition-like   : >= 2 new heavy-atom bonds (cycloadditions, double additions)
      - rearrangement-like    : same ring atom count but bonds reshuffled
      - complex/other         : anything else (fragmentation, ring opening, etc.)
    """
    rmol = Chem.MolFromSmiles(row["reactant_component_smiles_mapped"])
    pmol = Chem.MolFromSmiles(row["product_component_smiles_mapped"])
    if rmol is None or pmol is None:
        return "complex_or_other", "parse_failed", 0, 0
    try:
        maps = set(int(x) for x in str(row["target_ring_map_numbers"]).replace(";", " ").split() if x.strip().isdigit())
        r_target = {a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum() in maps}
        p_target = {a.GetIdx() for a in pmol.GetAtoms() if a.GetAtomMapNum() in maps}
    except Exception:
        return "complex_or_other", "parse_failed", 0, 0
    if not r_target or not p_target:
        return "complex_or_other", "empty_target", 0, 0
    # Bonds in target ring (both sides)
    r_target_bonds = set()
    for b in rmol.GetBonds():
        if b.GetBeginAtomIdx() in r_target and b.GetEndAtomIdx() in r_target:
            r_target_bonds.add((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                b.GetBondType()))
    p_target_bonds = set()
    for b in pmol.GetBonds():
        if b.GetBeginAtomIdx() in p_target and b.GetEndAtomIdx() in p_target:
            p_target_bonds.add((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                b.GetBondType()))

    # bonds involving target atoms and external atoms
    r_external_bonds = 0
    for b in rmol.GetBonds():
        i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i1 in r_target) != (i2 in r_target):
            r_external_bonds += 1
    p_external_bonds = 0
    for b in pmol.GetBonds():
        i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i1 in p_target) != (i2 in p_target):
            p_external_bonds += 1

    n_added_heavy_bonds = max(0, p_external_bonds - r_external_bonds)

    ring_size_r = len(r_target); ring_size_p = len(p_target)

    # Aromaticity loss count (rough)
    r_arom = sum(1 for i in r_target if rmol.GetAtomWithIdx(i).GetIsAromatic())
    p_arom = sum(1 for i in p_target if pmol.GetAtomWithIdx(i).GetIsAromatic())

    # Heuristic bond-order reduction (sum of bond orders in target)
    bond_order = {
        Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5,
    }
    r_bo = sum(bond_order.get(t, 1.0) for _, _, t in r_target_bonds)
    p_bo = sum(bond_order.get(t, 1.0) for _, _, t in p_target_bonds)

    # Apply rules (priority order)
    rule_log = []
    if ring_size_p != ring_size_r:
        # ring opens/closes
        if n_added_heavy_bonds >= 2:
            fam = "multi_addition_like"
            rule = f"ring_size_change:{ring_size_r}->{ring_size_p}, n_added_heavy_bonds={n_added_heavy_bonds}"
        else:
            fam = "complex_or_other"
            rule = f"ring_size_change:{ring_size_r}->{ring_size_p}"
    elif n_added_heavy_bonds >= 2:
        fam = "multi_addition_like"
        rule = f"ring_size_unchanged, n_added_heavy_bonds={n_added_heavy_bonds}"
    elif n_added_heavy_bonds == 1:
        fam = "single_addition_like"
        rule = f"n_added_heavy_bonds=1, external_bonds {r_external_bonds}->{p_external_bonds}"
    elif n_added_heavy_bonds == 0:
        # no new heavy-atom bond; if bond order dropped -> reduction-like
        if p_bo < r_bo - 0.5:
            fam = "reduction_like"
            rule = f"bond_order_drop {r_bo:.1f}->{p_bo:.1f}, n_added_heavy_bonds=0"
        elif abs(p_bo - r_bo) < 0.5 and len(p_target_bonds) != len(r_target_bonds):
            fam = "rearrangement_like"
            rule = f"same_BO, bond_count {len(r_target_bonds)}->{len(p_target_bonds)}"
        else:
            fam = "complex_or_other"
            rule = f"no_new_heavy_bond, BO change {r_bo:.1f}->{p_bo:.1f} (ambiguous)"
    else:
        fam = "complex_or_other"
        rule = f"n_added_heavy_bonds={n_added_heavy_bonds} (rare negative)"

    # confidence: high when evidence unambiguous
    if fam in ("single_addition_like", "multi_addition_like"):
        conf = "high"
    elif fam == "reduction_like" and r_bo > p_bo + 1.0:
        conf = "high"
    elif fam == "reduction_like":
        conf = "medium"
    elif fam == "rearrangement_like":
        conf = "medium"
    else:
        conf = "low"
    return fam, rule, n_added_heavy_bonds, ring_size_p - ring_size_r


def main():
    complete = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions_complete.csv", low_memory=False)
    rp = pd.read_csv(TIERA / "ring_pairs_ml.csv", low_memory=False)
    full = complete.merge(rp[["pair_id", "reaction_id", "target_ring_map_numbers",
                              "reactant_component_smiles_mapped",
                              "product_component_smiles_mapped"]], on="pair_id", how="left")
    print(f"[rfam] running classification on {len(full)} target pairs ...")
    rows = []
    for _, r in full.iterrows():
        fam, rule, n_added, dsz = classify_reaction_family(r)
        rows.append({"pair_id": r["pair_id"], "reaction_id": r["reaction_id"],
                     "reaction_family": fam, "n_added_heavy_bonds": n_added,
                     "ring_size_change": dsz,
                     "classification_rule": rule})
    rfam = pd.DataFrame(rows)
    rfam.to_csv(OUT / "reaction_family_assignment.csv", index=False)
    cnt = rfam["reaction_family"].value_counts().reset_index()
    cnt.columns = ["reaction_family", "count"]
    cnt.to_csv(OUT / "reaction_family_counts.csv", index=False)
    print(rfam["reaction_family"].value_counts())

    # audit sample 50 per family
    audit_dir = OUT / "reaction_family_contact_sheets"
    audit_dir.mkdir(exist_ok=True)
    full2 = rfam.merge(rp[["pair_id", "reactant_component_smiles",
                           "product_component_smiles"]], on="pair_id", how="left")
    audit_lines = []
    for fam in cnt["reaction_family"]:
        sub = full2[full2["reaction_family"] == fam].sample(
            min(len(full2[full2["reaction_family"] == fam]), 50),
            random_state=RANDOM_STATE)
        mols = [Chem.MolFromSmiles(s) for s in sub["reactant_component_smiles"]]
        mols = [m for m in mols if m is not None]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(220, 160),
                                       legends=[fam] * len(mols))
            img.save(audit_dir / f"audit_{fam}.png")
        audit_lines.append(f"- {fam}: total={len(full2[full2['reaction_family']==fam])}, "
                           f"audited={min(50, len(sub))} structures saved to audit_{fam}.png")

    (OUT / "reaction_family_audit_report.md").write_text(
        "# Reaction family audit\n\n"
        + "Conservative topology-based classification, NOT mechanism claims.\n\n"
        + "\n".join(audit_lines) + "\n"
    )
    print("[rfam] done")


if __name__ == "__main__":
    main()
