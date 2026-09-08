#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Pair-level inventory master table.

Reconciles the three counting levels behind "4,306 reactions -> 4,443
target-ring pairs -> 4,377 complete R/P vectors":

  * 4,443 pairs  (ring_pairs_ml.csv, 4,306 unique reactions; 137 reactions
                  contribute 2 target rings each -> +137)
  * 4,423 pairs  predicted (ring_pair_aromaticity_predictions.csv)
  * 4,377 pairs  complete R/P (ring_pair_aromaticity_predictions_complete.csv)

Outputs:
  fig6_analysis_v2/DATA_AUDIT/fig6_pair_inventory_master.csv   (1 row per pair, 4,443 rows)
  fig6_analysis_v2/DATA_AUDIT/fig6_pair_inventory_summary.csv  (status/family cross-tabs)

The ring-family assignment reuses the SAME classifier as Fig.6d
(fig6_v2_main.classify_ring_family) so the column is directly comparable with
PANEL_D_RING_FAMILY/ring_family_assignment.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rdkit import Chem

from fig6_v2_main import classify_ring_family

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
TIERA = ROOT / "dearom_ring_pairs_A_tierA"
OUT = ROOT / "fig6_analysis_v2" / "DATA_AUDIT"
OUT.mkdir(parents=True, exist_ok=True)


def family_of_row(row):
    """Re-run the Fig.6d family classifier on a ring_pairs_ml row."""
    smi = row.get("reactant_component_smiles_mapped")
    if not isinstance(smi, str) or not smi:
        return ("unknown", "", "", False, "", "parse_failed")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ("unknown", "", "", False, "", "parse_failed")
    try:
        maps = set(int(x) for x in str(row["target_ring_map_numbers"]).replace(";", " ").split()
                   if x.strip().isdigit())
        idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() in maps]
    except Exception:
        idx = []
    return classify_ring_family(mol, idx)


def main():
    rp = pd.read_csv(TIERA / "ring_pairs_ml.csv", low_memory=False)
    pred = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions.csv", low_memory=False)
    comp = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions_complete.csv", low_memory=False)

    # previously assigned families (complete set) — reuse verbatim
    fa = pd.read_csv(ROOT / "fig6_analysis_v2" / "PANEL_D_RING_FAMILY" / "ring_family_assignment.csv",
                     low_memory=False)
    fam_map = fa.set_index("pair_id")["ring_family"].to_dict()

    pred_pids = set(pred["pair_id"])
    comp_pids = set(comp["pair_id"])
    pred_pid_set = pred[pred["HOMA_pred_reactant"].notna()].set_index("pair_id")["HOMA_pred_reactant"].index
    r_ok = set(pred[pred["HOMA_pred_reactant"].notna()]["pair_id"])
    p_ok = set(pred[pred["HOMA_pred_product"].notna()]["pair_id"])

    rxn_npairs = rp["reaction_id"].value_counts()

    rows = []
    for _, r in rp.iterrows():
        pid = r["pair_id"]
        if pid in comp_pids:
            status = "complete"
            failed_side = "none"
        elif pid in pred_pids:
            status = "one_sided"
            if pid not in r_ok:
                failed_side = "reactant"
            elif pid not in p_ok:
                failed_side = "product"
            else:
                failed_side = "unknown"
        else:
            status = "absent_from_predictions"
            failed_side = "both"

        if pid in fam_map:
            fam = fam_map[pid]
            (rsize, hp, fused, nb, rule) = ("", "", "", "", "")
            fa_row = fa[fa["pair_id"] == pid]
            if len(fa_row):
                x = fa_row.iloc[0]
                rsize, hp, fused, nb, rule = (x["ring_size"], x["heteroatom_pattern"],
                                              x["is_fused"], x["neighbor_ring_types"],
                                              x["classification_rule"])
        else:
            fam, conf, hp, fused, nb, rule = family_of_row(r)
            rsize = r.get("target_ring_size", "")

        rows.append({
            "pair_id": pid,
            "reaction_id": r["reaction_id"],
            "status": status,
            "failed_side": failed_side,
            "n_target_pairs_in_reaction": int(rxn_npairs.get(r["reaction_id"], 1)),
            "is_multi_target_reaction": bool(rxn_npairs.get(r["reaction_id"], 1) > 1),
            "ring_family": fam,
            "ring_size": rsize,
            "heteroatom_pattern": hp,
            "is_fused": fused,
            "neighbor_ring_types": nb,
            "classification_rule": rule,
        })

    master = pd.DataFrame(rows)
    master.to_csv(OUT / "fig6_pair_inventory_master.csv", index=False)

    # ---- summary ----
    out = {}
    out["n_pairs_total"] = len(master)
    out["n_unique_reactions"] = master["reaction_id"].nunique()
    out["n_reactions_with_2_target_rings"] = int(
        master[master["is_multi_target_reaction"]]["reaction_id"].nunique())
    out["n_pairs_complete"] = int((master["status"] == "complete").sum())
    out["n_pairs_one_sided"] = int((master["status"] == "one_sided").sum())
    out["n_pairs_absent"] = int((master["status"] == "absent_from_predictions").sum())

    fs = master[master["status"] == "one_sided"]["failed_side"].value_counts()
    out["one_sided_failed_side_product"] = int(fs.get("product", 0))
    out["one_sided_failed_side_reactant"] = int(fs.get("reactant", 0))

    # reactions that lost everything
    kept_rxns = set(master[master["status"] == "complete"]["reaction_id"])
    out["n_reactions_lost_entirely"] = len(set(master["reaction_id"]) - kept_rxns)

    # family x status cross-tab (complete)
    ct = master[master["status"] == "complete"]["ring_family"].value_counts()

    # build a tidy summary csv: section,key,value
    sum_rows = [{"section": "counts", "key": k, "value": v} for k, v in out.items()]
    for fam, n in ct.items():
        sum_rows.append({"section": "complete_by_family", "key": fam, "value": int(n)})
    # one-sided / absent families
    for status in ("one_sided", "absent_from_predictions"):
        sub = master[master["status"] == status]["ring_family"].value_counts()
        for fam, n in sub.items():
            sum_rows.append({"section": f"{status}_by_family", "key": fam, "value": int(n)})

    pd.DataFrame(sum_rows).to_csv(OUT / "fig6_pair_inventory_summary.csv", index=False)

    print("master rows:", len(master))
    print(master["status"].value_counts().to_string())
    print("one_sided failed_side:", master[master["status"] == "one_sided"]["failed_side"].value_counts().to_dict())
    print("multi-target reactions:", out["n_reactions_with_2_target_rings"])
    print("reactions lost entirely:", out["n_reactions_lost_entirely"])
    print("\ncomplete by family:")
    print(ct.to_string())


if __name__ == "__main__":
    main()
