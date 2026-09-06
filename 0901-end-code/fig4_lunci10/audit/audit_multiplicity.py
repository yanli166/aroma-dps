"""Task 9 (按协议修正): multiplicity / context_key audit.

要求:
  - 解释为什么 1389 molecules -> 2153 ring-level rows
  - 解释 1026 unique ring×sub vs 1048 unique ring×pos×sub vs 2153 records
  - 检查 target ring identity / parent/core / attachment site / regioisomer / fused / multiple target rings
  - 给出经过审计的 context_key (不能只按 ring_name + ring_pos 就开始构建 pairs)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rdkit import Chem, RDLogger


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

RDLogger.DisableLog("rdApp.*")

PROJ_ROOT = Path("_PROJ_ROOT")
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"

OUT_JSON = AUDIT_OUT / "multiplicity_context_audit.json"
OUT_CSV = AUDIT_OUT / "multiplicity_breakdown.csv"


def main() -> None:
    df = pd.read_csv(CLEAN_MANIFEST)
    n_total = len(df)
    n_unique_smiles = df["canonical_smiles"].nunique()
    n_unique_ring = df["ring_name"].nunique()
    n_unique_sub = df["sub_name"].nunique()
    n_unique_ringpos = df["ring_pos"].nunique()
    n_unique_subtype = df["sub_type"].nunique()

    # various combinations
    ring_sub = df.groupby(["ring_name", "sub_name"]).size().reset_index(name="n")
    ring_pos_sub = df.groupby(["ring_name", "ring_pos", "sub_name"]).size().reset_index(name="n")
    ring_pos_sub_tgt = df.groupby(["ring_name", "ring_pos", "sub_name", "target_ring_id"]).size().reset_index(name="n")
    full = df.groupby(["ring_name", "ring_pos", "sub_name", "target_ring_id", "fused"]).size().reset_index(name="n")

    # 多重性来源分解: 一个 (ring_name, ring_pos, sub_name) 组合下的 record 数分布
    rs_records_per_comb = ring_pos_sub.groupby(["ring_name", "ring_pos", "sub_name"]).size().reset_index(name="n_records")
    rs_dist = rs_records_per_comb["n_records"].value_counts().sort_index().to_dict()

    # 平均每个分子有多少 ring-level records
    mol_rec = df.groupby("canonical_smiles").size().reset_index(name="n_records")
    mol_rec_dist = mol_rec["n_records"].value_counts().sort_index().to_dict()

    # 一个分子有多个 target ring 的情况
    n_mol_multi_ring = int((mol_rec["n_records"] > 1).sum())
    n_mol_single_ring = int((mol_rec["n_records"] == 1).sum())

    # 同 (ring, pos, sub) 但 target_ring_id 不同 -> 多个 target ring 的 fused 分子
    multi_target_in_same_ring = full.groupby(["ring_name", "ring_pos", "sub_name"]).size()
    n_have_multiple_target_rings = int((multi_target_in_same_ring > 1).sum())

    # HOMA/MBCO/NICS 等任务有 fused context, 多个环都标记
    # 检查 fused 列分布
    fused_dist = df["fused"].value_counts().to_dict() if "fused" in df.columns else {}

    # ring_size 分布
    if "ring_size" in df.columns:
        ring_size_dist = df["ring_size"].value_counts().to_dict()

    out = {
        "totals": {
            "n_total_records": n_total,
            "n_unique_molecules": n_unique_smiles,
            "n_unique_ring_name": n_unique_ring,
            "n_unique_sub_name": n_unique_sub,
            "n_unique_ring_pos": n_unique_ringpos,
            "n_unique_sub_type": n_unique_subtype,
            "n_unique_ring_x_sub": len(ring_sub),
            "n_unique_ring_x_pos_x_sub": len(ring_pos_sub),
            "n_unique_ring_x_pos_x_sub_x_target_ring": len(ring_pos_sub_tgt),
            "n_unique_ring_x_pos_x_sub_x_target_x_fused": len(full),
        },
        "multiplicity_origin": {
            "ring_pos__sub_records_per_combination": rs_dist,
            "molecule_records_per_molecule": mol_rec_dist,
            "n_molecules_with_multiple_records": n_mol_multi_ring,
            "n_molecules_with_single_record": n_mol_single_ring,
            "n_(ring,pos,sub)_combinations_with_multiple_target_rings": n_have_multiple_target_rings,
            "fused_distribution": fused_dist,
            "ring_size_distribution": ring_size_dist if "ring_size" in df.columns else {},
        },
        "context_key_recommendation": {
            "minimal_context": ["ring_name", "ring_pos", "target_ring_id"],
            "extended_context_optional": ["ring_name", "ring_pos", "target_ring_id", "fused", "ring_size", "sub_type"],
            "rationale": (
                "Pure (ring_name, ring_pos) is insufficient: target ring identity (target_ring_id) "
                "distinguishes cases where the same scaffold+position has multiple aromatic rings "
                "(e.g. fused bicyclic, naphthalene has 2 target rings per molecule). "
                "Sub_type (C-linked vs N-linked) further disambiguates electron-donor attachment. "
                "Pair construction must use at least (ring_name, ring_pos, target_ring_id) as the "
                "grouping unit; mismatched target_ring_id between two molecules means the delta "
                "compares aromaticity of chemically different rings."
            ),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["totals"], indent=2))
    print()
    print("multiplicity origin:")
    print(json.dumps(out["multiplicity_origin"], indent=2))
    print()
    print("context_key recommendation:")
    print(json.dumps(out["context_key_recommendation"], indent=2))
    print(f"  wrote {OUT_JSON}")

    # 输出按 (ring, pos, sub) 的 multiplicity 分布
    rs_records_per_comb.sort_values("n_records", ascending=False).to_csv(
        AUDIT_OUT / "records_per_ring_pos_sub.csv", index=False
    )
    print(f"  wrote records_per_ring_pos_sub.csv (top combos)")


if __name__ == "__main__":
    main()