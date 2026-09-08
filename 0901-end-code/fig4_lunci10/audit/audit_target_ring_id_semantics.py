"""Phase 6 prep audit (per user protocol): target_ring_id semantics check.

Question: is target_ring_id
  A) a cross-molecule comparable chemical identity (e.g. "the left ring of naphthalene",
     "the 5-ring of imidazole", "ring 0 of benzene-fused-X"), or
  B) only a local intra-molecule ring index (0/1/2) with no comparable meaning across molecules?

If B, then (ring_name, ring_pos, target_ring_id) is NOT a valid cross-molecule
chemical context key for pair construction.
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

PROJ_ROOT = Path(_PROJ_ROOT)
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"
OUT_JSON = AUDIT_OUT / "target_ring_id_semantics_audit.json"

RDKIT_VERSION = Chem.__dict__.get("__version__", "unknown")


def main() -> None:
    df = pd.read_csv(CLEAN_MANIFEST)
    print(f"[load] {len(df)} rows from clean manifest")

    # 1. Distribution of target_ring_id values
    print("\n[1] target_ring_id value distribution:")
    trid_dist = df["target_ring_id"].astype(str).value_counts().head(20)
    print(trid_dist.to_string())

    # 2. How often is target_ring_id simply "1,2,3,4,5,6" (an atom list) vs "1" (an index)?
    n_atom_list = sum(1 for x in df["target_ring_id"].astype(str) if "," in x)
    n_index = sum(1 for x in df["target_ring_id"].astype(str) if "," not in x and len(x) <= 4)
    print(f"\n[2] target_ring_id format:")
    print(f"  comma-separated (atom list): {n_atom_list}")
    print(f"  short index-like: {n_index}")

    # 3. Same ring_name + same target_ring_id across molecules?
    # If target_ring_id has comparable meaning, then for fixed ring_name + ring_pos,
    # the target_ring_id values should describe the SAME chemistry.
    # If it is just local indexing, identical strings in different molecules mean different things.
    cross = df.groupby(["ring_name", "ring_pos", "target_ring_id"]).size().reset_index(name="n")
    n_unique_cross = len(cross)
    print(f"\n[3] unique (ring_name, ring_pos, target_ring_id) combinations: {n_unique_cross}")

    # 4. Compare target_ring_atoms (which IS chemistry-defined) consistency across molecules
    #    that share the same (ring_name, ring_pos, target_ring_id)
    target_atoms_per_cross = df.groupby(["ring_name", "ring_pos", "target_ring_id"])["target_ring_atoms"].apply(
        lambda s: list(set(s.astype(str).tolist()))
    ).reset_index(name="unique_target_atoms_lists")
    multi_atom_lists = target_atoms_per_cross[
        target_atoms_per_cross["unique_target_atoms_lists"].apply(lambda L: len(L) > 1)
    ]
    n_cross_with_multi_target_atoms = len(multi_atom_lists)
    print(f"\n[4] cross-molecule combinations with multiple target_ring_atoms values:")
    print(f"  {n_cross_with_multi_target_atoms} of {n_unique_cross}")
    if n_cross_with_multi_target_atoms > 0:
        print("  -> SAME (ring_name, ring_pos, target_ring_id) corresponds to DIFFERENT target_ring_atoms in different molecules.")
        print("  -> target_ring_id is NOT a cross-molecule comparable identity (it's a local index).")
        decision = "B_local_intra_molecule_index"
    else:
        print("  -> SAME (ring_name, ring_pos, target_ring_id) always corresponds to SAME target_ring_atoms.")
        print("  -> target_ring_id has comparable chemistry across molecules.")
        decision = "A_cross_molecule_comparable"

    # 5. Sample inspection
    print(f"\n[5] sample 5 entries with target_ring_atoms and target_ring_id:")
    sample = df[["sample_id", "ring_name", "ring_pos", "target_ring_id", "target_ring_atoms", "raw_smiles"]].head(5)
    for _, r in sample.iterrows():
        print(f"  {r['sample_id']} ring={r['ring_name']} pos={r['ring_pos']} trid={r['target_ring_id']} atoms={r['target_ring_atoms']} smi={r['raw_smiles']}")

    out = {
        "rdkit_version": RDKIT_VERSION,
        "n_records": len(df),
        "target_ring_id_format_distribution": {
            "n_comma_separated_atom_list": n_atom_list,
            "n_short_index_like": n_index,
        },
        "n_unique_(ring_name, ring_pos, target_ring_id)": n_unique_cross,
        "n_cross_with_multiple_target_atoms": n_cross_with_multi_target_atoms,
        "decision": decision,
        "implication_for_phase_6_pair_construction": (
            "If decision=B: do NOT use (ring_name, ring_pos, target_ring_id) as the cross-molecule chemical context key. "
            "Redefine context_key using truly comparable chemistry: target-ring canonical subgraph, "
            "fused topology, attachment site, parent/core identity, ring size + heteroatom pattern."
            if decision.startswith("B") else
            "(ring_name, ring_pos, target_ring_id) is acceptable as cross-molecule context key."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDECISION: {decision}")
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()