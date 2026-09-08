#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix target-ring atom indices after RDKit canonical SMILES reordering.

Problem
-------
`target_atom_indices` was obtained from parsing `smiles_mapped`, while the ML model
is fed the separately canonicalized plain `smiles`. RDKit canonicalization can reorder
atoms, so those indices are not generally interchangeable.

Correct solution
----------------
Use `smiles_mapped` as the authoritative molecule:
1. Identify target atoms by atom-map numbers.
2. Make a copy and remove atom-map numbers WITHOUT changing atom order.
3. Canonicalize that copy with RDKit.
4. Read RDKit's `_smilesAtomOutputOrder`, which records which original atom index
   became each atom position in the emitted canonical SMILES.
5. Convert original mapped-molecule atom indices -> canonical plain-SMILES indices.
6. Parse the emitted plain SMILES again and validate the target atoms form the
   expected retained cycle.

Output
------
Adds:
  smiles_model
  target_atom_indices_original
  target_atom_indices_model
  target_ring_mask_model
  index_changed
  existing_smiles_matches_model
  target_cycle_valid_model
  target_cycle_edge_count_model
  index_status

The model should ONLY consume:
  smiles_model
  target_atom_indices_model
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

import pandas as pd
from rdkit import Chem


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip().strip("[](){}")
    if not s:
        return []
    s = s.replace(",", ";").replace(" ", ";")
    return [int(x) for x in s.split(";") if x.strip() and x.strip().lstrip("-").isdigit()]


def parse_target_maps(value: Any) -> Set[int]:
    return {x for x in parse_int_list(value) if x > 0}


def canonicalize_preserving_index_map(mapped_smiles: str, target_maps: Set[int]):
    mol = Chem.MolFromSmiles(str(mapped_smiles))
    if mol is None:
        raise ValueError("MAPPED_SMILES_PARSE_FAILED")

    mapnum_to_oldidx = {}
    for atom in mol.GetAtoms():
        mp = atom.GetAtomMapNum()
        if mp > 0:
            mapnum_to_oldidx[mp] = atom.GetIdx()

    missing = sorted(target_maps - set(mapnum_to_oldidx))
    if missing:
        raise ValueError(f"TARGET_MAPS_MISSING_IN_MAPPED_SMILES:{missing}")

    old_target_idx = sorted(mapnum_to_oldidx[m] for m in target_maps)

    # IMPORTANT: copy molecule first; preserve current atom indices.
    nomap = Chem.Mol(mol)
    for atom in nomap.GetAtoms():
        atom.SetAtomMapNum(0)

    # RDKit writes this property during MolToSmiles:
    # output_order[new_index] = old_index
    plain = Chem.MolToSmiles(
        nomap,
        canonical=True,
        isomericSmiles=True
    )

    if not nomap.HasProp("_smilesAtomOutputOrder"):
        raise RuntimeError("RDKIT_SMILES_OUTPUT_ORDER_NOT_AVAILABLE")

    output_order = ast.literal_eval(nomap.GetProp("_smilesAtomOutputOrder"))
    if len(output_order) != nomap.GetNumAtoms():
        raise RuntimeError("INVALID_SMILES_OUTPUT_ORDER_LENGTH")

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(output_order)}
    new_target_idx = sorted(old_to_new[i] for i in old_target_idx)

    # Reparse exactly what will be passed to the model.
    model_mol = Chem.MolFromSmiles(plain)
    if model_mol is None:
        raise RuntimeError("CANONICAL_PLAIN_SMILES_REPARSE_FAILED")
    if model_mol.GetNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError(
            f"ATOM_COUNT_CHANGED:{mol.GetNumAtoms()}->{model_mol.GetNumAtoms()}"
        )

    return mol, model_mol, plain, old_target_idx, new_target_idx, output_order


def ring_induced_graph_check(mol: Chem.Mol, indices: List[int]) -> Tuple[bool, int, List[int]]:
    """
    For a simple n-member target cycle:
      induced ring edge count should be n
      every target atom should have 2 target neighbors

    Stage-2 A has same-cycle edge retention = 1, so this should normally pass.
    """
    target = set(indices)
    edge_set = set()
    degree_inside = {i: 0 for i in indices}

    for i in indices:
        atom = mol.GetAtomWithIdx(i)
        for nb in atom.GetNeighbors():
            j = nb.GetIdx()
            if j in target:
                edge_set.add(tuple(sorted((i, j))))

    for i, j in edge_set:
        degree_inside[i] += 1
        degree_inside[j] += 1

    degrees = [degree_inside[i] for i in indices]
    valid = len(edge_set) == len(indices) and all(d == 2 for d in degrees)
    return valid, len(edge_set), degrees


def make_mask(n_atoms: int, indices: List[int]) -> str:
    target = set(indices)
    return "".join("1" if i in target else "0" for i in range(n_atoms))


def canonical_equivalent(smi1: str, smi2: str) -> bool:
    try:
        m1 = Chem.MolFromSmiles(str(smi1))
        m2 = Chem.MolFromSmiles(str(smi2))
        if m1 is None or m2 is None:
            return False
        c1 = Chem.MolToSmiles(m1, canonical=True, isomericSmiles=True)
        c2 = Chem.MolToSmiles(m2, canonical=True, isomericSmiles=True)
        return c1 == c2
    except Exception:
        return False


def process_row(row):
    result = {
        "smiles_model": "",
        "target_atom_indices_original": str(row.get("target_atom_indices", "")),
        "target_atom_indices_model": "",
        "target_ring_mask_model": "",
        "index_changed": False,
        "existing_smiles_matches_model": False,
        "target_cycle_valid_model": False,
        "target_cycle_edge_count_model": 0,
        "target_cycle_internal_degrees_model": "",
        "index_status": "ERROR",
        "index_error": "",
    }

    try:
        target_maps = parse_target_maps(row["target_ring_map_numbers"])
        if len(target_maps) < 3:
            raise ValueError("INVALID_TARGET_RING_MAP_NUMBERS")

        (
            mapped_mol,
            model_mol,
            plain,
            old_idx,
            new_idx,
            output_order,
        ) = canonicalize_preserving_index_map(
            row["smiles_mapped"],
            target_maps
        )

        cycle_valid, edge_count, degrees = ring_induced_graph_check(model_mol, new_idx)

        old_csv_idx = sorted(parse_int_list(row.get("target_atom_indices", "")))

        result.update({
            "smiles_model": plain,
            "target_atom_indices_original": ";".join(map(str, old_csv_idx)),
            "target_atom_indices_model": ";".join(map(str, new_idx)),
            "target_ring_mask_model": make_mask(model_mol.GetNumAtoms(), new_idx),
            "index_changed": old_csv_idx != new_idx,
            "existing_smiles_matches_model": canonical_equivalent(
                row.get("smiles", ""), plain
            ),
            "target_cycle_valid_model": cycle_valid,
            "target_cycle_edge_count_model": edge_count,
            "target_cycle_internal_degrees_model": ";".join(map(str, degrees)),
        })

        if not cycle_valid:
            result["index_status"] = "FAIL_TARGET_NOT_SIMPLE_CYCLE"
        elif len(new_idx) != len(target_maps):
            result["index_status"] = "FAIL_TARGET_SIZE_MISMATCH"
        elif not result["existing_smiles_matches_model"]:
            # Structure mismatch is more serious than an atom-index reorder.
            result["index_status"] = "FAIL_EXISTING_SMILES_STRUCTURE_MISMATCH"
        elif result["index_changed"]:
            result["index_status"] = "PASS_REINDEXED"
        else:
            result["index_status"] = "PASS_UNCHANGED"

    except Exception as exc:
        result["index_error"] = f"{type(exc).__name__}:{exc}"

    return pd.Series(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--errors", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    required = {
        "smiles",
        "smiles_mapped",
        "target_atom_indices",
        "target_ring_map_numbers",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fixed = df.apply(process_row, axis=1)
    out = pd.concat([df, fixed], axis=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    fail_mask = ~out["index_status"].astype(str).str.startswith("PASS")
    errors_path = Path(args.errors) if args.errors else out_path.with_name(
        out_path.stem + "_index_errors.csv"
    )
    out[fail_mask].to_csv(errors_path, index=False)

    status_counts = {
        str(k): int(v)
        for k, v in out["index_status"].value_counts(dropna=False).items()
    }

    report = {
        "n_samples": int(len(out)),
        "n_pass": int((~fail_mask).sum()),
        "n_fail": int(fail_mask.sum()),
        "n_index_changed": int(out["index_changed"].fillna(False).sum()),
        "fraction_index_changed": float(out["index_changed"].fillna(False).mean()),
        "n_existing_smiles_structure_match": int(
            out["existing_smiles_matches_model"].fillna(False).sum()
        ),
        "n_target_cycle_valid": int(
            out["target_cycle_valid_model"].fillna(False).sum()
        ),
        "status_counts": status_counts,
        "model_input_columns": {
            "smiles": "smiles_model",
            "ring_indices": "target_atom_indices_model"
        }
    }

    report_path = Path(args.report) if args.report else out_path.with_name(
        out_path.stem + "_index_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFixed dataset: {out_path}")
    print(f"Errors:        {errors_path}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
