#!/usr/bin/env python3
"""
P0-2: atom_on_ring Index-Base Audit

在 HOMA/NICS/MCBO 三个原始数据集上分别验证 raw index 作为 0-based 与 1-based
两种解释，检查是否对应真实 RDKit ring、ring size、元素组成及 index validity。
"""
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = Path("/home/ubuntu/aroma-dps-code/code_end/data1_end")
DATASETS = {
    "HOMA": DATA_DIR / "collet_homa_0716.csv",
    "NICS": DATA_DIR / "collet_nics_0716.csv",
    "MCBO": DATA_DIR / "collet_mbco_0716.csv",  # alias
}

OUTPUT_DIR = Path("/home/ubuntu/aroma-dps/p0_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_atom_on_ring(val):
    """Parse atom_on_ring from string '[1, 2, 3]' to list [1, 2, 3]."""
    if isinstance(val, str):
        return ast.literal_eval(val)
    if isinstance(val, list):
        return val
    return []


def audit_molecule(smiles, atom_indices, ring_size_csv):
    """Check both 0-based and 1-based interpretations against RDKit rings.

    Returns dict with results for each interpretation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid_smiles": False}

    mol_h = Chem.AddHs(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    n_total = mol_h.GetNumAtoms()

    ring_info = mol.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]

    result = {
        "valid_smiles": True,
        "n_heavy_atoms": n_heavy,
        "n_total_atoms": n_total,
        "n_rings": len(atom_rings),
        "ring_sizes_in_mol": sorted(set(len(r) for r in atom_rings)),
        "csv_ring_size": ring_size_csv,
        "n_target_atoms": len(atom_indices),
    }

    # --- 0-based interpretation: use indices as-is ---
    idx_0based = [i for i in atom_indices if isinstance(i, (int, float))]
    valid_0 = all(0 <= i < n_heavy for i in idx_0based)
    # Check if these atoms form a ring
    idx_0based_set = set(int(i) for i in idx_0based)
    matched_ring_0 = None
    for ring in atom_rings:
        if idx_0based_set == ring:
            matched_ring_0 = len(ring)
            break
    # Also check subset match (some rings may have extra atoms)
    subset_match_0 = None
    for ring in atom_rings:
        if idx_0based_set.issubset(ring) and len(idx_0based_set) == len(ring):
            subset_match_0 = len(ring)
            break
    # Ring size match
    ring_size_match_0 = (matched_ring_0 == ring_size_csv) if matched_ring_0 else False
    # Element composition
    elements_0 = []
    if valid_0:
        for i in idx_0based:
            elements_0.append(mol.GetAtomWithIdx(int(i)).GetSymbol())
    result["0based"] = {
        "indices": idx_0based,
        "all_valid": valid_0,
        "exact_ring_match": matched_ring_0,
        "ring_size_match": ring_size_match_0,
        "elements": elements_0,
    }

    # --- 1-based interpretation: subtract 1 ---
    idx_1based = [int(i) - 1 for i in atom_indices if isinstance(i, (int, float))]
    valid_1 = all(0 <= i < n_heavy for i in idx_1based)
    idx_1based_set = set(idx_1based)
    matched_ring_1 = None
    for ring in atom_rings:
        if idx_1based_set == ring:
            matched_ring_1 = len(ring)
            break
    ring_size_match_1 = (matched_ring_1 == ring_size_csv) if matched_ring_1 else False
    elements_1 = []
    if valid_1:
        for i in idx_1based:
            elements_1.append(mol.GetAtomWithIdx(int(i)).GetSymbol())
    result["1based"] = {
        "indices": idx_1based,
        "all_valid": valid_1,
        "exact_ring_match": matched_ring_1,
        "ring_size_match": ring_size_match_1,
        "elements": elements_1,
    }

    # --- Determination ---
    score_0 = 0
    score_1 = 0
    if result["0based"]["all_valid"]:
        score_0 += 1
    if result["0based"]["exact_ring_match"] is not None:
        score_0 += 3
    if result["0based"]["ring_size_match"]:
        score_0 += 1

    if result["1based"]["all_valid"]:
        score_1 += 1
    if result["1based"]["exact_ring_match"] is not None:
        score_1 += 3
    if result["1based"]["ring_size_match"]:
        score_1 += 1

    result["score_0based"] = score_0
    result["score_1based"] = score_1
    result["verdict"] = "0based" if score_0 > score_1 else ("1based" if score_1 > score_0 else "ambiguous")
    return result


def audit_dataset(name, path):
    """Audit all rows in a dataset."""
    print(f"\n{'='*60}")
    print(f"Auditing {name}: {path}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Total rows: {len(df)}")

    results = []
    verdicts = Counter()
    mismatches = []

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        atom_on_ring = parse_atom_on_ring(row["atom_on_ring"])
        ring_size = int(row["Ring_Size"])
        ring_id = int(row["Ring_ID"])

        audit = audit_molecule(smiles, atom_on_ring, ring_size)
        audit["row_idx"] = idx
        audit["smiles"] = smiles
        audit["ring_id"] = ring_id
        audit["atom_on_ring_raw"] = str(atom_on_ring)
        audit["dataset"] = name

        results.append(audit)
        verdict = audit.get("verdict", "error")
        verdicts[verdict] += 1

        if verdict != "0based" and verdict != "1based":
            mismatches.append(audit)
        elif verdict == "ambiguous":
            mismatches.append(audit)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(df)} rows...")

    # Summary
    print(f"\n  --- Summary for {name} ---")
    print(f"  Total rows: {len(df)}")
    print(f"  Verdicts: {dict(verdicts)}")
    print(f"  0-based wins: {verdicts.get('0based', 0)} ({verdicts.get('0based', 0)/len(df)*100:.1f}%)")
    print(f"  1-based wins: {verdicts.get('1based', 0)} ({verdicts.get('1based', 0)/len(df)*100:.1f}%)")
    print(f"  Ambiguous: {verdicts.get('ambiguous', 0)}")
    print(f"  Errors: {verdicts.get('error', 0)}")

    # Show some examples of each type
    for v in ["0based", "1based", "ambiguous", "error"]:
        examples = [r for r in results if r.get("verdict") == v][:3]
        if examples:
            print(f"\n  --- Examples ({v}) ---")
            for ex in examples:
                print(f"    SMILES: {ex['smiles'][:50]}")
                print(f"    atom_on_ring: {ex['atom_on_ring_raw']}")
                print(f"    Ring_ID={ex['ring_id']}, CSV Ring_Size={ex['csv_ring_size']}")
                print(f"    n_heavy={ex.get('n_heavy_atoms','?')}, n_rings={ex.get('n_rings','?')}")
                if "0based" in ex:
                    print(f"    0-based: valid={ex['0based']['all_valid']}, "
                          f"ring_match={ex['0based']['exact_ring_match']}, "
                          f"size_match={ex['0based']['ring_size_match']}")
                    print(f"      elements: {ex['0based']['elements']}")
                if "1based" in ex:
                    print(f"    1-based: valid={ex['1based']['all_valid']}, "
                          f"ring_match={ex['1based']['exact_ring_match']}, "
                          f"size_match={ex['1based']['ring_size_match']}")
                    print(f"      elements: {ex['1based']['elements']}")

    # Save detailed results
    out_csv = OUTPUT_DIR / f"atom_on_ring_audit_{name}.csv"
    rows_out = []
    for r in results:
        row_out = {
            "dataset": r["dataset"],
            "row_idx": r["row_idx"],
            "smiles": r["smiles"],
            "ring_id": r["ring_id"],
            "csv_ring_size": r["csv_ring_size"],
            "n_heavy_atoms": r.get("n_heavy_atoms"),
            "n_rings": r.get("n_rings"),
            "n_target_atoms": r.get("n_target_atoms"),
            "atom_on_ring_raw": r["atom_on_ring_raw"],
            "verdict": r.get("verdict"),
            "score_0based": r.get("score_0based"),
            "score_1based": r.get("score_1based"),
            "0based_all_valid": r.get("0based", {}).get("all_valid"),
            "0based_ring_match": r.get("0based", {}).get("exact_ring_match"),
            "0based_size_match": r.get("0based", {}).get("ring_size_match"),
            "0based_elements": str(r.get("0based", {}).get("elements", [])),
            "1based_all_valid": r.get("1based", {}).get("all_valid"),
            "1based_ring_match": r.get("1based", {}).get("exact_ring_match"),
            "1based_size_match": r.get("1based", {}).get("ring_size_match"),
            "1based_elements": str(r.get("1based", {}).get("elements", [])),
        }
        rows_out.append(row_out)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\n  Detailed results saved to: {out_csv}")

    return verdicts, len(df)


def main():
    print("P0-2: atom_on_ring Index-Base Audit")
    print(f"Output directory: {OUTPUT_DIR}")

    all_verdicts = {}
    all_totals = {}

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        verdicts, total = audit_dataset(name, path)
        all_verdicts[name] = verdicts
        all_totals[name] = total

    # Global summary
    print(f"\n{'='*60}")
    print("GLOBAL SUMMARY")
    print(f"{'='*60}")
    for name in all_verdicts:
        v = all_verdicts[name]
        t = all_totals[name]
        print(f"\n  {name} ({t} rows):")
        print(f"    0-based wins: {v.get('0based', 0)} ({v.get('0based', 0)/t*100:.2f}%)")
        print(f"    1-based wins: {v.get('1based', 0)} ({v.get('1based', 0)/t*100:.2f}%)")
        print(f"    Ambiguous: {v.get('ambiguous', 0)} ({v.get('ambiguous', 0)/t*100:.2f}%)")
        print(f"    Errors: {v.get('error', 0)} ({v.get('error', 0)/t*100:.2f}%)")

    # Final determination
    total_0 = sum(v.get("0based", 0) for v in all_verdicts.values())
    total_1 = sum(v.get("1based", 0) for v in all_verdicts.values())
    total_amb = sum(v.get("ambiguous", 0) for v in all_verdicts.values())
    total_err = sum(v.get("error", 0) for v in all_verdicts.values())
    total_all = sum(all_totals.values())

    print(f"\n  OVERALL ({total_all} rows across all datasets):")
    print(f"    0-based: {total_0} ({total_0/total_all*100:.2f}%)")
    print(f"    1-based: {total_1} ({total_1/total_all*100:.2f}%)")
    print(f"    Ambiguous: {total_amb} ({total_amb/total_all*100:.2f}%)")
    print(f"    Errors: {total_err} ({total_err/total_all*100:.2f}%)")

    if total_0 > total_1 * 2:
        conclusion = "0-based"
    elif total_1 > total_0 * 2:
        conclusion = "1-based"
    else:
        conclusion = "MIXED — requires manual investigation"

    print(f"\n  CONCLUSION: atom_on_ring indices are {conclusion}")

    # Save summary JSON
    summary = {
        "audit_type": "atom_on_ring_index_base",
        "datasets": list(DATASETS.keys()),
        "total_rows": total_all,
        "verdicts": {
            "0based": total_0,
            "1based": total_1,
            "ambiguous": total_amb,
            "error": total_err,
        },
        "conclusion": conclusion,
        "per_dataset": {
            name: {"total": all_totals[name], "verdicts": all_verdicts[name]}
            for name in all_verdicts
        },
    }
    summary_path = OUTPUT_DIR / "atom_on_ring_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
