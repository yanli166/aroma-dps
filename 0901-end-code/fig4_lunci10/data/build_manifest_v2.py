"""build_manifest v2 (FIXED): use corrected CSV's New_ID (molecule-level) + Ring_ID.

Critical bug fix:
  v1 used df.iterrows() index (0..2152) as sample_id (e1..e2153), losing the
  real molecule-level ID (New_ID = e1..e1389). v1 also recomputed target rings
  from scratch instead of using the authoritative Ring_Atoms in the corrected CSV.

v2 protocol:
  - sample_id = New_ID (e1..e1389, molecule-level, matches .gjf/.log/.fchk)
  - target_ring_id = Ring_ID (1, 2, 3)
  - target_ring_atoms = Ring_Atoms (from corrected CSV, e.g. "[4,9,8,7,6,5]")
  - descriptor values (HOMA, MBCO, NICS_iso, NICS_ZZ) come directly from corrected CSV
  - ring_name / ring_pos / sub_name / sub_type / fused etc. merged from begin.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# fig4_lunci10/data/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(str(THIS_DIR)))))
CODE_END = PROJ_ROOT / "archive/deprecated/code_end"
FIG4_ROOT = PROJ_ROOT / "0901-end-code/fig4_lunci10"
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
CONFIG_PATH = FIG4_ROOT / "configs" / "fig4_lunci10.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f) if path.suffix == ".json" else __import__("yaml").safe_load(f)


def safe_canonical(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else ""


def murcko_scaffold(smi: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""


def parse_ring_atoms(atom_str: str) -> List[int]:
    """Parse "[4,9,8,7,6,5]" or "4,9,8,7,6,5" -> [4,9,8,7,6,5]."""
    if not isinstance(atom_str, str) or not atom_str:
        return []
    s = atom_str.replace("[", "").replace("]", "").replace(" ", "")
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []


def ring_size_and_fused(mol: Chem.Mol) -> Dict[str, Any]:
    if mol is None:
        return {"ring_size": None, "fused": None, "non_fused": None}
    ri = mol.GetRingInfo()
    aromatic = [r for r in ri.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not aromatic:
        return {"ring_size": None, "fused": None, "non_fused": True}
    sizes = [len(r) for r in aromatic]
    fused = len(aromatic) > 1
    # determine fused by shared atoms
    if not fused and len(aromatic) > 1:
        for i in range(len(aromatic)):
            for j in range(i+1, len(aromatic)):
                if set(aromatic[i]) & set(aromatic[j]):
                    fused = True
                    break
            if fused:
                break
    return {
        "ring_size": max(sizes),
        "fused": bool(fused),
        "non_fused": not bool(fused),
    }


def heteroatom_composition(mol: Chem.Mol) -> str:
    if mol is None:
        return ""
    counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "C" and sym != "H":
            counts[sym] = counts.get(sym, 0) + 1
    if not counts:
        return "C_only"
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def main() -> None:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(CONFIG_PATH)
    paths = cfg["paths"]

    corrected_path = Path(paths["lunci10_test"])
    if not corrected_path.is_absolute():
        corrected_path = Path(PROJ_ROOT) / corrected_path
    begin_path = Path(paths["lunci10_begin"])
    if not begin_path.is_absolute():
        begin_path = Path(PROJ_ROOT) / begin_path
    if not corrected_path.is_file():
        raise FileNotFoundError(f"corrected data not found: {corrected_path}")
    if not begin_path.is_file():
        raise FileNotFoundError(f"begin annotation not found: {begin_path}")

    df = pd.read_csv(corrected_path)
    begin = pd.read_csv(begin_path)
    print(f"[load] corrected: {len(df)} rows, {df['New_ID'].nunique()} unique molecules")
    print(f"[load] begin: {len(begin)} rows")

    # Merge corrected + begin by molecule-level ID
    # corrected: New_ID (e1..e1389)
    # begin: no (e1..e1389) — rename to New_ID for merge
    if "no" in begin.columns and "New_ID" in df.columns:
        begin_merge = begin.rename(columns={"no": "New_ID"})
        df = df.merge(begin_merge, on="New_ID", how="left", suffixes=("", "_begin"))
        print(f"[merge] on New_ID (molecule-level), merged shape: {df.shape}")
    else:
        # try smiles
        for cand in ("smiles", "SMILES"):
            if cand in begin.columns and cand in df.columns:
                df = df.merge(begin, on=cand, how="left", suffixes=("", "_begin"))
                print(f"[merge] on {cand}, merged shape: {df.shape}")
                break
        else:
            raise RuntimeError(f"Cannot merge corrected and begin")

    # Pick SMILES column
    smi_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else None)
    if smi_col is None:
        raise RuntimeError(f"No SMILES-like column in {list(df.columns)}")

    # Build manifest rows — KEY FIX: use New_ID as sample_id, Ring_Atoms as target_ring_atoms
    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_smi = row.get(smi_col)
        canonical = safe_canonical(str(raw_smi)) if raw_smi is not None else ""
        mol = Chem.MolFromSmiles(canonical) if canonical else None
        scaffold = murcko_scaffold(canonical) if canonical else ""
        ring_info = ring_size_and_fused(mol) if mol else {"ring_size": None, "fused": None, "non_fused": None}
        hetero = heteroatom_composition(mol) if mol else ""

        # KEY FIX: use corrected CSV's Ring_Atoms (authoritative, from .gjf/.log analysis)
        ring_atoms_str = str(row.get("Ring_Atoms", ""))
        ring_atoms_list = parse_ring_atoms(ring_atoms_str)
        ring_id = row.get("Ring_ID")

        # ring_name / ring_pos / sub_name / sub_type / fused from begin.csv (molecule-level)
        ring_name = row.get("ring_name") if "ring_name" in df.columns else None
        ring_pos = row.get("ring_pos") if "ring_pos" in df.columns else None
        sub_name = row.get("sub_name") if "sub_name" in df.columns else None
        sub_type = row.get("sub_type") if "sub_type" in df.columns else None

        # descriptors from corrected CSV (authoritative, from DFT logs)
        homa = row.get("HOMA") if "HOMA" in df.columns else None
        nics_iso = row.get("NICS_iso") if "NICS_iso" in df.columns else None
        nics_zz = row.get("NICS_ZZ") if "NICS_ZZ" in df.columns else (
            row.get("NICS_1zz") if "NICS_1zz" in df.columns else None
        )
        mbco = row.get("MBCO") if "MBCO" in df.columns else None

        # KEY FIX: sample_id = New_ID (e1..e1389, molecule-level, matches .gjf/.log/.fchk)
        out_rows.append({
            "sample_id": str(row.get("New_ID", "")),  # molecule-level ID
            "canonical_smiles": canonical,
            "raw_smiles": str(raw_smi) if raw_smi is not None else "",
            "ring_name": ring_name if ring_name is not None else "",
            "Ring_ID": int(ring_id) if pd.notna(ring_id) else "",
            "ring_pos": ring_pos if ring_pos is not None else "",
            "target_ring_id": int(ring_id) if pd.notna(ring_id) else "",
            "target_ring_atoms": ring_atoms_str,  # authoritative, from corrected CSV
            "sub_name": sub_name if sub_name is not None else "",
            "sub_type": sub_type if sub_type is not None else "",
            "fused": ring_info["fused"] if ring_info["fused"] is not None else "",
            "non_fused": ring_info["non_fused"] if ring_info["non_fused"] is not None else "",
            "ring_size": row.get("Ring_Size") if "Ring_Size" in df.columns else ring_info["ring_size"],
            "heteroatom_composition": hetero,
            "murcko_scaffold": scaffold,
            "HOMA": homa,
            "NICS_iso": nics_iso,
            "NICS_ZZ": nics_zz,
            "MBCO": mbco,
            "source": "lunci10-test-corrected-v2",
        })

    manifest = pd.DataFrame(out_rows)
    out_csv = AUDIT_OUT / "lunci10_manifest.csv"
    manifest.to_csv(out_csv, index=False)
    print(f"[write] {out_csv} ({len(manifest)} rows)")

    # Verify
    n_unique_mol = manifest["sample_id"].nunique()
    n_unique_smi = manifest["canonical_smiles"].nunique()
    print(f"\n[verify] unique sample_ids (molecules): {n_unique_mol}")
    print(f"[verify] unique canonical_smiles: {n_unique_smi}")
    print(f"[verify] rows: {len(manifest)}")

    # Cross-check with corrected CSV
    corrected_ids = set(df["New_ID"].astype(str).unique().tolist())
    manifest_ids = set(manifest["sample_id"].astype(str).unique().tolist())
    print(f"[verify] IDs match corrected: {len(corrected_ids & manifest_ids)} / {len(corrected_ids)} ({len(corrected_ids & manifest_ids)/len(corrected_ids)*100:.1f}%)")

    # sample ID range
    id_nums = sorted([int(x[1:]) for x in manifest_ids if x.startswith("e")])
    print(f"[verify] sample_id range: e{id_nums[0]} .. e{id_nums[-1]}")


if __name__ == "__main__":
    main()