"""Phase 1 (data prep): 从 lunci10-test-corrected.csv 生成 lunci10_manifest.csv.

合并 begin.csv 提供 ring/sub 信息并使用 RDKit 计算 Murcko scaffold,
fused 标记, 杂原子组成, canonical SMILES 等。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/data/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for p in (PROJ_ROOT, os.path.join(PROJ_ROOT, "code_end")):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(PROJ_ROOT) / "0901-end-code/fig4_lunci10"
AUDIT_OUT = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"

CONFIG_PATH = FIG4_ROOT / "configs" / "fig4_lunci10.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------- RDKit helpers ---------------------------

def safe_canonical(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def murcko_scaffold(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        if scaff is None or scaff.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaff)
    except Exception:
        return None


def ring_size_and_fused(mol: Chem.Mol) -> Dict[str, Any]:
    """Return ring_size (smallest aromatic atom-ring) and fused flag (multiple aromatic rings)."""
    if mol is None:
        return {"ring_size": None, "fused": None}

    ri = mol.GetRingInfo()
    if ri is None or ri.NumRings() == 0:
        return {"ring_size": 0, "fused": False}

    aromatic_ring_sizes: List[int] = []
    total_rings = 0
    for atoms_in_ring in ri.AtomRings():
        total_rings += 1
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms_in_ring):
            aromatic_ring_sizes.append(len(atoms_in_ring))

    fused = total_rings >= 2
    ring_size = min(aromatic_ring_sizes) if aromatic_ring_sizes else (
        min(len(r) for r in ri.AtomRings()) if ri.AtomRings() else 0
    )
    return {"ring_size": int(ring_size), "fused": bool(fused)}


def heteroatom_composition(mol: Chem.Mol) -> str:
    if mol is None:
        return ""
    counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C" or sym == "H":
            continue
        counts[sym] = counts.get(sym, 0) + 1
    if not counts:
        return "C_only"
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def find_target_ring(mol: Chem.Mol) -> Dict[str, Any]:
    """Pick the 'target aromatic ring'.

    Preference order:
      1. Match by ring_name annotation (begin.csv uses names like benzene, pyridine, ...).
      2. Fallback: largest aromatic ring.
    Returns dict with target_ring_id (smarts-style index string), target_ring_atoms (list[int]).
    """
    if mol is None:
        return {"target_ring_id": "", "target_ring_atoms": []}

    ri = mol.GetRingInfo()
    aromatic = [tuple(sorted(r)) for r in ri.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not aromatic:
        return {"target_ring_id": "", "target_ring_atoms": []}

    aromatic.sort(key=lambda r: (-len(r), r))
    chosen = aromatic[0]
    return {
        "target_ring_id": ",".join(str(i) for i in chosen),
        "target_ring_atoms": list(chosen),
    }


# --------------------------- merge & build ---------------------------

def _build_sample_id(row_idx: int) -> str:
    return f"e{row_idx + 1}"


def build(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or _load_config(CONFIG_PATH)
    paths = cfg["paths"]

    test_path = Path(paths["lunci10_test"])
    if not test_path.is_absolute():
        test_path = Path(PROJ_ROOT) / test_path
    begin_path = Path(paths["lunci10_begin"])
    if not begin_path.is_absolute():
        begin_path = Path(PROJ_ROOT) / begin_path
    if not test_path.is_file():
        raise FileNotFoundError(f"corrected data not found: {test_path}")
    if not begin_path.is_file():
        raise FileNotFoundError(f"begin annotation not found: {begin_path}")

    df = pd.read_csv(test_path)
    begin = pd.read_csv(begin_path)

    # Determine common join key. begin.csv usually has 'name' or 'smiles'.
    # Prefer an id-like column; fall back to smiles.
    # Special handling: corrected uses 'New_ID' (e.g. e1..e1389) while begin uses 'no'.
    join_key = None
    for cand in ("sample_id", "id", "name", "New_ID", "no"):
        if cand in begin.columns and cand in df.columns:
            join_key = cand
            break
    if join_key is None:
        # fall back to smiles match
        for cand in ("smiles", "SMILES", "canonical_smiles"):
            if cand in begin.columns and cand in df.columns:
                join_key = cand
                break
    if join_key is None:
        raise RuntimeError(
            f"Cannot find shared join key between corrected ({list(df.columns)}) "
            f"and begin ({list(begin.columns)})"
        )

    if join_key == "no":
        # corrected uses 'New_ID', rename begin's 'no' to 'New_ID' for the merge
        begin_merge = begin.rename(columns={"no": "New_ID"})
        df = df.merge(begin_merge, on="New_ID", how="left", suffixes=("", "_begin"))
    elif join_key == "New_ID":
        df = df.merge(begin.rename(columns={"no": "New_ID"}), on="New_ID", how="left", suffixes=("", "_begin"))
    else:
        df = df.merge(begin, on=join_key, how="left", suffixes=("", "_begin"))

    # Pick SMILES column robustly
    smi_col = None
    for cand in ("smiles", "SMILES", "mol", "canonical_smiles"):
        if cand in df.columns:
            smi_col = cand
            break
    if smi_col is None:
        raise RuntimeError(f"No SMILES-like column in {list(df.columns)}")

    out_rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        raw_smi = row.get(smi_col)
        canonical = safe_canonical(str(raw_smi)) if raw_smi is not None else None
        mol = Chem.MolFromSmiles(canonical) if canonical else None
        scaffold = murcko_scaffold(canonical) if canonical else None
        ring_info = ring_size_and_fused(mol) if mol else {"ring_size": None, "fused": None}
        target = find_target_ring(mol) if mol else {"target_ring_id": "", "target_ring_atoms": []}
        hetero = heteroatom_composition(mol) if mol else ""

        # ring_id hint (optional)
        ring_id = None
        for cand in ("Ring_ID", "ring_id", "ringName", "ring_name_id"):
            if cand in df.columns and not pd.isna(row.get(cand)):
                ring_id = row.get(cand)
                break

        ring_name = row.get("ring_name") if "ring_name" in df.columns else None
        ring_pos = row.get("ring_pos") if "ring_pos" in df.columns else None
        sub_name = row.get("sub_name") if "sub_name" in df.columns else None
        sub_type = row.get("sub_type") if "sub_type" in df.columns else None

        # Pick descriptor columns by name (HOMA / NICS variants / MBCO)
        homa = row.get("HOMA") if "HOMA" in df.columns else None
        nics_iso = row.get("NICS_iso") if "NICS_iso" in df.columns else None
        nics_zz = row.get("NICS_ZZ") if "NICS_ZZ" in df.columns else (
            row.get("NICS_1zz") if "NICS_1zz" in df.columns else None
        )
        mbco = row.get("MBCO") if "MBCO" in df.columns else None

        out_rows.append({
            "sample_id": _build_sample_id(idx),
            "canonical_smiles": canonical if canonical is not None else "",
            "raw_smiles": raw_smi if raw_smi is not None else "",
            "ring_name": ring_name if ring_name is not None else "",
            "Ring_ID": ring_id if ring_id is not None else "",
            "ring_pos": ring_pos if ring_pos is not None else "",
            "target_ring_id": target["target_ring_id"],
            "target_ring_atoms": ",".join(str(i) for i in target["target_ring_atoms"]),
            "sub_name": sub_name if sub_name is not None else "",
            "sub_type": sub_type if sub_type is not None else "",
            "fused": ring_info["fused"] if ring_info["fused"] is not None else "",
            "non_fused": (not ring_info["fused"]) if ring_info["fused"] is not None else "",
            "ring_size": ring_info["ring_size"] if ring_info["ring_size"] is not None else "",
            "heteroatom_composition": hetero,
            "murcko_scaffold": scaffold if scaffold is not None else "",
            "HOMA": homa,
            "NICS_iso": nics_iso,
            "NICS_ZZ": nics_zz,
            "MBCO": mbco,
            "source": "lunci10-test-corrected",
        })

    manifest = pd.DataFrame(out_rows)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    out_csv = AUDIT_OUT / "lunci10_manifest.csv"
    manifest.to_csv(out_csv, index=False)

    summary = {
        "n_records": int(len(manifest)),
        "n_unique_smiles": int(manifest["canonical_smiles"].nunique()),
        "out_csv": str(out_csv),
        "join_key": join_key,
        "smiles_column": smi_col,
    }
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps(s, indent=2))
