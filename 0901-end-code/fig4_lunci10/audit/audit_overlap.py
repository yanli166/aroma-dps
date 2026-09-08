"""Phase 0 5-level overlap audit: 比较 lunci10 manifest 与内部训练集.

Levels:
  1. exact canonical SMILES
  2. InChIKey (first 14)
  3. Murcko scaffold
  4. ring-family (canonicalized core ring signature)
  5. substituent (sub_name set overlap)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.inchi import MolToInchiKey

PROJ_ROOT = _PROJ_ROOT
for p in (PROJ_ROOT, f"{PROJ_ROOT}/archive/deprecated/code_end"):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(_PROJ_ROOT) / "0901-end-code/fig4_lunci10"
AUDIT_OUT = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"
MANIFEST_PATH = AUDIT_OUT / "lunci10_manifest.csv"
CONFIG_PATH = FIG4_ROOT / "configs" / "fig4_lunci10.yaml"
OUT_CSV = AUDIT_OUT / "lunci10_overlap_audit.csv"


def _load_config(p: Path) -> Dict[str, Any]:
    import yaml

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _canon(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _inchikey14(smi: str) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MolToInchiKey(m)[:14]
    except Exception:
        return None


def _murcko(smi: str) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else ""


def _ring_family(mol: Chem.Mol) -> str:
    """Coarse ring-family signature: aromatic ring sorted atom-rank string."""
    if mol is None:
        return ""
    ri = mol.GetRingInfo()
    aromatic = [tuple(sorted(r)) for r in ri.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not aromatic:
        return ""
    aromatic.sort(key=lambda r: (-len(r), r))
    return ",".join(str(i) for i in aromatic[0])


def _aggregate_internal(cfg: Dict[str, Any]) -> Dict[str, Set[Any]]:
    """Aggregate SMILES, InChIKey14, Murcko, ring-family, substituents across internal training files."""
    paths = [Path(cfg["paths"]["internal_train_homa"]),
             Path(cfg["paths"]["internal_train_mbco"]),
             Path(cfg["paths"]["internal_train_nics"])]

    smiles_set: Set[str] = set()
    ikey_set: Set[str] = set()
    scaffold_set: Set[str] = set()
    ringfam_set: Set[str] = set()
    sub_set: Set[str] = set()

    for p in paths:
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        # canonical smiles
        smi_col = None
        for cand in ("canonical_smiles", "smiles", "SMILES", "mol"):
            if cand in df.columns:
                smi_col = cand
                break
        if smi_col is None:
            continue
        for s in df[smi_col].dropna().astype(str):
            c = _canon(s)
            if c:
                smiles_set.add(c)
            mol = Chem.MolFromSmiles(c) if c else None
            if mol is not None:
                ik = _inchikey14(c)
                if ik:
                    ikey_set.add(ik)
                sc = _murcko(c)
                if sc:
                    scaffold_set.add(sc)
                rf = _ring_family(mol)
                if rf:
                    ringfam_set.add(rf)

        for col in ("sub_name", "substituent", "Substituent"):
            if col in df.columns:
                for v in df[col].dropna().astype(str):
                    sub_set.add(v.strip())

    return {
        "smiles": smiles_set,
        "inchikey14": ikey_set,
        "scaffold": scaffold_set,
        "ring_family": ringfam_set,
        "substituent": sub_set,
    }


def audit_overlap(
    manifest_path: Path | None = None,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    manifest_path = manifest_path or MANIFEST_PATH
    config_path = config_path or CONFIG_PATH

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}; run build_manifest.py first")

    cfg = _load_config(config_path)
    df = pd.read_csv(manifest_path)

    repo = _aggregate_internal(cfg)
    n_internal_smiles = len(repo["smiles"])
    n_internal_ikey = len(repo["inchikey14"])
    n_internal_scaffold = len(repo["scaffold"])
    n_internal_rf = len(repo["ring_family"])
    n_internal_sub = len(repo["substituent"])

    flags = {
        "exact_molecule_seen": [],
        "inchikey_seen": [],
        "scaffold_seen": [],
        "ring_family_seen": [],
        "substituent_seen": [],
    }
    for _, row in df.iterrows():
        smi = row.get("canonical_smiles", "")
        flags["exact_molecule_seen"].append(bool(smi) and smi in repo["smiles"])
        ik = _inchikey14(smi) if smi else None
        flags["inchikey_seen"].append(bool(ik) and ik in repo["inchikey14"])
        sc = _murcko(smi) if smi else None
        flags["scaffold_seen"].append(bool(sc) and sc in repo["scaffold"])
        mol = Chem.MolFromSmiles(smi) if smi else None
        rf = _ring_family(mol) if mol else ""
        flags["ring_family_seen"].append(bool(rf) and rf in repo["ring_family"])
        sub = row.get("sub_name", "")
        flags["substituent_seen"].append(bool(sub) and str(sub).strip() in repo["substituent"])

    for k, v in flags.items():
        df[k] = v

    # Aggregate summary
    summary = {k: int(sum(v)) for k, v in flags.items()}
    summary["n_total"] = int(len(df))
    summary["n_internal_smiles"] = n_internal_smiles
    summary["n_internal_inchikey14"] = n_internal_ikey
    summary["n_internal_scaffold"] = n_internal_scaffold
    summary["n_internal_ring_family"] = n_internal_rf
    summary["n_internal_substituent"] = n_internal_sub
    summary["exact_overlap_rate"] = (
        summary["exact_molecule_seen"] / summary["n_total"] if summary["n_total"] else 0.0
    )
    summary["scaffold_overlap_rate"] = (
        summary["scaffold_seen"] / summary["n_total"] if summary["n_total"] else 0.0
    )
    summary["substituent_overlap_rate"] = (
        summary["substituent_seen"] / summary["n_total"] if summary["n_total"] else 0.0
    )

    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    summary["output_csv"] = str(OUT_CSV)
    return summary


if __name__ == "__main__":
    out = audit_overlap()
    print(json.dumps(out, indent=2, ensure_ascii=False))
