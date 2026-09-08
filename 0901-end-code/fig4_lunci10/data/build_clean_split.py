"""Task 2 (按协议修正): 建立 lunci10_clean_external 与 lunci10_exact_seen.

严格要求 (来自用户协议):
  - 不修改 internal training data, 不重训 Fig.3 frozen models
  - 排除按 molecule identity 完成: 若某个 exact-overlap molecule 在 lunci10_unified.csv 中
    对应多个 ring-level records, 则所有对应 records 全部进入 exact_seen, 一律不进入 clean_external
  - 禁止硬编码 "77 行"
  - 输出:
      lunci10_clean_manifest.csv
      lunci10_exact_seen_manifest.csv
      clean_split_statistics.json
  - 重新统计两个集合的:
      * N unique molecules
      * N ring-level records
      * 每个 descriptor 的 valid N
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.inchi import MolToInchiKey

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "archive/deprecated/code_end"
FIG4_ROOT = PROJ_ROOT / "0901-end-code/fig4_lunci10"
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"

# 输入
LUNCI10_MANIFEST = AUDIT_OUT / "lunci10_manifest.csv"
INTERNAL_HOMA = CODE_END / "data1_end/collet_homa_0716.csv"
INTERNAL_MBCO = CODE_END / "data1_end/collet_mbco_0716.csv"
INTERNAL_NICS = CODE_END / "data1_end/collet_nics_0716.csv"

# 输出
CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"
SEEN_MANIFEST = AUDIT_OUT / "lunci10_exact_seen_manifest.csv"
STATS_JSON = AUDIT_OUT / "clean_split_statistics.json"


def canonical_smiles(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def inchikey14(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        return MolToInchiKey(mol)[:14]
    except Exception:
        return ""


def collect_internal_identity() -> Dict[str, Set[str]]:
    """聚合所有 internal training 数据的 canonical SMILES 与 InChIKey14.

    Note: 严格按用户协议, 不修改 internal training data. 只读取其 identity.
    """
    canonicals: Set[str] = set()
    inchikeys: Set[str] = set()

    for path in (INTERNAL_HOMA, INTERNAL_MBCO, INTERNAL_NICS):
        df = pd.read_csv(path)
        for s in df["smiles"].dropna().astype(str).tolist():
            c = canonical_smiles(s)
            if c:
                canonicals.add(c)
            ik = inchikey14(s)
            if ik:
                inchikeys.add(ik)

    return {"canonical": canonicals, "inchikey14": inchikeys}


def main() -> None:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    print("[load] lunci10 manifest...")
    df = pd.read_csv(LUNCI10_MANIFEST)
    n_ring_records_total = len(df)
    print(f"  ring-level records: {n_ring_records_total}")

    print("[internal] collect identity from all 3 training files...")
    identity = collect_internal_identity()
    print(f"  unique canonical SMILES in internal: {len(identity['canonical'])}")
    print(f"  unique InChIKey14 in internal: {len(identity['inchikey14'])}")

    # ---- per-row overlap flags (canonical & inchikey from raw_smiles) ----
    # 用 raw_smiles 而不是 canonical_smiles, 内部数据集会自己 canonicalize 后再比对
    canon_l10 = []
    ik_l10 = []
    for s in df["raw_smiles"].astype(str).tolist():
        canon_l10.append(canonical_smiles(s))
        ik_l10.append(inchikey14(s))
    df["_canonical_smiles"] = canon_l10
    df["_inchikey14"] = ik_l10

    exact_seen_by_smiles = df["_canonical_smiles"].isin(identity["canonical"])
    exact_seen_by_ik = df["_inchikey14"].isin(identity["inchikey14"])
    # 任一 identity 命中即视为 exact-seen
    exact_seen_mask = exact_seen_by_smiles | exact_seen_by_ik

    # ---- molecule-level exclusion (per protocol: molecule identity, not row identity) ----
    # 同一 canonical SMILES 的所有 ring-level records 必须同进同出
    seen_molecules = set(df.loc[exact_seen_mask, "_canonical_smiles"].unique().tolist())
    seen_molecules = {m for m in seen_molecules if m}  # drop empty
    print(f"  unique molecules (canonical SMILES) that match internal: {len(seen_molecules)}")

    molecule_seen_mask = df["_canonical_smiles"].isin(seen_molecules)
    n_mol_seen = df.loc[molecule_seen_mask, "_canonical_smiles"].nunique()
    n_mol_clean = df.loc[~molecule_seen_mask, "_canonical_smiles"].nunique()
    n_rows_seen = int(molecule_seen_mask.sum())
    n_rows_clean = int((~molecule_seen_mask).sum())
    print(f"  exact-seen molecules: {n_mol_seen} / rows: {n_rows_seen}")
    print(f"  clean molecules: {n_mol_clean} / rows: {n_rows_clean}")

    # ---- write two manifests ----
    df_clean = df.loc[~molecule_seen_mask].drop(columns=["_canonical_smiles", "_inchikey14"]).copy()
    df_seen = df.loc[molecule_seen_mask].drop(columns=["_canonical_smiles", "_inchikey14"]).copy()
    df_clean.to_csv(CLEAN_MANIFEST, index=False)
    df_seen.to_csv(SEEN_MANIFEST, index=False)
    print(f"  wrote {CLEAN_MANIFEST}")
    print(f"  wrote {SEEN_MANIFEST}")

    # ---- per-task valid N for each subset ----
    descriptors = {
        "HOMA": "HOMA",
        "NICS_iso": "NICS_iso",
        "NICS_ZZ": "NICS_ZZ",
        "MBCO": "MBCO",
    }

    def valid_n(sub: pd.DataFrame, col: str) -> int:
        s = pd.to_numeric(sub[col], errors="coerce")
        return int(s.notna().sum())

    clean_stats = {k: valid_n(df_clean, v) for k, v in descriptors.items()}
    seen_stats = {k: valid_n(df_seen, v) for k, v in descriptors.items()}

    stats = {
        "protocol": "molecule_identity_exclusion",
        "rules": [
            "if any ring-level record of a molecule has canonical SMILES or InChIKey14 in internal training, ALL that molecule's ring-level records go to exact_seen",
            "exact-seen rows are diagnostic only, never mixed into main external MAE/RMSE/R2",
        ],
        "internal_identity_source": [
            str(INTERNAL_HOMA.relative_to(PROJ_ROOT)),
            str(INTERNAL_MBCO.relative_to(PROJ_ROOT)),
            str(INTERNAL_NICS.relative_to(PROJ_ROOT)),
        ],
        "n_internal_unique_canonical": len(identity["canonical"]),
        "n_internal_unique_inchikey14": len(identity["inchikey14"]),
        "lunci10_total": {
            "ring_records": n_ring_records_total,
            "unique_molecules": int(df["_canonical_smiles"].nunique()),
        },
        "lunci10_clean_external": {
            "ring_records": n_rows_clean,
            "unique_molecules": n_mol_clean,
            "valid_labels": clean_stats,
        },
        "lunci10_exact_seen": {
            "ring_records": n_rows_seen,
            "unique_molecules": n_mol_seen,
            "valid_labels": seen_stats,
            "note": "diagnostic only; MUST NOT enter main external MAE/RMSE/R2",
        },
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {STATS_JSON}")
    print()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()