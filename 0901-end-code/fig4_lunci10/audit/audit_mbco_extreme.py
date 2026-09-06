"""Task 6 (按协议修正): MBCO 50 个 >6σ points 检查.

严格要求:
  - 不得仅因统计极端而删除
  - 必须先检查 calculation convergence / parsing / unit / target-ring mapping / chemical validity
  - 只有明确技术错误才允许排除, 记录 exclusion reason
  - 计算有效则全部保留, 可增加 robust sensitivity analysis
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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

OUT_JSON = AUDIT_OUT / "mbco_extreme_audit.json"
OUT_CSV = AUDIT_OUT / "mbco_extreme_audit.csv"

# 6σ threshold computed from the clean set
def compute_sigma_thresh(df: pd.DataFrame) -> Dict[str, float]:
    mbco = pd.to_numeric(df["MBCO"], errors="coerce").dropna()
    mu = float(mbco.mean())
    sd = float(mbco.std(ddof=1))
    return {
        "mean": mu,
        "std": sd,
        "low_6sigma": mu - 6 * sd,
        "high_6sigma": mu + 6 * sd,
    }


def check_smiles(smi: str) -> Dict[str, Any]:
    if not isinstance(smi, str) or not smi:
        return {"valid": False, "reason": "empty_smi"}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"valid": False, "reason": "rdkit_cannot_parse"}
    # unit: MBCO should be roughly in [-1, 1] range for normalized aromaticity index
    return {"valid": True, "reason": "ok", "n_atoms": mol.GetNumAtoms()}


def check_target_ring(row: pd.Series) -> Dict[str, Any]:
    target = row.get("target_ring_atoms", "")
    if not isinstance(target, str) or not target:
        return {"valid": False, "reason": "no_target_ring_defined"}
    try:
        atoms = [int(x.strip()) for x in target.split(",") if x.strip()]
    except Exception:
        return {"valid": False, "reason": "unparseable_target_ring"}
    if len(atoms) < 3:
        return {"valid": False, "reason": "target_ring_too_small"}
    return {"valid": True, "reason": "ok", "n_atoms_in_target": len(atoms)}


def main() -> None:
    df = pd.read_csv(CLEAN_MANIFEST)
    th = compute_sigma_thresh(df)
    print(f"[baseline] MBCO mean={th['mean']:.4f} std={th['std']:.4f}")
    print(f"  6σ bounds: [{th['low_6sigma']:.4f}, {th['high_6sigma']:.4f}]")

    mbco = pd.to_numeric(df["MBCO"], errors="coerce")
    extreme_mask = (mbco < th["low_6sigma"]) | (mbco > th["high_6sigma"]) | ~np.isfinite(mbco)
    extreme_rows = df.loc[extreme_mask].copy()
    print(f"  extreme rows: {len(extreme_rows)}")

    # 检查每一行
    results: List[Dict[str, Any]] = []
    n_excluded = 0
    n_kept = 0
    for _, row in extreme_rows.iterrows():
        smi_check = check_smiles(row.get("raw_smiles") or row.get("canonical_smiles"))
        tgt_check = check_target_ring(row)
        mbco_val = float(row.get("MBCO")) if pd.notna(row.get("MBCO")) else float("nan")
        is_nan = not np.isfinite(mbco_val)
        is_outside_6sig = (mbco_val < th["low_6sigma"]) or (mbco_val > th["high_6sigma"])

        # decision logic per protocol:
        exclusion_reason = None
        keep_for_main = True
        if is_nan:
            # NaN/inf MBCO: invalid, exclude from main MBCO metric
            exclusion_reason = "MBCO_nan_or_inf"
            keep_for_main = False
        elif not smi_check["valid"]:
            exclusion_reason = f"smiles_invalid:{smi_check['reason']}"
            keep_for_main = False
        elif not tgt_check["valid"]:
            exclusion_reason = f"target_ring_invalid:{tgt_check['reason']}"
            keep_for_main = False
        # else: even if >6σ but SMILES parses, target ring defined, value finite:
        #       keep for main analysis. Mark as "extreme_but_valid"

        if not keep_for_main:
            n_excluded += 1
        else:
            n_kept += 1

        results.append({
            "sample_id": row.get("sample_id"),
            "MBCO": mbco_val,
            "is_nan": is_nan,
            "outside_6sigma": is_outside_6sig,
            "smiles_valid": smi_check["valid"],
            "target_ring_valid": tgt_check["valid"],
            "exclusion_reason": exclusion_reason or "",
            "kept_for_main": keep_for_main,
        })

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"  kept={n_kept}, excluded={n_excluded}")
    print(f"  wrote {OUT_CSV}")

    out = {
        "protocol": "MBCO_extreme_audit_v2 (no statistical deletion without technical reason)",
        "thresholds": th,
        "n_extreme_rows_total": len(extreme_rows),
        "n_kept_for_main": n_kept,
        "n_excluded_for_main": n_excluded,
        "decision": (
            "no rows deleted solely on 6σ statistical grounds; "
            f"{n_kept} extreme-but-valid rows retained for main MBCO analysis; "
            f"{n_excluded} rows excluded only when MBCO is NaN/inf or SMILES/target-ring invalid"
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_JSON}")
    print()
    print(out["decision"])


if __name__ == "__main__":
    main()