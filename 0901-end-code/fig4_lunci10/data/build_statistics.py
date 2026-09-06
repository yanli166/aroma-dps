"""Phase 1 (stats): 统计 lunci10 manifest 并写入 lunci10_statistics.json (唯一统计来源).

也会输出 duplicate_scaffold_substituent_contexts.csv 解释 N=1389 > 38*30 的原因。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/data/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for p in (PROJ_ROOT, os.path.join(PROJ_ROOT, "code_end")):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(PROJ_ROOT) / "0901-end-code/fig4_lunci10"
AUDIT_OUT = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"
MANIFEST_PATH = AUDIT_OUT / "lunci10_manifest.csv"
STATS_PATH = AUDIT_OUT / "lunci10_statistics.json"
DUP_PATH = AUDIT_OUT / "duplicate_scaffold_substituent_contexts.csv"

DESCRIPTORS = ["HOMA", "NICS_iso", "NICS_ZZ", "MBCO"]


def _percentiles(s: pd.Series, qs=(0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    arr = s.to_numpy()
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(q * 100)}"] = float(np.quantile(arr, q))
    return out


def _descriptor_block(s: pd.Series) -> Dict[str, Any]:
    """Return descriptor statistics including missing, NaN/inf/extreme."""
    raw_count = int(s.shape[0])
    nan_mask = s.isna()
    n_missing = int(nan_mask.sum())
    not_nan = s[~nan_mask]
    inf_count = 0
    finite_count = 0
    if not not_nan.empty:
        arr = not_nan.to_numpy(dtype=float, copy=True)
        inf_mask = ~np.isfinite(arr)
        inf_count = int(inf_mask.sum())
        finite = arr[np.isfinite(arr)]
        finite_count = int(finite.size)
    else:
        finite = np.array([], dtype=float)

    if finite.size:
        finite_min = float(finite.min())
        finite_max = float(finite.max())
        finite_mean = float(finite.mean())
        finite_std = float(finite.std(ddof=0))
        # extreme values: |z| > 6
        if finite_std > 0:
            z = np.abs((finite - finite_mean) / finite_std)
            extreme_count = int((z > 6).sum())
        else:
            extreme_count = 0
    else:
        finite_min = finite_max = finite_mean = finite_std = float("nan")
        extreme_count = 0

    return {
        "n_raw": raw_count,
        "n_missing_or_nan": n_missing,
        "n_inf": inf_count,
        "n_finite": finite_count,
        "n_extreme_6sigma": extreme_count,
        "min": finite_min,
        "max": finite_max,
        "mean": finite_mean,
        "std": finite_std,
        "percentiles": _percentiles(pd.Series(finite)),
    }


def build(manifest_path: Path | None = None) -> Dict[str, Any]:
    manifest_path = manifest_path or MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path} — run build_manifest.py first"
        )

    df = pd.read_csv(manifest_path)

    n_records = int(len(df))
    n_unique_smiles = int(df["canonical_smiles"].nunique(dropna=True))
    n_unique_ring_name = int(df["ring_name"].astype(str).nunique())
    n_unique_sub_name = int(df["sub_name"].astype(str).nunique())
    n_unique_ring_pos = int(df["ring_pos"].astype(str).nunique())
    n_unique_sub_type = int(df["sub_type"].astype(str).nunique())

    df["__ring_x_sub__"] = df["ring_name"].astype(str) + "||" + df["sub_name"].astype(str)
    n_unique_ring_x_sub = int(df["__ring_x_sub__"].nunique())

    df["__ctx__"] = (
        df["ring_name"].astype(str)
        + "||"
        + df["ring_pos"].astype(str)
        + "||"
        + df["sub_name"].astype(str)
    )
    n_unique_context = int(df["__ctx__"].nunique())

    # descriptor blocks
    descriptors: Dict[str, Any] = {}
    for col in DESCRIPTORS:
        if col in df.columns:
            descriptors[col] = _descriptor_block(df[col])
        else:
            descriptors[col] = {"error": "column_not_found"}

    # duplicates table: ring_name × ring_pos × sub_name combos that appear > 1
    df["__dup_key__"] = df["__ctx__"]
    counts = df.groupby("__dup_key__").size().reset_index(name="count")
    dup_rows: List[Dict[str, Any]] = []
    for _, row in counts.iterrows():
        if row["count"] <= 1:
            continue
        key = row["__dup_key__"]
        parts = key.split("||")
        ring_name, ring_pos, sub_name = parts[0], parts[1], parts[2]
        sub = df[df["__dup_key__"] == key]
        # representative substituents (ortho/meta/para on the same ring)
        sub_types = ",".join(sorted({str(x) for x in sub["sub_type"].dropna().unique()}))
        n_unique_smiles_in_ctx = int(sub["canonical_smiles"].nunique())
        n_unique_sub_in_ctx = int(sub["sub_name"].nunique())
        # sample up to 3 rows
        sample = sub.head(3)[["sample_id", "canonical_smiles", "sub_name", "sub_type"]].to_dict("records")
        dup_rows.append({
            "ring_name": ring_name,
            "ring_pos": ring_pos,
            "sub_name": sub_name,
            "n_occurrences": int(row["count"]),
            "n_unique_sub_in_ctx": n_unique_sub_in_ctx,
            "n_unique_smiles_in_ctx": n_unique_smiles_in_ctx,
            "sub_types_present": sub_types,
            "example_rows": str(sample),
        })

    dup_df = pd.DataFrame(dup_rows).sort_values(
        "n_occurrences", ascending=False
    ) if dup_rows else pd.DataFrame(
        columns=["ring_name", "ring_pos", "sub_name", "n_occurrences",
                 "n_unique_sub_in_ctx", "n_unique_smiles_in_ctx",
                 "sub_types_present", "example_rows"]
    )
    dup_df.to_csv(DUP_PATH, index=False)

    stats: Dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "n_records": n_records,
        "n_unique_smiles": n_unique_smiles,
        "n_unique_ring_name": n_unique_ring_name,
        "n_unique_sub_name": n_unique_sub_name,
        "n_unique_ring_pos": n_unique_ring_pos,
        "n_unique_sub_type": n_unique_sub_type,
        "n_unique_ring_name_x_sub_name": n_unique_ring_x_sub,
        "n_unique_ring_name_x_ring_pos_x_sub_name": n_unique_context,
        "descriptors": descriptors,
        "duplicate_contexts_csv": str(DUP_PATH),
        "duplicate_contexts_n": int(len(dup_df)),
        "duplicate_total_excess_rows": int(
            max(0, dup_df["n_occurrences"].sum() - dup_df.shape[0])
        ) if not dup_df.empty else 0,
    }

    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


if __name__ == "__main__":
    out = build()
    print(json.dumps(out, indent=2))
