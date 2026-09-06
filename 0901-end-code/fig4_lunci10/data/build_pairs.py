"""Phase 6 (pair building): 基于 lunci10 manifest 构建 pair manifest.

化学 context key:
  - key = (ring_name, ring_pos, target_ring_id, sub_type if necessary)
  - 在同一 context 内, 仅保留 standardized sub_name 排序后的 (i, j) (i < j)
  - ΔA_ij = A_i - A_j (orientation deterministic)
  - NICS sign convention: ΔNICS = NICS_i - NICS_j

主要产物:
  - pair_context_audit.csv   每个 context: N molecules, N substituents, possible pair count
  - lunci10_pair_manifest.csv canonical pair (i, j) 全集
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

OUT_DIR = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/03_pairwise"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAIR_CTX_CSV = OUT_DIR / "pair_context_audit.csv"
PAIR_MANIFEST_CSV = OUT_DIR / "lunci10_pair_manifest.csv"


def build_context_key(row: pd.Series) -> Tuple[str, str, str, str]:
    """Construct deterministic chemical context key.

    key = (ring_name, ring_pos, target_ring_id, sub_type)
    """
    return (
        str(row.get("ring_name", "")),
        str(row.get("ring_pos", "")),
        str(row.get("target_ring_id", "")),
        str(row.get("sub_type", "")),
    )


def build(
    manifest_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main entrypoint.

    Steps:
      1. 读 manifest, 按 (ring_name, ring_pos, target_ring_id, sub_type) 分组
      2. 每个 context 内按 standardized sub_name 排序, 枚举 i<j
      3. canonical pair: (sub_i < sub_j) → 仅保留一次 (i, j) 而非 (j, i)
      4. 对每个 pair 计算 ΔA = A_i - A_j (主指标 = MAE)
      5. 输出 context_audit 与 pair_manifest
    """
    manifest_path = manifest_path or MANIFEST_PATH
    out_dir = out_dir or OUT_DIR

    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)

    # ---- 0. 标准化 sub_name ----
    df["__sub_std__"] = df["sub_name"].astype(str).str.strip()
    df["__ctx__"] = df.apply(build_context_key, axis=1)

    # ---- 1. context audit ----
    ctx_groups = df.groupby("__ctx__")
    ctx_rows: List[Dict[str, Any]] = []
    for ctx_key, sub in ctx_groups:
        n_mol = int(len(sub))
        n_unique_sub = int(sub["__sub_std__"].nunique())
        # possible pair = C(n_unique_sub, 2) — canonical i<j
        n_pairs_possible = int(n_unique_sub * (n_unique_sub - 1) // 2)
        # 实际能形成的 canonical pair 数 (受多分子同一 sub_name 影响; 取 max unique)
        # 这里理论值 = C(n_unique_sub, 2)
        ctx_rows.append({
            "ring_name": ctx_key[0],
            "ring_pos": ctx_key[1],
            "target_ring_id": ctx_key[2],
            "sub_type": ctx_key[3],
            "n_molecules": n_mol,
            "n_unique_substituents": n_unique_sub,
            "n_pairs_possible": n_pairs_possible,
        })
    ctx_df = pd.DataFrame(ctx_rows).sort_values("n_molecules", ascending=False)
    ctx_df.to_csv(out_dir / PAIR_CTX_CSV.name, index=False)
    if verbose:
        print(f"[pair] contexts={len(ctx_df)}, total possible pairs={int(ctx_df['n_pairs_possible'].sum())}")

    # ---- 2. 在每个 context 内枚举 i<j ----
    pair_rows: List[Dict[str, Any]] = []
    descriptor_cols = {
        "HOMA": "HOMA",
        "NICS_iso": "NICS_iso",
        "NICS_ZZ": "NICS_ZZ",  # ΔNICS = NICS_i - NICS_j (sign preserved)
        "MBCO": "MBCO",
    }

    for ctx_key, sub in ctx_groups:
        # 按 standardized sub_name 排序 (deterministic orientation)
        sorted_sub = sub.sort_values("__sub_std__", kind="stable").reset_index(drop=False)
        # index 列保留 manifest 原始行号以便回查
        for i in range(len(sorted_sub)):
            for j in range(i + 1, len(sorted_sub)):
                row_i = sorted_sub.iloc[i]
                row_j = sorted_sub.iloc[j]
                # canonical orientation: sub_i < sub_j (already ensured by sorting)
                # i.e. row_i.__sub_std__ <= row_j.__sub_std__
                if str(row_i["__sub_std__"]) > str(row_j["__sub_std__"]):
                    # 防御性: 实际不会触发, 因为已排序
                    continue
                pair = {
                    "ring_name": ctx_key[0],
                    "ring_pos": ctx_key[1],
                    "target_ring_id": ctx_key[2],
                    "sub_type": ctx_key[3],
                    "sub_i": str(row_i["__sub_std__"]),
                    "sub_j": str(row_j["__sub_std__"]),
                    "sample_id_i": str(row_i.get("sample_id", "")),
                    "sample_id_j": str(row_j.get("sample_id", "")),
                    "canonical_smiles_i": str(row_i.get("canonical_smiles", "")),
                    "canonical_smiles_j": str(row_j.get("canonical_smiles", "")),
                }
                # ground-truth deltas
                for tk, col in descriptor_cols.items():
                    vi = pd.to_numeric(pd.Series([row_i.get(col)]), errors="coerce").iloc[0]
                    vj = pd.to_numeric(pd.Series([row_j.get(col)]), errors="coerce").iloc[0]
                    if pd.notna(vi) and pd.notna(vj):
                        delta = float(vi) - float(vj)  # ΔA_ij = A_i - A_j
                    else:
                        delta = float("nan")
                    pair[f"delta_{tk}_true"] = delta
                pair_rows.append(pair)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / PAIR_MANIFEST_CSV.name, index=False)
    if verbose:
        print(f"[pair] written {len(pair_df)} canonical pairs → {PAIR_MANIFEST_CSV.name}")

    return {
        "n_contexts": int(len(ctx_df)),
        "n_canonical_pairs": int(len(pair_df)),
        "context_audit_csv": str(out_dir / PAIR_CTX_CSV.name),
        "pair_manifest_csv": str(out_dir / PAIR_MANIFEST_CSV.name),
    }


if __name__ == "__main__":
    out = build()
    print(json.dumps(out, indent=2, ensure_ascii=False))
