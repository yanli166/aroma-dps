"""Phase 12 (position analysis): 在相同 scaffold + 相同 substituent 条件下比较
ΔA_ortho, ΔA_meta, ΔA_para.

约束:
  - 禁止混合不同 scaffold 后声称 "para stronger than meta"; 必须 within-(scaffold,
    substituent) 比较.
  - 检查 pairing 时 C-linked 与 N-linked substituent 是否应直接比较 — 用
    sub_linked_atom 字段 (C/N) 分别报告 C-only / N-only 性能.
  - 输出 position_effect_summary.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = "_PROJ_ROOT"
CODE_END = f"{PROJ_ROOT}/code_end"
for p in (PROJ_ROOT, CODE_END):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from layer4_substituent.code.hammett_constants import identify_substituents  # type: ignore

FIG4_ROOT = Path("_PROJ_ROOT/0901-end-code/fig4_lunci10")
MANIFEST_PATH = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/00_audit/lunci10_manifest.csv")
ABSOLUTE_CSV = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/01_external_absolute/lunci10_absolute_predictions.csv")

OUT_DIR = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/06_position")
OUT_DIR.mkdir(parents=True, exist_ok=True)
POSITION_CSV = OUT_DIR / "position_effect_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]


def _linked_atom(smi: str, sub_position: str) -> str:
    """判断取代基是否通过 C 或 N 连接到环 (C-linked vs N-linked)."""
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return "unknown"
    try:
        subs = identify_substituents(mol)
    except Exception:
        return "unknown"
    if not subs:
        return "unknown"
    s = subs[0]
    ring_atom = mol.GetAtomWithIdx(s["ring_atom_idx"])
    # 取代基第一个原子 (连接环的那个)
    sub_first = mol.GetAtomWithIdx(s["atom_idx"])
    return "N-linked" if str(sub_first.GetSymbol()) == "N" else (
        "C-linked" if str(sub_first.GetSymbol()) == "C" else "other"
    )


def _per_molecule_position(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        smi = str(row.get("canonical_smiles", "") or "")
        sub_name = str(row.get("sub_name", "") or "").strip()
        ring_pos = str(row.get("ring_pos", "") or "")
        from rdkit import Chem, RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi) if smi else None
        pos = ""
        if mol is not None:
            try:
                subs = identify_substituents(mol)
                if subs:
                    pos = subs[0]["position"]
            except Exception:
                pos = ""
        if not pos:
            pos = ring_pos if ring_pos else "other"
        linked = _linked_atom(smi, pos)
        rows.append({
            "sample_id": row.get("sample_id", ""),
            "canonical_smiles": smi,
            "sub_name": sub_name,
            "ring_name": row.get("ring_name", ""),
            "ring_pos": ring_pos,
            "position": pos,
            "linked_atom": linked,
        })
    return pd.DataFrame(rows)


def _per_position_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """对每个 (task, ring_name, sub_name, position, linked_atom) 给 ΔA 与误差."""
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        true_col = "NICS_ZZ" if task == "NICS_1zz" else task
        if true_col not in merged.columns:
            continue
        sub_df = merged.dropna(subset=[true_col, task]).copy()
        if sub_df.empty:
            continue
        for (ring_name, sub_name, position, linked), g in sub_df.groupby(
            ["ring_name", "sub_name", "position", "linked_atom"]
        ):
            if len(g) < 2:
                continue
            y_t = pd.to_numeric(g[true_col], errors="coerce").to_numpy(dtype=float)
            y_p = pd.to_numeric(g[task], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y_t) & np.isfinite(y_p)
            if valid.sum() < 2:
                continue
            y_t, y_p = y_t[valid], y_p[valid]
            err = y_p - y_t
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            sp_rho = float(stats.spearmanr(y_t, y_p).statistic) if len(y_t) >= 3 else float("nan")
            pr_r = float(stats.pearsonr(y_t, y_p).statistic) if len(y_t) >= 3 else float("nan")
            rows.append({
                "task": task,
                "ring_name": ring_name,
                "sub_name": sub_name,
                "position": position,
                "linked_atom": linked,
                "n_molecules": int(len(g)),
                "MAE": mae,
                "RMSE": rmse,
                "spearman_rho": sp_rho,
                "pearson_r": pr_r,
                "y_true_mean": float(np.mean(y_t)),
                "y_pred_mean": float(np.mean(y_p)),
                "bias": float(np.mean(err)),
            })
    return pd.DataFrame(rows)


def _pairwise_position_compare(merged: pd.DataFrame) -> pd.DataFrame:
    """同 (ring_name, sub_name) 内的 (meta vs para) / (ortho vs meta) / (ortho vs para)
    直接 Δ 比较; 禁止跨 scaffold 比较.

    输出每个 (task, ring_name, sub_name, position_a, position_b, linked_atom)
    的 ΔA_diff = mean A(pos_a) - mean A(pos_b).
    """
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        true_col = "NICS_ZZ" if task == "NICS_1zz" else task
        if true_col not in merged.columns:
            continue
        for (ring_name, sub_name, linked), g in merged.groupby(
            ["ring_name", "sub_name", "linked_atom"]
        ):
            # 对每个 position 算均值
            means = g.groupby("position")[true_col].agg(["mean", "count"]).reset_index()
            if means.empty:
                continue
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    a = means.iloc[i]
                    b = means.iloc[j]
                    if a["count"] < 1 or b["count"] < 1:
                        continue
                    rows.append({
                        "task": task,
                        "ring_name": ring_name,
                        "sub_name": sub_name,
                        "linked_atom": linked,
                        "position_a": a["position"],
                        "position_b": b["position"],
                        "delta_A_a_minus_b": float(a["mean"] - b["mean"]),
                        "n_a": int(a["count"]),
                        "n_b": int(b["count"]),
                    })
    return pd.DataFrame(rows)


def main(verbose: bool = True) -> Dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    absolute = pd.read_csv(ABSOLUTE_CSV) if ABSOLUTE_CSV.is_file() else pd.DataFrame()
    pos_df = _per_molecule_position(manifest)
    # merge predictions
    if not absolute.empty:
        main_pred = absolute[absolute["model"] == "RC_MPNN"]
        if main_pred.empty:
            main_pred = absolute
        wide = main_pred.pivot_table(
            index="sample_id", columns="task", values="y_pred", aggfunc="mean"
        ).reset_index()
        true_wide = main_pred.pivot_table(
            index="sample_id", columns="task", values="y_true", aggfunc="mean"
        ).reset_index().rename(columns={"HOMA": "HOMA_true",
                                       "NICS_1zz": "NICS_1zz_true",
                                       "MBCO": "MBCO_true",
                                       "NICS_iso": "NICS_iso_true"})
        merged = pos_df.merge(wide, on="sample_id", how="left")
        merged = merged.merge(true_wide, on="sample_id", how="left", suffixes=("", "_true_df"))
        if "NICS_ZZ" in manifest.columns:
            zz = manifest[["sample_id", "NICS_ZZ"]].rename(columns={"NICS_ZZ": "NICS_1zz_true"})
            merged = merged.drop(columns=["NICS_1zz_true"], errors="ignore").merge(zz, on="sample_id", how="left")
    else:
        merged = pos_df.copy()
        for t in TASKS:
            merged[t] = float("nan")
            merged[f"{t}_true"] = float("nan")

    per_pos_metrics = _per_position_metrics(merged)
    pairwise = _pairwise_position_compare(merged)

    # 单独 C-only / N-only 性能
    c_only = per_pos_metrics[per_pos_metrics["linked_atom"] == "C-linked"]
    n_only = per_pos_metrics[per_pos_metrics["linked_atom"] == "N-linked"]

    # 写入主表 (per-position metrics + linked_atom 标注)
    per_pos_metrics.to_csv(POSITION_CSV, index=False)

    # 额外: linked_atom summary
    linked_summary = (
        per_pos_metrics.groupby(["task", "linked_atom"])["MAE"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={"count": "n_groups", "mean": "MAE_mean", "std": "MAE_std"})
    )
    linked_csv = OUT_DIR / "linked_atom_summary.csv"
    linked_summary.to_csv(linked_csv, index=False)

    # 额外: 跨 position pairwise
    pairwise_csv = OUT_DIR / "position_pairwise_compare.csv"
    pairwise.to_csv(pairwise_csv, index=False)

    if verbose:
        print(f"[position] saved {POSITION_CSV} (rows={len(per_pos_metrics)})")
        print(f"[position] saved {linked_csv}")
        print(f"[position] saved {pairwise_csv}")
        # 防止 Simpson paradox 提示
        print("[position] IMPORTANT: position-pairwise comparisons are within "
              "(ring_name, sub_name) only — DO NOT average across scaffolds.")

    return {
        "position_csv": str(POSITION_CSV),
        "linked_summary_csv": str(linked_csv),
        "pairwise_compare_csv": str(pairwise_csv),
        "n_groups_total": int(len(per_pos_metrics)),
        "n_C_only": int(len(c_only)),
        "n_N_only": int(len(n_only)),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))