"""Phase 10 (scaffold bias diagnostic): 对每个 context/scaffold 计算 bias_g 与
各误差指标, 并分析 |bias| 与 absolute MAE / ΔMAE 的关系.

约束:
  - bias_g 只用于 diagnostic, 禁止 oracle correction (即: 不允许基于 bias_g
    对预测做后处理以提升指标, 因为这等价于泄漏真值).
  - 报告:
      * absolute MAE_g, pairwise ΔMAE_g
      * residual SD after removing group bias (诊断预测分散)
      * delta Spearman, delta sign accuracy
      * |bias| vs absolute MAE; |bias| vs delta MAE (相关性)
  - 输入: Phase 4 (lunci10_absolute_predictions.csv) + Phase 7
    (lunci10_pair_predictions.csv) 产物.
  - 输出: scaffold_context_bias.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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

FIG4_ROOT = Path("_PROJ_ROOT/0901-end-code/fig4_lunci10")
ABSOLUTE_CSV = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/01_external_absolute/lunci10_absolute_predictions.csv")
PAIR_CSV = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/03_pairwise/lunci10_pair_predictions.csv")
OUT_DIR = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/04_bias")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCAFFOLD_BIAS_CSV = OUT_DIR / "scaffold_context_bias.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]


def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> Optional[float]:
    sub = np.stack([x, y], axis=1)
    sub = sub[np.isfinite(sub).all(axis=1)]
    if sub.shape[0] < 3:
        return None
    try:
        v = fn(sub[:, 0], sub[:, 1])
        return float(v.statistic) if hasattr(v, "statistic") else float(v[0])
    except Exception:
        return None


def _group_bias(df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (scaffold/context) × task 给出 bias_g, MAE_g, residual SD, etc.

    Definition:
      bias_g = mean(y_pred - y_true)  within group g
      MAE_g  = mean(|y_pred - y_true|)
      residual_after_bias = y_pred - y_true - bias_g
      residual SD_g = std(residual_after_bias)
    """
    rows: List[Dict[str, Any]] = []
    metric_keys = []
    for task in TASKS:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue
        truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
        if truth_col not in sub.columns:
            continue
        sub["__y_true__"] = pd.to_numeric(sub[truth_col], errors="coerce")
        sub = sub.dropna(subset=["__y_true__", "y_pred"])
        if sub.empty:
            continue
        # 用 (ring_name, ring_pos, target_ring_id) 构造 context 键; fallback 到 ring_name
        ctx_cols = [c for c in ("ring_name", "ring_pos", "target_ring_id", "sub_type") if c in sub.columns]
        if not ctx_cols:
            continue
        sub["__ctx__"] = sub[ctx_cols].astype(str).agg("||".join, axis=1)
        sub["__ring__"] = sub["ring_name"].astype(str) if "ring_name" in sub.columns else sub["__ctx__"]
        # per-context
        for ctx_key, g in sub.groupby("__ctx__"):
            err = g["y_pred"].astype(float).to_numpy() - g["__y_true__"].astype(float).to_numpy()
            bias = float(np.mean(err))
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            resid = err - bias
            resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
            rows.append({
                "task": task,
                "group_type": "context",
                "group_key": ctx_key,
                "ring_name": str(g["ring_name"].iloc[0]) if "ring_name" in g.columns else "",
                "n": int(len(g)),
                "bias_g": bias,
                "MAE_g": mae,
                "RMSE_g": rmse,
                "residual_sd_after_bias": resid_sd,
                "y_true_mean": float(np.mean(g["__y_true__"])),
                "y_pred_mean": float(np.mean(g["y_pred"])),
            })
        # per-ring (coarser)
        for ring_key, g in sub.groupby("__ring__"):
            err = g["y_pred"].astype(float).to_numpy() - g["__y_true__"].astype(float).to_numpy()
            bias = float(np.mean(err))
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            resid = err - bias
            resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
            rows.append({
                "task": task,
                "group_type": "ring",
                "group_key": ring_key,
                "ring_name": str(ring_key),
                "n": int(len(g)),
                "bias_g": bias,
                "MAE_g": mae,
                "RMSE_g": rmse,
                "residual_sd_after_bias": resid_sd,
                "y_true_mean": float(np.mean(g["__y_true__"])),
                "y_pred_mean": float(np.mean(g["y_pred"])),
            })
    return pd.DataFrame(rows)


def _pairwise_metrics_per_group(pair_df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (context/scaffold) 计算 ΔA_ij 的 MAE / Spearman / sign_acc."""
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        sub = pair_df[pair_df["task"] == task].copy()
        if sub.empty:
            continue
        truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
        delta_col = f"delta_{truth_col}_true" if task == "NICS_1zz" else f"delta_{task}_true"
        if delta_col not in sub.columns:
            continue
        sub = sub.dropna(subset=[delta_col, "delta_pred"])
        if sub.empty:
            continue
        ctx_cols = [c for c in ("ring_name", "ring_pos", "target_ring_id", "sub_type") if c in sub.columns]
        sub["__ctx__"] = sub[ctx_cols].astype(str).agg("||".join, axis=1)
        sub["__ring__"] = sub["ring_name"].astype(str) if "ring_name" in sub.columns else sub["__ctx__"]
        for grp_type, col in (("context", "__ctx__"), ("ring", "__ring__")):
            for key, g in sub.groupby(col):
                d_t = g[delta_col].astype(float).to_numpy()
                d_p = g["delta_pred"].astype(float).to_numpy()
                err = d_p - d_t
                mae = float(np.mean(np.abs(err)))
                sp = _safe_corr(d_t, d_p, lambda a, b: spearmanr(a, b))
                pr = _safe_corr(d_t, d_p, lambda a, b: pearsonr(a, b))
                sign_acc = float(np.mean(np.sign(d_t) == np.sign(d_p)))
                rows.append({
                    "task": task,
                    "group_type": grp_type,
                    "group_key": str(key),
                    "ring_name": (str(g["ring_name"].iloc[0])
                                  if "ring_name" in g.columns else ""),
                    "n_pairs": int(len(g)),
                    "delta_MAE_g": mae,
                    "delta_spearman_g": sp if sp is not None else float("nan"),
                    "delta_pearson_g": pr if pr is not None else float("nan"),
                    "delta_sign_acc_g": sign_acc,
                })
    return pd.DataFrame(rows)


def main(verbose: bool = True) -> Dict[str, Any]:
    if not ABSOLUTE_CSV.is_file():
        raise FileNotFoundError(f"missing absolute predictions: {ABSOLUTE_CSV}")
    abs_df = pd.read_csv(ABSOLUTE_CSV)
    pair_df = pd.read_csv(PAIR_CSV) if PAIR_CSV.is_file() else pd.DataFrame()

    # 绝对误差的 per-context & per-ring bias
    bias_df = _group_bias(abs_df)
    if pair_df.empty:
        pw_df = pd.DataFrame()
    else:
        pw_df = _pairwise_metrics_per_group(pair_df)

    # 合并: per-context 对齐 (task, group_key, group_type, ring_name)
    if not pw_df.empty:
        merged = bias_df.merge(
            pw_df,
            on=["task", "group_type", "group_key", "ring_name"],
            how="outer",
        )
    else:
        merged = bias_df.copy()
        merged["n_pairs"] = 0
        merged["delta_MAE_g"] = float("nan")
        merged["delta_spearman_g"] = float("nan")
        merged["delta_pearson_g"] = float("nan")
        merged["delta_sign_acc_g"] = float("nan")

    # 计算 |bias| 与 absolute MAE / ΔMAE 相关性
    corr_rows: List[Dict[str, Any]] = []
    for task in TASKS:
        for grp_type in ("context", "ring"):
            sub = merged[(merged["task"] == task) & (merged["group_type"] == grp_type)].copy()
            if sub.empty:
                continue
            sub["abs_bias"] = sub["bias_g"].abs()
            abs_corr = _safe_corr(
                sub["abs_bias"].to_numpy(dtype=float),
                sub["MAE_g"].to_numpy(dtype=float),
                lambda a, b: spearmanr(a, b),
            )
            pw_corr = _safe_corr(
                sub["abs_bias"].to_numpy(dtype=float),
                sub["delta_MAE_g"].to_numpy(dtype=float),
                lambda a, b: spearmanr(a, b),
            )
            corr_rows.append({
                "task": task,
                "group_type": grp_type,
                "n_groups": int(len(sub)),
                "spearman_abs_bias_vs_MAE_g": abs_corr if abs_corr is not None else float("nan"),
                "spearman_abs_bias_vs_delta_MAE_g": pw_corr if pw_corr is not None else float("nan"),
                "note": "diagnostic only; bias_g not used for oracle correction",
            })

    merged.to_csv(SCAFFOLD_BIAS_CSV, index=False)
    corr_csv = OUT_DIR / "bias_vs_error_correlation.csv"
    pd.DataFrame(corr_rows).to_csv(corr_csv, index=False)

    if verbose:
        print(f"[scaffold_bias] saved {SCAFFOLD_BIAS_CSV} (rows={len(merged)})")
        print(f"[scaffold_bias] saved {corr_csv}")
        if corr_rows:
            for r in corr_rows:
                print(f"  [{r['task']}/{r['group_type']}] "
                      f"ρ(|bias|, MAE)={r['spearman_abs_bias_vs_MAE_g']:.3f} "
                      f"ρ(|bias|, ΔMAE)={r['spearman_abs_bias_vs_delta_MAE_g']:.3f}")

    return {
        "scaffold_bias_csv": str(SCAFFOLD_BIAS_CSV),
        "correlation_csv": str(corr_csv),
        "n_groups": int(len(merged)),
        "n_correlation_rows": int(len(corr_rows)),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))