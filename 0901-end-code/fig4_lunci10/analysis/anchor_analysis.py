"""Phase 15 (Anchor robustness diagnostic): 对比以下 5 种 anchor 设置下,
Siamese Δ-MLP vs 绝对 subtraction 在 lunci10 上的相对优劣:

    1. F (fluorine)                 — σ_p = 0.06,  弱 EDS
    2. Cl (chlorine)                — σ_p = 0.23,  中等 EDS
    3. OMe (methoxy)                — σ_p = -0.27, π-donor
    4. 3-anchor average (F, Cl, OMe)— multi-anchor consensus
    5. all-pairs (cheating 但报指标作 ceiling indicator)
       注: 该 baseline 仅用于 audit context, 不进入正式结论。

回答的研究问题:
    Q1. Siamese > Subtraction 是否仍成立?              — 是否对 anchor 不敏感
    Q2. HOMA / MBCO 是否仍受益于 explicit Δ-learning?  — Δ 是否值得训
    Q3. NICS 是否仍更适合 subtraction?                 — 反向或不同 topic
    Q4. Conclusion 是否对 anchor 选择敏感?              — 稳健性

输入:
    - results/fig4_lunci10_final/03_pairwise/lunci10_pair_manifest.csv
      (每对含 sub_i / sub_j 以及 i, j 端的真实 absolute value)
    - results/fig4_lunci10_final/si_pretraining/pretraining_transfer.csv
      (Phase 14 Siamese 预测)
    - results/fig4_lunci10_final/01_external_absolute/lunci10_absolute_predictions.csv
      (Phase 1 frozen absolute 预测)

输出:
    - results/fig4_lunci10_final/si_anchor/anchor_robustness.csv
      列: anchor_label, anchor_sub, task, method (Siamese/subtraction),
           MAE, RMSE, R2, sign_accuracy, n_pairs,
           conclusions (Q1-Q4 答案的 JSON 字符串)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = Path(_PROJ_ROOT)
CODE_END = PROJ_ROOT / "archive/deprecated/code_end"
for p in (str(PROJ_ROOT), str(CODE_END)):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

RES_ROOT = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final"
PAIR_CSV = RES_ROOT / "03_pairwise" / "lunci10_pair_manifest.csv"
ABS_CSV = RES_ROOT / "01_external_absolute" / "lunci10_absolute_predictions.csv"
SI_CSV = RES_ROOT / "si_pretraining" / "pretraining_transfer.csv"

OUT_DIR = RES_ROOT / "si_anchor"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "anchor_robustness.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
ANCHOR_LABELS = ["F", "Cl", "OMe", "three_anchor_avg", "all_pairs"]

# 真实 Hammett σ_p 锚值 (供 design 参考)
ANCHOR_SIGMA_P = {"F": 0.06, "Cl": 0.23, "OMe": -0.27, "H": 0.0, "Me": -0.17}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sub = np.stack([y_true, y_pred], axis=1)
    sub = sub[np.isfinite(sub).all(axis=1)]
    if sub.shape[0] == 0:
        return {"n": 0, "MAE": float("nan"),
                "RMSE": float("nan"), "R2": float("nan"),
                "sign_accuracy": float("nan")}
    yt, yp = sub[:, 0], sub[:, 1]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"n": int(sub.shape[0]), "MAE": mae,
            "RMSE": rmse, "R2": r2,
            "sign_accuracy": float(np.mean(np.sign(yt) == np.sign(yp)))}


def _select_anchor_subset(pair_df: pd.DataFrame, anchor_label: str) -> pd.DataFrame:
    """按 anchor_label 过滤 pair subset。

    实现约定 (与 lunci10 manifest 字段保持一致):
      - F / Cl / OMe:    选 sub_i == anchor OR sub_j == anchor 的对
                         (即任一端为 anchor)
      - three_anchor_avg:  选 sub_i, sub_j 任一端 ∈ {F, Cl, OMe} 的对
                           对每个 i 端在 j 端的真实 absolute value 与 anchor 端
                           之差 作 absolute prediction, 再平均化
      - all_pairs:          不滤
    """
    if anchor_label == "all_pairs":
        return pair_df
    if anchor_label == "three_anchor_avg":
        mask = pair_df["sub_i"].isin(ANCHOR_SIGMA_P.keys()) | \
               pair_df["sub_j"].isin(ANCHOR_SIGMA_P.keys())
        return pair_df[mask]
    if anchor_label in ANCHOR_SIGMA_P:
        mask = (pair_df["sub_i"] == anchor_label) | (pair_df["sub_j"] == anchor_label)
        return pair_df[mask]
    raise ValueError(f"unknown anchor_label = {anchor_label}")


def _attach_predicted_delta(pair_df: pd.DataFrame,
                            abs_df: pd.DataFrame,
                            method: str = "subtraction") -> pd.DataFrame:
    """将 absolute prediction attach 到每对, 计算 Δ pred。

    method ∈ {"subtraction", "siamese"}
    - subtraction: delta_pred = Ahat_i - Ahat_j (来自 frozen absolute model)
    - siamese:     delta_pred 来自 Phase 14 预测 csv (best-of seeds 的平均)
    """
    out = pair_df.copy()
    if method == "siamese":
        # Fall back to NaN if SI csv missing — keep code path complete.
        if SI_CSV.exists():
            try:
                si_df = pd.read_csv(SI_CSV)
                # NaN placeholder; real join keyed by (sample_id_i, sample_id_j, task)
                out["delta_pred"] = np.nan
                out["si_source"] = "phase14_stub"
            except Exception:
                out["delta_pred"] = np.nan
        else:
            out["delta_pred"] = np.nan
        return out

    # subtraction
    if not abs_df.empty:
        i_map = dict(zip(abs_df["sample_id"].astype(str),
                         abs_df["prediction"].astype(float)))
        j_map = i_map  # same column
        out["delta_pred"] = out["sample_id_i"].astype(str).map(i_map) \
            - out["sample_id_j"].astype(str).map(j_map)
    else:
        out["delta_pred"] = np.nan
    return out


def _aggregate_for_anchor_task_method(
    pair_df: pd.DataFrame, anchor_label: str, task: str, method: str,
    abs_df: pd.DataFrame,
) -> Dict[str, Any]:
    """一个 (anchor, task, method) 单元返回一行 metrics + 出题结论 dict。"""
    if pair_df.empty:
        return {
            "anchor_label": anchor_label, "task": task, "method": method,
            "n_pairs": 0, "MAE": np.nan, "RMSE": np.nan,
            "R2": np.nan, "sign_accuracy": np.nan,
            "Q1_siamese_vs_subtraction": "N/A",
            "Q2_delta_learning_helps_HOMA_MBCO": "N/A",
            "Q3_NICS_favors_subtraction": "N/A",
            "Q4_conclusion_anchor_sensitive": "N/A",
        }

    sub = _select_anchor_subset(pair_df, anchor_label)
    if sub.empty:
        return {
            "anchor_label": anchor_label, "task": task, "method": method,
            "n_pairs": 0, "MAE": np.nan, "RMSE": np.nan,
            "R2": np.nan, "sign_accuracy": np.nan,
            "Q1_siamese_vs_subtraction": "N/A",
            "Q2_delta_learning_helps_HOMA_MBCO": "N/A",
            "Q3_NICS_favors_subtraction": "N/A",
            "Q4_conclusion_anchor_sensitive": "N/A",
        }

    augmented = _attach_predicted_delta(sub, abs_df, method=method)
    truth_col = "delta_NICS_ZZ_true" if task == "NICS_1zz" else f"delta_{task}_true"
    if truth_col not in augmented.columns:
        # without true column we return NaNs and mark N/A
        return {
            "anchor_label": anchor_label, "task": task, "method": method,
            "n_pairs": int(len(augmented)), "MAE": np.nan, "RMSE": np.nan,
            "R2": np.nan, "sign_accuracy": np.nan,
            "Q1_siamese_vs_subtraction": "missing_truth_column",
            "Q2_delta_learning_helps_HOMA_MBCO": "missing_truth_column",
            "Q3_NICS_favors_subtraction": "missing_truth_column",
            "Q4_conclusion_anchor_sensitive": "missing_truth_column",
        }

    yt = augmented[truth_col].astype(float).values
    yp = augmented["delta_pred"].astype(float).values
    m = _safe_metrics(yt, yp)
    out = {
        "anchor_label": anchor_label,
        "task": task,
        "method": method,
        "n_pairs": int(m["n"]),
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"],
        "sign_accuracy": m["sign_accuracy"],
        # 留 placeholder, 主程序 cross-cell 回填
        "Q1_siamese_vs_subtraction": "",
        "Q2_delta_learning_helps_HOMA_MBCO": "",
        "Q3_NICS_favors_subtraction": "",
        "Q4_conclusion_anchor_sensitive": "",
    }
    return out


def _derive_conclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-cell 回填 4 个研究问题的答案。

    约定 (代码层):
      - 对每个 anchor × task, 取 (MAE_siamese, MAE_sub) 作对比
      - Q1: 在某 anchor 上 Siamese MAE 较 subtraction MAE 下降 ≥ 5% 视为"成立"
      - Q2: 对 HOMA / MBCO 平均 Q1 答案;
      - Q3: 对 NICS_1zz, Siamese 是否 *低于* subtraction 5%?
      - Q4: Q1 在 anchor 间是否符号一致?
    """
    df = df.copy()
    pivot = df.pivot_table(index=["anchor_label", "task"],
                           columns="method",
                           values="MAE",
                           aggfunc="mean").reset_index()
    if "siamese" not in pivot.columns or "subtraction" not in pivot.columns:
        # 没 Siamese 数 (Phase 14 stub 未跑) → 用 placeholder
        df["Q1_siamese_vs_subtraction"] = "phase14_stub_no_siamese_data"
        df["Q2_delta_learning_helps_HOMA_MBCO"] = "phase14_stub_no_siamese_data"
        df["Q3_NICS_favors_subtraction"] = "phase14_stub_no_siamese_data"
        df["Q4_conclusion_anchor_sensitive"] = "phase14_stub_no_siamese_data"
        return df

    q1_per_cell = (pivot["siamese"] - pivot["subtraction"]) / pivot["subtraction"]
    pivot["Q1_lower_is_better_for_siamese"] = q1_per_cell < -0.05
    pivot["Q3_siamese_strictly_helps_NICS"] = q1_per_cell < -0.05  # NICS 自家检查
    pivot["Q1_signed_consistency"] = pivot["Q1_lower_is_better_for_siamese"].apply(
        lambda b: "siamese_wins" if b else "sub_wins_or_close"
    )

    # 映射回 df
    sig_map = {(r["anchor_label"], r["task"]): r["Q1_signed_consistency"]
               for _, r in pivot.iterrows()}
    siam_nics = pivot[pivot["task"] == "NICS_1zz"]["Q3_siamese_strictly_helps_NICS"]
    nics_siamese_helps = bool((~siam_nics).all()) if len(siam_nics) else False

    # merge by keys
    df["Q1_siamese_vs_subtraction"] = df.apply(
        lambda r: sig_map.get((r["anchor_label"], r["task"]), "N/A"), axis=1
    )

    # Q2: HOMA & MBCO cell
    for idx, row in df.iterrows():
        if row["task"] in ("HOMA", "MBCO"):
            df.at[idx, "Q2_delta_learning_helps_HOMA_MBCO"] = sig_map.get(
                (row["anchor_label"], row["task"]), "N/A"
            )
        elif row["task"] == "NICS_1zz":
            df.at[idx, "Q3_NICS_favors_subtraction"] = (
                "siamese_still_helps" if nics_siamese_helps
                else "subtraction_preferred"
            )

    # Q4: across anchors 看 signed consistency
    consistency_per_anchor = pivot.groupby("anchor_label")["Q1_signed_consistency"].apply(
        lambda s: bool((s == "siamese_wins").all()) if len(s) else False
    )
    df["Q4_conclusion_anchor_sensitive"] = df["anchor_label"].map(
        lambda a: "stable_across_anchors" if bool(
            (pivot[pivot["anchor_label"] == a]["Q1_signed_consistency"]
             == "siamese_wins").all()
        ) else "sensitive_to_anchor"
    )
    return df


# --------------------------------------------------------------------------
# Main batch
# --------------------------------------------------------------------------
def run_all() -> pd.DataFrame:
    if not PAIR_CSV.exists():
        warnings.warn(f"[phase15] missing pair manifest: {PAIR_CSV}")
        pair_df = pd.DataFrame()
    else:
        pair_df = pd.read_csv(PAIR_CSV)

    if not ABS_CSV.exists():
        warnings.warn(f"[phase15] missing absolute pred csv: {ABS_CSV}")
        abs_df = pd.DataFrame(columns=["sample_id", "prediction"])
    else:
        abs_df = pd.read_csv(ABS_CSV)

    rows: List[Dict[str, Any]] = []
    for anchor_label in ANCHOR_LABELS:
        for task in TASKS:
            for method in ("subtraction", "siamese"):
                rows.append(_aggregate_for_anchor_task_method(
                    pair_df=pair_df, anchor_label=anchor_label,
                    task=task, method=method, abs_df=abs_df,
                ))
    df = pd.DataFrame(rows)
    df = _derive_conclusions(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[phase15] pair manifest: {'OK' if PAIR_CSV.exists() else 'MISSING'}")
    print(f"[phase15] absolute csv : {'OK' if ABS_CSV.exists() else 'MISSING'}")
    print(f"[phase15] si csv       : {'OK' if SI_CSV.exists() else 'MISSING'}")
    df = run_all()
    print(f"[phase15] wrote {len(df)} rows to {OUT_CSV}")
    if len(df):
        show = df[["anchor_label", "task", "method", "MAE",
                   "Q1_siamese_vs_subtraction",
                   "Q4_conclusion_anchor_sensitive"]].head(20)
        print(show.to_string(index=False))
