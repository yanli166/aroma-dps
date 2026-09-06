"""Phase 11 (Hammett analysis): 对 lunci10 分子确定 substituent identity / σm /
σp / ortho/meta/para position, 并分析 ΔHOMA / ΔNICS / ΔMBCO vs σm / σp.

约束:
  - 使用现有 Hansch 1991 lookup (hammett_constants.HAMMETT_TABLE) — 禁止根据
    lunci10 相关性反推修改 σ 常数.
  - ortho 单独分析, 不使用 σm/σp 代替 (ortho steric/hyperconjugation effects
    与 Hammett equation 标准框架不一致).
  - Anchor protocol: 固定 anchors {F, Cl, OMe}; 统计每个 anchor 在 lunci10 中的
    coverage (ΔA_sub vs ΔA_anchor), 输出 anchor_coverage.csv.
  - Hammett 分析: pooled + within-scaffold/context (避免 Simpson's paradox).
  - 报告: Spearman ρ (优先) / Pearson r / slope / N / 95% CI.

产物:
  - anchor_coverage.csv
  - hammett_matches.csv
  - hammett_context_summary.csv
  - hammett_global_summary.csv
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

# 必须 import 原项目 Hammett constants
from layer4_substituent.code.hammett_constants import (  # type: ignore
    HAMMETT_SIGMA_META, HAMMETT_SIGMA_PARA,
    identify_substituents, HAMMETT_TABLE,
)

FIG4_ROOT = Path("_PROJ_ROOT/0901-end-code/fig4_lunci10")
MANIFEST_PATH = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/00_audit/lunci10_manifest.csv")
ABSOLUTE_CSV = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/01_external_absolute/lunci10_absolute_predictions.csv")

OUT_DIR = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/05_hammett")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_COV_CSV = OUT_DIR / "anchor_coverage.csv"
MATCH_CSV = OUT_DIR / "hammett_matches.csv"
CTX_SUMMARY_CSV = OUT_DIR / "hammett_context_summary.csv"
GLOBAL_SUMMARY_CSV = OUT_DIR / "hammett_global_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
ANCHORS = ["F", "Cl", "OMe"]  # OMe 即 OCH3 in table
ANCHOR_ALIASES = {"OMe": "OCH3"}


# --------------------------------------------------------------------------
# Lookup helpers (使用 Hansch 1991 表; 禁止修改)
# --------------------------------------------------------------------------
def _lookup_sigma(sub_name: str) -> Dict[str, float]:
    """根据 sub_name 在 HAMMETT_TABLE 中查 (σm, σp); 找不到返回 0/0."""
    name = ANCHOR_ALIASES.get(sub_name, sub_name)
    for smarts, (sm, sp, alias) in HAMMETT_TABLE.items():
        if alias == name or alias == sub_name:
            return {"sigma_m": float(sm), "sigma_p": float(sp),
                    "substituent": alias, "smarts": smarts}
    return {"sigma_m": 0.0, "sigma_p": 0.0,
            "substituent": sub_name, "smarts": ""}


def _normalize_anchor(a: str) -> str:
    return ANCHOR_ALIASES.get(a, a)


# --------------------------------------------------------------------------
# 工具: 给定 lunci10 manifest + absolute predictions, 生成 per-molecule Hammett
# 信息 (substituent identity, σ, position), 然后构造 ΔA_sub vs anchor.
# --------------------------------------------------------------------------
def _per_molecule_hammett(manifest: pd.DataFrame, absolute: pd.DataFrame) -> pd.DataFrame:
    """为每个 lunci10 分子记录: ring_name, sub_name, σm, σp, position (o/m/p),
    true value, predicted value (取主模型 RC_MPNN).

    注意: lunci10 是 2-取代萘 (位置 1=α, 2=β), 通过 sub_name 与 position 字段来
    判断 ortho/meta/para — 这里利用 begin.csv 的 ring_pos 列.
    """
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")

    # 合并 prediction
    if not absolute.empty:
        # 取 RC_MPNN 主模型
        main = absolute[absolute["model"] == "RC_MPNN"].copy()
        if main.empty:
            main = absolute.copy()
        pred_wide = main.pivot_table(
            index="sample_id", columns="task", values="y_pred", aggfunc="mean"
        ).reset_index()
        true_wide = main.pivot_table(
            index="sample_id", columns="task", values="y_true", aggfunc="mean"
        ).reset_index().rename(columns={"HOMA": "HOMA_true",
                                       "NICS_1zz": "NICS_1zz_true",
                                       "MBCO": "MBCO_true",
                                       "NICS_iso": "NICS_iso_true"})
    else:
        pred_wide = pd.DataFrame(columns=["sample_id"])
        true_wide = pd.DataFrame(columns=["sample_id"])

    df = manifest.merge(pred_wide, on="sample_id", how="left")
    df = df.merge(true_wide, on="sample_id", how="left", suffixes=("", "_true_df"))
    # 整理 NICS true column
    if "NICS_ZZ" in df.columns:
        df["NICS_1zz_true"] = df["NICS_ZZ"]

    # 决定每个分子的 position (ortho/meta/para): 优先 ring_pos 字段;
    # 否则按 identify_substituents 的 position 字段; 否则 fallback.
    positions: List[str] = []
    sigmas_m: List[float] = []
    sigmas_p: List[float] = []
    sub_ids: List[str] = []
    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles", "") or "")
        sub_name = str(row.get("sub_name", "") or "").strip()
        ring_pos = str(row.get("ring_pos", "") or "")
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
        # σ lookup
        sigma = _lookup_sigma(sub_name)
        positions.append(pos)
        sigmas_m.append(sigma["sigma_m"])
        sigmas_p.append(sigma["sigma_p"])
        sub_ids.append(sigma["substituent"])
    df["position"] = positions
    df["sigma_m"] = sigmas_m
    df["sigma_p"] = sigmas_p
    df["sub_id"] = sub_ids
    return df


# --------------------------------------------------------------------------
# Anchor coverage
# --------------------------------------------------------------------------
def _anchor_coverage(per_mol: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for anchor in ANCHORS:
        anchor_norm = _normalize_anchor(anchor)
        anchors = per_mol[per_mol["sub_id"] == anchor_norm]
        n_anchor = int(len(anchors))
        # coverage: (sub, anchor) pair 可行的 sub 数 (同 ring_name + 同 position)
        sub_in_scope = per_mol[per_mol["sub_id"] != anchor_norm]
        pair_counts = sub_in_scope.groupby(["ring_name", "position", "sub_id"]).size().reset_index(name="n")
        # 过滤: 同一 (ring_name, position) 必须至少有一个 anchor
        eligible = pair_counts.merge(
            anchors[["ring_name", "position"]].drop_duplicates().assign(has_anchor=True),
            on=["ring_name", "position"], how="inner",
        )
        n_eligible_sub = int(eligible["sub_id"].nunique())
        n_eligible_pair = int(len(eligible))
        rows.append({
            "anchor": anchor,
            "anchor_substituent_id": anchor_norm,
            "n_anchor_molecules": n_anchor,
            "n_ring_position_covered": int(anchors[["ring_name", "position"]].drop_duplicates().shape[0]),
            "n_eligible_substituents_for_subtraction": n_eligible_sub,
            "n_eligible_sub_anchor_pairs": n_eligible_pair,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Hammett matches (ΔA_sub - ΔA_anchor vs σm/σp)
# --------------------------------------------------------------------------
def _hammett_matches(per_mol: pd.DataFrame) -> pd.DataFrame:
    """对每个 (ring_name, position, anchor) 构造 ΔA_sub - ΔA_anchor.

    ΔA_sub   = A(sub) - A(anchor), true
    ΔAhat_sub = Ahat(sub) - Ahat(anchor), predicted
    """
    rows: List[Dict[str, Any]] = []
    for anchor in ANCHORS:
        anchor_norm = _normalize_anchor(anchor)
        anchors = per_mol[per_mol["sub_id"] == anchor_norm]
        if anchors.empty:
            continue
        for task in TASKS:
            true_col = f"{task}_true" if f"{task}_true" in per_mol.columns else (
                "NICS_ZZ" if task == "NICS_1zz" and "NICS_ZZ" in per_mol.columns else None
            )
            pred_col = task if task in per_mol.columns else None
            if not true_col or not pred_col:
                continue
            for (ring_name, position), sub_anchors in anchors.groupby(["ring_name", "position"]):
                # sub in same ring+position
                subs = per_mol[(per_mol["ring_name"] == ring_name)
                                & (per_mol["position"] == position)
                                & (per_mol["sub_id"] != anchor_norm)]
                if subs.empty:
                    continue
                # 每对 (sub, anchor) 一次记录
                for _, arow in sub_anchors.iterrows():
                    for _, srow in subs.iterrows():
                        y_t_a = pd.to_numeric(pd.Series([arow.get(true_col)]),
                                                errors="coerce").iloc[0]
                        y_t_s = pd.to_numeric(pd.Series([srow.get(true_col)]),
                                                errors="coerce").iloc[0]
                        y_p_a = pd.to_numeric(pd.Series([arow.get(pred_col)]),
                                                errors="coerce").iloc[0]
                        y_p_s = pd.to_numeric(pd.Series([srow.get(pred_col)]),
                                                errors="coerce").iloc[0]
                        if not (np.isfinite(y_t_a) and np.isfinite(y_t_s)
                                and np.isfinite(y_p_a) and np.isfinite(y_p_s)):
                            continue
                        rows.append({
                            "task": task,
                            "ring_name": ring_name,
                            "position": position,
                            "anchor": anchor,
                            "anchor_id": anchor_norm,
                            "substituent": srow["sub_id"],
                            "sigma_m": float(srow["sigma_m"]),
                            "sigma_p": float(srow["sigma_p"]),
                            "anchor_sigma_m": float(arow["sigma_m"]),
                            "anchor_sigma_p": float(arow["sigma_p"]),
                            "delta_sigma_m": float(srow["sigma_m"] - arow["sigma_m"]),
                            "delta_sigma_p": float(srow["sigma_p"] - arow["sigma_p"]),
                            "delta_A_true": float(y_t_s - y_t_a),
                            "delta_A_pred": float(y_p_s - y_p_a),
                            "n_pair": 1,
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 对每个 (task, ring_name, position, anchor, substituent) 聚合 (可能有多个
    # anchor 分子 — 取 mean).
    grouped = df.groupby(["task", "ring_name", "position", "anchor",
                           "anchor_id", "substituent"], as_index=False).agg(
        sigma_m=("sigma_m", "mean"),
        sigma_p=("sigma_p", "mean"),
        anchor_sigma_m=("anchor_sigma_m", "mean"),
        anchor_sigma_p=("anchor_sigma_p", "mean"),
        delta_sigma_m=("delta_sigma_m", "mean"),
        delta_sigma_p=("delta_sigma_p", "mean"),
        delta_A_true=("delta_A_true", "mean"),
        delta_A_pred=("delta_A_pred", "mean"),
        n_pair=("n_pair", "sum"),
    )
    return grouped


# --------------------------------------------------------------------------
# Correlation helper (Spearman preferred; also Pearson)
# --------------------------------------------------------------------------
def _correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    sub = np.stack([x, y], axis=1)
    sub = sub[np.isfinite(sub).all(axis=1)]
    n = int(sub.shape[0])
    if n < 3:
        return {"n": n, "spearman_rho": float("nan"),
                "pearson_r": float("nan"),
                "slope": float("nan"), "intercept": float("nan"),
                "spearman_ci_low": float("nan"), "spearman_ci_high": float("nan")}
    sp_rho, sp_p = stats.spearmanr(sub[:, 0], sub[:, 1])
    pr_r, pr_p = stats.pearsonr(sub[:, 0], sub[:, 1])
    slope, intercept, _, _, _ = stats.linregress(sub[:, 0], sub[:, 1])
    # bootstrap CI for Spearman (95%)
    rng = np.random.RandomState(42)
    boots: List[float] = []
    for _ in range(200):
        idx = rng.randint(0, n, n)
        try:
            r, _ = stats.spearmanr(sub[idx, 0], sub[idx, 1])
            if np.isfinite(r):
                boots.append(r)
        except Exception:
            continue
    ci_low = float(np.quantile(boots, 0.025)) if boots else float("nan")
    ci_high = float(np.quantile(boots, 0.975)) if boots else float("nan")
    return {
        "n": n,
        "spearman_rho": float(sp_rho),
        "spearman_p": float(sp_p),
        "pearson_r": float(pr_r),
        "pearson_p": float(pr_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
    }


# --------------------------------------------------------------------------
# Summary: pooled + per-(ring_name, position) context
# --------------------------------------------------------------------------
def _hammett_summary(matches: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind: 'global' (pooled across contexts) or 'context' (per ring_name)."""
    rows: List[Dict[str, Any]] = []
    if matches.empty:
        return pd.DataFrame()
    if kind == "global":
        groups = [("global", matches)]
    else:
        groups = list(matches.groupby(["task", "ring_name", "position"]))
    if kind == "global":
        for task, sub in matches.groupby("task"):
            for value_kind in ("true", "pred"):
                val = f"delta_A_{value_kind}"
                # 排除 ortho (单独分析)
                sub_no_ortho = sub[sub["position"] != "ortho"]
                for sigma_kind in ("sigma_m", "sigma_p", "delta_sigma_m", "delta_sigma_p"):
                    cor = _correlations(
                        sub_no_ortho[sigma_kind].to_numpy(dtype=float),
                        sub_no_ortho[val].to_numpy(dtype=float),
                    )
                    rows.append({
                        "scope": "pooled_no_ortho",
                        "task": task,
                        "anchor": "ALL",
                        "ring_name": "ALL",
                        "position": "meta+para",
                        "value_kind": value_kind,
                        "sigma_kind": sigma_kind,
                        **cor,
                    })
                # ortho 单独分析 (不与 σm/σp 直接配对)
                sub_ortho = sub[sub["position"] == "ortho"]
                if not sub_ortho.empty:
                    for value_kind in ("true", "pred"):
                        cor_ortho = _correlations(
                            np.zeros(len(sub_ortho)),  # dummy
                            sub_ortho[f"delta_A_{value_kind}"].to_numpy(dtype=float),
                        )
                        rows.append({
                            "scope": "ortho_only",
                            "task": task,
                            "anchor": "ALL",
                            "ring_name": "ALL",
                            "position": "ortho",
                            "value_kind": value_kind,
                            "sigma_kind": "n/a",
                            **cor_ortho,
                            "note": "ortho not used for sigma correlation; stats describe Δ distribution only",
                        })
    else:
        for key, sub in groups:
            if len(sub) < 3:
                continue
            task, ring_name, position = key
            if position == "ortho":
                continue  # 仅对 meta/para 做 Hammett 相关
            for value_kind in ("true", "pred"):
                val = f"delta_A_{value_kind}"
                for sigma_kind in ("sigma_m", "sigma_p"):
                    cor = _correlations(
                        sub[sigma_kind].to_numpy(dtype=float),
                        sub[val].to_numpy(dtype=float),
                    )
                    rows.append({
                        "scope": "within_context",
                        "task": task,
                        "anchor": "ALL",
                        "ring_name": ring_name,
                        "position": position,
                        "value_kind": value_kind,
                        "sigma_kind": sigma_kind,
                        **cor,
                    })
    return pd.DataFrame(rows)


def main(verbose: bool = True) -> Dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    absolute = pd.read_csv(ABSOLUTE_CSV) if ABSOLUTE_CSV.is_file() else pd.DataFrame()

    per_mol = _per_molecule_hammett(manifest, absolute)
    cov = _anchor_coverage(per_mol)
    cov.to_csv(ANCHOR_COV_CSV, index=False)

    matches = _hammett_matches(per_mol)
    if not matches.empty:
        matches.to_csv(MATCH_CSV, index=False)

    global_sum = _hammett_summary(matches, kind="global")
    context_sum = _hammett_summary(matches, kind="context")
    if not global_sum.empty:
        global_sum.to_csv(GLOBAL_SUMMARY_CSV, index=False)
    if not context_sum.empty:
        context_sum.to_csv(CTX_SUMMARY_CSV, index=False)

    if verbose:
        print(f"[hammett] saved {ANCHOR_COV_CSV}")
        print(f"[hammett] saved {MATCH_CSV} (rows={len(matches)})")
        print(f"[hammett] saved {GLOBAL_SUMMARY_CSV} (rows={len(global_sum)})")
        print(f"[hammett] saved {CTX_SUMMARY_CSV} (rows={len(context_sum)})")

    return {
        "anchor_coverage_csv": str(ANCHOR_COV_CSV),
        "matches_csv": str(MATCH_CSV),
        "global_summary_csv": str(GLOBAL_SUMMARY_CSV),
        "context_summary_csv": str(CTX_SUMMARY_CSV),
        "n_matches": int(len(matches)),
        "n_global_rows": int(len(global_sum)),
        "n_context_rows": int(len(context_sum)),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))