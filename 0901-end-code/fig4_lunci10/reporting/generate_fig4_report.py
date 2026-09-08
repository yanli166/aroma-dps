"""Phase 13 (final report generator): 按科学问题组织报告 (非代码脚本顺序).

组织:
  Part 1: Does the ring-conditioned representation generalize to unseen molecules?
  Part 2: What determines external generalization difficulty?
  Part 3: Are substituent-induced relative effects more transferable than absolute aromaticity?
  Part 4: Why can relative predictions generalize better? (scaffold-level calibration bias cancellation)
  Part 5: Are the learned perturbations chemically meaningful? (Hammett + position)

约束:
  - 不超过数据证据, 不根据预期故事选择性解释结果.
  - 输出 fig4_summary.csv + FIG4_RESULTS_REPORT.md.

输入 (来自先前 phases 的 CSV):
  - results/fig4_lunci10_final/00_audit/  (manifest, overlap)
  - results/fig4_lunci10_final/01_external_absolute/  (absolute predictions, summary)
  - results/fig4_lunci10_final/02_novelty/  (category summary, NN tanimoto)
  - results/fig4_lunci10_final/03_pairwise/  (pair predictions, summary, feasibility)
  - results/fig4_lunci10_final/04_bias/  (scaffold bias)
  - results/fig4_lunci10_final/05_hammett/  (anchor coverage, matches, summaries)
  - results/fig4_lunci10_final/06_position/  (position effect)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

PROJ_ROOT = _PROJ_ROOT
CODE_END = f"{PROJ_ROOT}/archive/deprecated/code_end"
for p in (PROJ_ROOT, CODE_END):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(_PROJ_ROOT) / "0901-end-code/fig4_lunci10"
FINAL_DIR = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final"
ABS_SUMMARY = FINAL_DIR / "01_external_absolute" / "fig4a_generalization_summary.csv"
ABS_PRED = FINAL_DIR / "01_external_absolute" / "lunci10_absolute_predictions.csv"
NOVELTY_CAT = FINAL_DIR / "02_novelty" / "novelty_category_summary.csv"
NOVELTY_NN = FINAL_DIR / "02_novelty" / "nearest_neighbor_analysis.csv"
NOVELTY_RING = FINAL_DIR / "02_novelty" / "ring_family_error.csv"
NOVELTY_SUB = FINAL_DIR / "02_novelty" / "substituent_error.csv"
PAIR_SUMMARY = FINAL_DIR / "03_pairwise" / "fig4c_pairwise_summary.csv"
PAIR_PRED = FINAL_DIR / "03_pairwise" / "lunci10_pair_predictions.csv"
FEASIBILITY = FINAL_DIR / "03_pairwise" / "internal_pair_feasibility.json"
SCAFFOLD_BIAS = FINAL_DIR / "04_bias" / "scaffold_context_bias.csv"
BIAS_CORR = FINAL_DIR / "04_bias" / "bias_vs_error_correlation.csv"
ANCHOR_COV = FINAL_DIR / "05_hammett" / "anchor_coverage.csv"
HAMMETT_MATCH = FINAL_DIR / "05_hammett" / "hammett_matches.csv"
HAMMETT_GLOBAL = FINAL_DIR / "05_hammett" / "hammett_global_summary.csv"
HAMMETT_CTX = FINAL_DIR / "05_hammett" / "hammett_context_summary.csv"
POSITION_CSV = FINAL_DIR / "06_position" / "position_effect_summary.csv"
LINKED_CSV = FINAL_DIR / "06_position" / "linked_atom_summary.csv"
POSITION_PAIR = FINAL_DIR / "06_position" / "position_pairwise_compare.csv"

OUT_SUMMARY_CSV = FINAL_DIR / "fig4_summary.csv"
OUT_REPORT_MD = FINAL_DIR / "FIG4_RESULTS_REPORT.md"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]


def _safe_read(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _format_float(x: Any, digits: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------
def section_1(df_abs_sum: Optional[pd.DataFrame], df_abs_pred: Optional[pd.DataFrame],
              df_novelty_cat: Optional[pd.DataFrame],
              df_nn: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Part 1: Does the ring-conditioned representation generalize to unseen
    molecules?
    """
    out: Dict[str, Any] = {"title": "Ring-conditioned representation generalization",
                            "tables": [], "text": []}
    if df_abs_sum is None or df_abs_sum.empty:
        out["text"].append("Absolute prediction summary not available — Part 1 cannot be evaluated.")
        return out

    # 绝对误差 micro
    tab = df_abs_sum[df_abs_sum["task"].isin(TASKS)].copy()
    tab = tab.sort_values(["task", "model"])
    out["tables"].append(("Absolute prediction (micro)", tab))
    out["text"].append(
        "Per-task MAE / OOD penalty: see table. We deliberately do NOT pick "
        "the best model — all numbers are reported with the same frozen seed."
    )

    # novelty category MAE
    if df_novelty_cat is not None and not df_novelty_cat.empty:
        cat = df_novelty_cat.copy()
        out["tables"].append(("Category A/B/C/D MAE", cat))
        out["text"].append(
            "Category A (seen scaffold + seen substituent): expected to be the "
            "easiest. Category D (both unseen): expected to be the hardest. "
            "Comparison reports the actual MAE gap, not a normalized one."
        )

    if df_nn is not None and not df_nn.empty:
        nn = df_nn.copy()
        if "nn_tanimoto" in nn.columns:
            nn_t = nn.groupby("novelty_category")["nn_tanimoto"].agg(
                ["count", "mean", "std", "median"]
            ).reset_index()
            out["tables"].append(("Nearest Tanimoto by category", nn_t))
            out["text"].append(
                "Tanimoto similarity to nearest internal neighbor is reported "
                "as a descriptor of novelty. Low Tanimoto in Category D indicates "
                "truly OOD molecules."
            )
    return out


def section_2(df_abs_pred: Optional[pd.DataFrame], df_abs_sum: Optional[pd.DataFrame],
              df_novelty_ring: Optional[pd.DataFrame],
              df_novelty_sub: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Part 2: What determines external generalization difficulty?"""
    out: Dict[str, Any] = {"title": "Determinants of external generalization difficulty",
                            "tables": [], "text": []}
    if df_novelty_ring is not None and not df_novelty_ring.empty:
        ring = df_novelty_ring.sort_values("mean_abs_err", ascending=False).head(15)
        out["tables"].append(("Top-15 ring family errors", ring))
        out["text"].append(
            "Ring-family MAE distribution: highest-error rings are highlighted. "
            "Caveat: small N in some bins may inflate MAE."
        )
    if df_novelty_sub is not None and not df_novelty_sub.empty:
        sub = df_novelty_sub.sort_values("mean_abs_err", ascending=False).head(15)
        out["tables"].append(("Top-15 substituent errors", sub))
        out["text"].append(
            "Substituent-level MAE: indicates which functional groups are most "
            "mis-predicted by frozen models."
        )
    if df_abs_pred is not None and not df_abs_pred.empty and "ring_name" in df_abs_pred.columns:
        # 简化: 按 ring_name 报告 main model (RC_MPNN) 的 MAE
        sub = df_abs_pred[df_abs_pred["model"] == "RC_MPNN"].copy()
        if sub.empty:
            sub = df_abs_pred.copy()
        if "abs_err" in sub.columns:
            agg = sub.groupby(["ring_name", "task"])["abs_err"].agg(
                ["count", "mean", "std"]
            ).reset_index().rename(columns={"mean": "MAE", "std": "std"})
            out["tables"].append(("Per-ring × task MAE (RC_MPNN)", agg))
    return out


def section_3(df_pair_sum: Optional[pd.DataFrame], df_abs_sum: Optional[pd.DataFrame]
              ) -> Dict[str, Any]:
    """Part 3: Are substituent-induced relative effects more transferable than absolute
    aromaticity?"""
    out: Dict[str, Any] = {"title": "Relative vs absolute transferability",
                            "tables": [], "text": []}
    # absolute MAE (micro)
    if df_abs_sum is not None and not df_abs_sum.empty:
        abs_main = df_abs_sum[(df_abs_sum["task"].isin(TASKS))
                              & (df_abs_sum["model"] == "RC_MPNN")]
        out["tables"].append(("Absolute micro MAE (RC_MPNN)", abs_main[
            ["task", "MAE", "RMSE", "R2", "OOD_penalty"]
        ]))
    # pair MAE (micro)
    if df_pair_sum is not None and not df_pair_sum.empty:
        pair_main = df_pair_sum[(df_pair_sum["task"].isin(TASKS))
                                & (df_pair_sum["model"] == "RC_MPNN")]
        out["tables"].append(("Pairwise Δ micro MAE (RC_MPNN)", pair_main[
            ["task", "micro_MAE", "micro_sign_accuracy",
             "micro_pearson_r", "micro_spearman_rho"]
        ]))
        out["text"].append(
            "Direct comparison: |absolute MAE| vs |pairwise Δ MAE| on the same "
            "frozen RC_MPNN model. The relative metric can be smaller because "
            "shared scaffold-level systematic error cancels in subtraction."
        )
    return out


def section_4(df_pair_pred: Optional[pd.DataFrame], df_bias: Optional[pd.DataFrame],
              df_bias_corr: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Part 4: Why can relative predictions generalize better? (scaffold-level
    calibration bias cancellation)."""
    out: Dict[str, Any] = {"title": "Scaffold-level bias cancellation hypothesis",
                            "tables": [], "text": []}
    if df_bias is not None and not df_bias.empty:
        ring_bias = df_bias[df_bias["group_type"] == "ring"].copy()
        out["tables"].append(("Per-ring bias_g + MAE_g + residual SD", ring_bias[
            ["task", "ring_name", "n", "bias_g", "MAE_g", "residual_sd_after_bias"]
        ].sort_values("MAE_g", ascending=False).head(20)))
        out["text"].append(
            "bias_g = mean(pred - true) within ring. residual_sd_after_bias "
            "indicates within-ring dispersion after removing the constant offset. "
            "Note: bias_g is used ONLY for diagnostic, NOT for oracle correction."
        )
    if df_bias_corr is not None and not df_bias_corr.empty:
        out["tables"].append(("Spearman |bias| vs error", df_bias_corr))
        out["text"].append(
            "If absolute MAE is largely driven by bias_g, the subtraction Δ "
            "cancels that bias and the |Δ MAE| should be smaller than |abs MAE|. "
            "Empirical ρ(|bias|, MAE) and ρ(|bias|, ΔMAE) quantify this."
        )
    return out


def section_5(df_anchor: Optional[pd.DataFrame], df_match: Optional[pd.DataFrame],
              df_global: Optional[pd.DataFrame], df_ctx: Optional[pd.DataFrame],
              df_position: Optional[pd.DataFrame],
              df_linked: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Part 5: Are the learned perturbations chemically meaningful? (Hammett + position)."""
    out: Dict[str, Any] = {"title": "Chemical plausibility (Hammett & position)",
                            "tables": [], "text": []}
    if df_anchor is not None and not df_anchor.empty:
        out["tables"].append(("Anchor coverage", df_anchor))
        out["text"].append(
            "Anchor coverage reports how many (ring, position) groups have an "
            "anchor (F, Cl, OMe) molecule and how many substituents can be "
            "compared against it via subtraction. Low coverage on OMe may limit "
            "para-series conclusions."
        )
    if df_global is not None and not df_global.empty:
        out["tables"].append(("Hammett global summary (pooled, meta+para)",
                              df_global[df_global["scope"] == "pooled_no_ortho"]))
    if df_ctx is not None and not df_ctx.empty:
        out["tables"].append(("Hammett within-context summary",
                              df_ctx.head(30)))
    out["text"].append(
        "ortho entries are excluded from σ correlation; they appear in a separate "
        "ortho-only summary to avoid misuse of σm/σp for steric/hyperconjugation "
        "effects. Within-context correlation prevents Simpson's paradox."
    )
    if df_position is not None and not df_position.empty:
        out["tables"].append(("Position effect (per ring_name × sub_name × position)",
                              df_position.head(40)))
    if df_linked is not None and not df_linked.empty:
        out["tables"].append(("Linked-atom summary (C vs N)", df_linked))
        out["text"].append(
            "C-linked vs N-linked performance are NOT averaged together; the table "
            "reports separate MAE for each, since chemical environment may differ."
        )
    return out


def _table_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_no data_"
    sub = df.head(max_rows)
    try:
        return sub.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        # fallback: simple CSV-style
        return "```\n" + sub.to_csv(index=False) + "```\n"


def _render_section(s: Dict[str, Any]) -> str:
    md = [f"## {s['title']}\n"]
    for line in s.get("text", []):
        md.append(line + "\n")
    for title, df in s.get("tables", []):
        md.append(f"\n### {title}\n")
        md.append(_table_to_md(df) + "\n")
    return "\n".join(md)


def main(verbose: bool = True) -> Dict[str, Any]:
    # ---- 0. 加载全部产物 ----
    df_abs_sum = _safe_read(ABS_SUMMARY)
    df_abs_pred = _safe_read(ABS_PRED)
    df_novelty_cat = _safe_read(NOVELTY_CAT)
    df_nn = _safe_read(NOVELTY_NN)
    df_novelty_ring = _safe_read(NOVELTY_RING)
    df_novelty_sub = _safe_read(NOVELTY_SUB)
    df_pair_sum = _safe_read(PAIR_SUMMARY)
    df_pair_pred = _safe_read(PAIR_PRED)
    feasibility = _safe_read_json(FEASIBILITY)
    df_bias = _safe_read(SCAFFOLD_BIAS)
    df_bias_corr = _safe_read(BIAS_CORR)
    df_anchor = _safe_read(ANCHOR_COV)
    df_match = _safe_read(HAMMETT_MATCH)
    df_global = _safe_read(HAMMETT_GLOBAL)
    df_ctx = _safe_read(HAMMETT_CTX)
    df_position = _safe_read(POSITION_CSV)
    df_linked = _safe_read(LINKED_CSV)
    df_position_pair = _safe_read(POSITION_PAIR)

    sections = [
        section_1(df_abs_sum, df_abs_pred, df_novelty_cat, df_nn),
        section_2(df_abs_pred, df_abs_sum, df_novelty_ring, df_novelty_sub),
        section_3(df_pair_sum, df_abs_sum),
        section_4(df_pair_pred, df_bias, df_bias_corr),
        section_5(df_anchor, df_match, df_global, df_ctx,
                  df_position, df_linked),
    ]

    # ---- 1. 写 fig4_summary.csv (宽表, 跨 question × task × model) ----
    summary_rows: List[Dict[str, Any]] = []
    if df_abs_sum is not None and not df_abs_sum.empty:
        for _, r in df_abs_sum.iterrows():
            summary_rows.append({
                "question": "Q1_generalization",
                "task": r.get("task", ""),
                "model": r.get("model", ""),
                "metric": "absolute_MAE",
                "value": r.get("MAE", float("nan")),
                "secondary_value": r.get("OOD_penalty", float("nan")),
                "scope": "micro",
                "source_csv": str(ABS_SUMMARY.name),
            })
    if df_pair_sum is not None and not df_pair_sum.empty:
        for _, r in df_pair_sum.iterrows():
            summary_rows.append({
                "question": "Q3_relative_transferability",
                "task": r.get("task", ""),
                "model": r.get("model", ""),
                "metric": "delta_MAE_micro",
                "value": r.get("micro_MAE", float("nan")),
                "secondary_value": r.get("micro_sign_accuracy", float("nan")),
                "scope": "micro",
                "source_csv": str(PAIR_SUMMARY.name),
            })
            summary_rows.append({
                "question": "Q3_relative_transferability",
                "task": r.get("task", ""),
                "model": r.get("model", ""),
                "metric": "delta_MAE_macro_context",
                "value": r.get("macro_context_MAE", float("nan")),
                "secondary_value": r.get("macro_context_sign_accuracy", float("nan")),
                "scope": "macro_context",
                "source_csv": str(PAIR_SUMMARY.name),
            })
    if df_bias_corr is not None and not df_bias_corr.empty:
        for _, r in df_bias_corr.iterrows():
            summary_rows.append({
                "question": "Q4_bias_cancellation",
                "task": r.get("task", ""),
                "model": "ALL",
                "metric": "spearman_abs_bias_vs_MAE",
                "value": r.get("spearman_abs_bias_vs_MAE_g", float("nan")),
                "secondary_value": r.get("spearman_abs_bias_vs_delta_MAE_g", float("nan")),
                "scope": r.get("group_type", ""),
                "source_csv": str(BIAS_CORR.name),
            })
    if df_global is not None and not df_global.empty:
        for _, r in df_global.iterrows():
            summary_rows.append({
                "question": "Q5_chemical_plausibility",
                "task": r.get("task", ""),
                "model": "ALL",
                "metric": f"hammett_{r.get('sigma_kind','')}_{r.get('value_kind','')}",
                "value": r.get("spearman_rho", float("nan")),
                "secondary_value": r.get("pearson_r", float("nan")),
                "scope": r.get("scope", ""),
                "source_csv": str(HAMMETT_GLOBAL.name),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    # ---- 2. 写 FIG4_RESULTS_REPORT.md ----
    md_lines: List[str] = []
    md_lines.append("# Fig.4 — Ring-Conditioned Generalization to lunci10\n")
    md_lines.append("## Scope & protocol\n")
    if feasibility is not None:
        md_lines.append(f"- Protocol: **{feasibility.get('protocol_I_feasible')}** "
                         f"({'Protocol I' if feasibility.get('protocol_I_feasible') else 'Protocol II'})\n")
        md_lines.append(f"- Rationale: {feasibility.get('rationale', '')}\n")
        md_lines.append(f"- Internal pair rows: {feasibility.get('n_internal_pairs', 0)}\n")
    md_lines.append("\nThe report is organized by *scientific question*, not by code script order. "
                     "Each section states the evidence available and the limitations of that evidence. "
                     "We do not select results based on a preferred narrative; negative or null findings "
                     "are kept as such.\n")
    for s in sections:
        md_lines.append("\n" + _render_section(s))

    # Limitations footer
    md_lines.append("\n## Limitations & caveats\n")
    md_lines.append("- **Strict zero-shot:** no fit / fine-tune / early-stop on lunci10.\n")
    md_lines.append("- **Scaffold-level bias:** reported as diagnostic only — not used for oracle correction.\n")
    md_lines.append("- **Hammett σ:** Hansch 1991 constants used as-is. σ values were not adjusted based on lunci10 fit.\n")
    md_lines.append("- **ortho effects:** excluded from σ correlation; steric / hyperconjugation differs from σm/σp framework.\n")
    md_lines.append("- **Within-context only:** any position / Hammett comparisons across scaffolds would risk Simpson's paradox — not reported.\n")
    md_lines.append("- **Simpson's paradox safeguard:** position-pairwise comparisons are restricted to (ring_name, sub_name) groups.\n")

    OUT_REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    if verbose:
        print(f"[report] saved {OUT_SUMMARY_CSV} (rows={len(summary_df)})")
        print(f"[report] saved {OUT_REPORT_MD}")

    return {
        "summary_csv": str(OUT_SUMMARY_CSV),
        "report_md": str(OUT_REPORT_MD),
        "n_summary_rows": int(len(summary_df)),
        "protocol": ("I" if feasibility and feasibility.get("protocol_I_feasible")
                     else "II"),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))