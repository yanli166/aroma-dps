#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — NEW main panel 6f: Ring-family explanatory strength across descriptors.

For each descriptor L in {L_HOMA, L_nMCBO, L_NICS}:
    L ~ RingFamily
using:
  - 5-fold grouped cross-validation grouped by reaction_id (GroupKFold)
  - cross-validated R^2 with bootstrap 95% CI (clustered bootstrap at reaction level)
  - nonparametric effect size: Kruskal-Wallis epsilon^2

The old "top-6 discordant structures" analysis (panel_f) is retained but is now
an SI figure (discordant cases are the rare exceptions to the family pattern).
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
TIERA = ROOT / "dearom_ring_pairs_A_tierA"
RINGFAM = ROOT / "fig6_analysis_v2" / "PANEL_D_RING_FAMILY" / "ring_family_assignment.csv"
OUT = ROOT / "fig6_analysis_v2" / "PANEL_F_DISCORDANT"
OUT.mkdir(parents=True, exist_ok=True)

DESC = ["L_HOMA", "L_nMCBO", "L_NICS"]
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)


def kw_epsilon_sq(group_values):
    """Kruskal-Wallis epsilon^2 (nonparametric effect size).

    eps2 = (H - k + 1) / (N - k)
    where H = KW statistic, k = number of groups, N = total n.
    """
    vals = []
    labels = []
    for gi, g in enumerate(group_values):
        vals.extend(g)
        labels.extend([gi] * len(g))
    if len(vals) < 3 or len(set(labels)) < 2:
        return float("nan"), float("nan")
    H, p = stats.kruskal(*group_values)
    N = len(vals)
    k = len([g for g in group_values if len(g) > 0])
    eps2 = (H - k + 1) / (N - k)
    return float(eps2), float(p)


def cv_r2_by_reaction(y, groups, ring_family, n_splits=5, n_boot=2000, seed=RANDOM_STATE):
    """5-fold GroupKFold CV R2 (linear regression on one-hot ring family), with
    reaction-clustered bootstrap 95% CI of the mean CV R2."""
    df = pd.DataFrame({"y": y, "group": groups, "ring_family": ring_family})
    df = df.dropna(subset=["y"]).copy()
    # only keep well-populated families to make grouping meaningful
    df = df[df.groupby("ring_family")["ring_family"].transform("count") >= 5]
    X_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        df[["ring_family"]].astype(str))
    yv = df["y"].values
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for tr, te in gkf.split(X_enc, yv, groups=df["group"].values):
        model = LinearRegression().fit(X_enc[tr], yv[tr])
        pred = model.predict(X_enc[te])
        ss_res = np.sum((yv[te] - pred) ** 2)
        ss_tot = np.sum((yv[te] - yv[te].mean()) ** 2)
        fold_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    mean_cv_r2 = float(np.nanmean(fold_r2))

    # clustered bootstrap: resample reactions, recompute CV R2 (use same splits recomputed)
    boot = np.empty(n_boot)
    rng = np.random.default_rng(seed)
    reactions = df["group"].unique()
    for b in range(n_boot):
        rx_sub = rng.choice(reactions, size=len(reactions), replace=True)
        idx = df[df["group"].isin(rx_sub)].index
        sub = df.loc[idx]
        if sub["ring_family"].nunique() < 2:
            boot[b] = np.nan
            continue
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
            sub[["ring_family"]].astype(str))
        yv2 = sub["y"].values
        # reuse GroupKFold on the resampled set
        gkf2 = GroupKFold(n_splits=min(n_splits, sub["group"].nunique()))
        f2 = []
        for tr, te in gkf2.split(enc, yv2, groups=sub["group"].values):
            m = LinearRegression().fit(enc[tr], yv2[tr])
            pr = m.predict(enc[te])
            s_r = np.sum((yv2[te] - pr) ** 2)
            s_t = np.sum((yv2[te] - yv2[te].mean()) ** 2)
            f2.append(1 - s_r / s_t if s_t > 0 else np.nan)
        boot[b] = np.nanmean(f2)
    lo, hi = np.percentile(boot[~np.isnan(boot)], [2.5, 97.5])
    return {
        "mean_cv_r2": mean_cv_r2,
        "fold_r2": fold_r2,
        "boot_CI95_lo": float(lo),
        "boot_CI95_hi": float(hi),
        "n_reactions": int(df["group"].nunique()),
        "n_pairs": int(len(df)),
    }


def main():
    complete = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions_complete.csv",
                           low_memory=False)
    complete = complete.rename(columns={
        "Delta_HOMA": "L_HOMA", "Delta_nMCBO": "L_nMCBO", "Delta_NICS_star": "L_NICS"})
    rp = pd.read_csv(TIERA / "ring_pairs_ml.csv", low_memory=False,
                     usecols=["pair_id", "reaction_id"])
    ringfam = pd.read_csv(RINGFAM, low_memory=False)
    df = complete.merge(rp, on="pair_id", how="left")
    df = df.merge(ringfam[["pair_id", "ring_family"]], on="pair_id", how="left")
    df = df[df["ring_family"] != "unknown"].copy()
    print(f"[6f] master df: {len(df)} pairs, {df['reaction_id'].nunique()} reactions")

    rows = []
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for i, desc in enumerate(DESC):
        res = cv_r2_by_reaction(df[desc], groups=df["reaction_id"],
                                ring_family=df["ring_family"],
                                n_splits=5, n_boot=2000)
        groups_kw = [df.loc[df["ring_family"] == f, desc].dropna().values
                     for f in df["ring_family"].unique()]
        eps2, p_kw = kw_epsilon_sq(groups_kw)
        rows.append({"descriptor": desc,
                     **{k: v for k, v in res.items()},
                     "kw_epsilon2": eps2, "kw_p": p_kw})
        color = {"L_HOMA": "#1f77b4", "L_nMCBO": "#2ca02c", "L_NICS": "#9467bd"}[desc]
        ax.errorbar(i, res["mean_cv_r2"],
                    yerr=[[res["mean_cv_r2"] - res["boot_CI95_lo"]],
                          [res["boot_CI95_hi"] - res["mean_cv_r2"]]],
                    fmt="o", color=color, capsize=4, markersize=8, markeredgecolor="white",
                    markeredgewidth=1.2)
        ax.text(i, res["boot_CI95_hi"] + 0.02, f"ε²={eps2:.2f}",
                ha="center", fontsize=7.5, color="#333")
    ax.axhline(0, color="#666", lw=0.8, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["L_HOMA\n(geometric)", "L_nMCBO\n(delocalization)", "L_NICS\n(magnetic)"],
                       fontsize=8)
    ax.set_ylabel("cross-validated R² (ring family)")
    ax.set_title("Fig.6f — Ring-family explanatory strength\n(5-fold GroupKFold by reaction, "
                 "bootstrap 95% CI)", fontsize=8.5)
    ax.set_ylim(-0.05, 0.75)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"Fig6f_ring_family_cv_r2.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(OUT / "fig6f_ring_family_cv_r2_stats.csv", index=False)
    (OUT / "fig6f_ring_family_cv_r2_stats.json").write_text(
        json.dumps(stat_df.to_dict(orient="records"), indent=2))
    print(stat_df.to_string(index=False))
    print("[6f] done")


if __name__ == "__main__":
    main()
