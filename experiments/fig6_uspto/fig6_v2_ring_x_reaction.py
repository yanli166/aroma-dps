#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Ring × Reaction heatmaps, controlled slices, variance decomposition.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
TIERA = ROOT / "dearom_ring_pairs_A_tierA"
RINGFAM_DIR = ROOT / "fig6_analysis_v2" / "PANEL_D_RING_FAMILY"
RFAM_DIR = ROOT / "fig6_analysis_v2" / "REACTION_FAMILY_OPTIONAL"
RXCROSS = ROOT / "fig6_analysis_v2" / "VARIANCE_DECOMPOSITION_OPTIONAL"
RXCROSS.mkdir(parents=True, exist_ok=True)

DESC = ["L_HOMA", "L_nMCBO", "L_NICS"]
COLOR = {"L_HOMA": "#1f77b4", "L_nMCBO": "#2ca02c", "L_NICS": "#9467bd"}
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)


def main():
    complete = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions_complete.csv",
                           low_memory=False)
    complete = complete.rename(columns={
        "Delta_HOMA": "L_HOMA", "Delta_nMCBO": "L_nMCBO", "Delta_NICS_star": "L_NICS",
    })
    ringfam = pd.read_csv(RINGFAM_DIR / "ring_family_assignment.csv", low_memory=False)
    rxfam = pd.read_csv(RFAM_DIR / "reaction_family_assignment.csv", low_memory=False)
    df = complete.merge(ringfam[["pair_id", "ring_family"]], on="pair_id", how="left")
    df = df.merge(rxfam[["pair_id", "reaction_family"]], on="pair_id", how="left")
    df.to_csv(RXCROSS / "ring_x_reaction_master.csv", index=False)

    # ---- Ring × Reaction heatmap ----
    fam_keep_r = df.groupby("ring_family").filter(lambda g: len(g) >= 30)["ring_family"].unique().tolist()
    fam_keep_r = ["benzene-type", "phenol-type", "naphthalene-type", "indole-type",
                  "furan-type", "benzofuran-type", "benzothiophene-type",
                  "pyridine-type", "quinoline-type"]
    fam_keep_x = ["reduction_like", "single_addition_like", "multi_addition_like"]
    sub = df[df["ring_family"].isin(fam_keep_r) & df["reaction_family"].isin(fam_keep_x)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    for ax, desc in zip(axes, DESC):
        # median per cell, only cells with n>=15
        M = np.full((len(fam_keep_r), len(fam_keep_x)), np.nan)
        N = np.zeros_like(M)
        for i, rfam in enumerate(fam_keep_r):
            for j, xfam in enumerate(fam_keep_x):
                cell = sub[(sub["ring_family"] == rfam) & (sub["reaction_family"] == xfam)]
                if len(cell) >= 15:
                    M[i, j] = cell[desc].median()
                    N[i, j] = len(cell)
        vmax = np.nanpercentile(np.abs(M), 95)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(fam_keep_x)))
        ax.set_xticklabels(fam_keep_x, rotation=20, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(fam_keep_r)))
        ax.set_yticklabels(fam_keep_r, fontsize=7.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}\nn={int(N[i,j])}",
                            ha="center", va="center", fontsize=6.2,
                            color=("white" if abs(M[i, j]) > vmax * 0.55 else "black"))
        ax.set_title(f"{desc} (median, n≥15)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Ring × Reaction family (median L)", fontsize=10, y=1.04)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(RXCROSS / f"Fig6_RingReaction_heatmaps.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---- Multidimensional response heatmap (z_robust, ring×reaction -> 3 desc) ----
    global_med = {d: float(np.median(df[d])) for d in DESC}
    global_mad = {d: float(1.4826 * np.median(np.abs(df[d] - global_med[d]))) for d in DESC}
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    n_rows = len(fam_keep_r); n_cols = len(fam_keep_x) * 3
    M = np.full((n_rows, n_cols), np.nan)
    labels = []
    for i, rfam in enumerate(fam_keep_r):
        for j, xfam in enumerate(fam_keep_x):
            for k, d in enumerate(DESC):
                col = j * 3 + k
                cell = sub[(sub["ring_family"] == rfam) & (sub["reaction_family"] == xfam)]
                if len(cell) >= 15:
                    m = cell[d].median()
                    M[i, col] = (m - global_med[d]) / global_mad[d] if global_mad[d] > 0 else 0
            labels.append((j, k, xfam))
    vmax = np.nanpercentile(np.abs(M), 95)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    col_lab = []
    for j, xf in enumerate(fam_keep_x):
        for d in DESC:
            col_lab.append(f"{xf}\n{d.replace('L_','')}")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_lab, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{r}" for r in fam_keep_r], fontsize=7.5)
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color=("white" if abs(M[i, j]) > vmax * 0.55 else "black"))
    ax.set_title("Multidimensional z_robust response (Ring × Reaction)\n"
                 "(0 = global median; >0 = above-average loss)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("z_robust", fontsize=7.5)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(RXCROSS / f"Fig6_RingReaction_zrobust_heatmap.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---- Controlled Slice A: fixed reaction family, compare ring families ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    slice_x = "reduction_like"
    sub_slice = sub[sub["reaction_family"] == slice_x]
    for ax, desc in zip(axes, DESC):
        data = [sub_slice.loc[sub_slice["ring_family"] == rfam, desc].dropna().values
                for rfam in fam_keep_r]
        vp = ax.violinplot(data, positions=range(len(fam_keep_r)),
                           widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(COLOR[desc]); b.set_alpha(0.45); b.set_edgecolor("#333")
        ax.boxplot(data, positions=range(len(fam_keep_r)), widths=0.18,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor="#333"),
                   medianprops=dict(color="#C44E52", lw=1.4))
        ax.axhline(0, color="#666", lw=0.8, ls="--")
        ax.set_xticks(range(len(fam_keep_r)))
        ax.set_xticklabels(fam_keep_r, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(desc)
        ax.set_title(f"Slice A: reaction = {slice_x}\n{desc}", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(RXCROSS / f"Fig6_slice_A_fixed_reaction.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---- Controlled Slice B: fixed ring family, compare reaction families ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    slice_r = "benzene-type"
    sub_slice = sub[sub["ring_family"] == slice_r]
    for ax, desc in zip(axes, DESC):
        data = [sub_slice.loc[sub_slice["reaction_family"] == xfam, desc].dropna().values
                for xfam in fam_keep_x]
        vp = ax.violinplot(data, positions=range(len(fam_keep_x)),
                           widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(COLOR[desc]); b.set_alpha(0.45); b.set_edgecolor("#333")
        ax.boxplot(data, positions=range(len(fam_keep_x)), widths=0.18,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor="#333"),
                   medianprops=dict(color="#C44E52", lw=1.4))
        ax.axhline(0, color="#666", lw=0.8, ls="--")
        ax.set_xticks(range(len(fam_keep_x)))
        ax.set_xticklabels(fam_keep_x, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel(desc)
        ax.set_title(f"Slice B: ring = {slice_r}\n{desc}", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(RXCROSS / f"Fig6_slice_B_fixed_ring.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---- Variance decomposition ----
    rows = []
    fam_keep_full = [f for f in fam_keep_r if (df["ring_family"] == f).sum() >= 30]
    fam_keep_x2 = [f for f in fam_keep_x if (df["reaction_family"] == f).sum() >= 30]
    sub2 = df[df["ring_family"].isin(fam_keep_full) & df["reaction_family"].isin(fam_keep_x2)].copy()
    for desc in DESC:
        # main-effects OLS via numpy
        y = sub2[desc].dropna().values
        sub2d = sub2.dropna(subset=[desc])
        # Build dummy matrices
        rf_dummies = pd.get_dummies(sub2d["ring_family"], drop_first=True).astype(float)
        xf_dummies = pd.get_dummies(sub2d["reaction_family"], drop_first=True).astype(float)
        Xm = np.column_stack([np.ones(len(sub2d)), rf_dummies.values, xf_dummies.values])
        # OLS
        beta, *_ = np.linalg.lstsq(Xm, sub2d[desc].values, rcond=None)
        ss_tot = np.sum((sub2d[desc].values - sub2d[desc].mean()) ** 2)
        ss_res = np.sum((sub2d[desc].values - Xm @ beta) ** 2)
        # reduce model: Ring only
        Xr = np.column_stack([np.ones(len(sub2d)), rf_dummies.values])
        b_r, *_ = np.linalg.lstsq(Xr, sub2d[desc].values, rcond=None)
        ss_res_r = np.sum((sub2d[desc].values - Xr @ b_r) ** 2)
        # Reaction only
        Xx = np.column_stack([np.ones(len(sub2d)), xf_dummies.values])
        b_x, *_ = np.linalg.lstsq(Xx, sub2d[desc].values, rcond=None)
        ss_res_x = np.sum((sub2d[desc].values - Xx @ b_x) ** 2)
        # Variance fractions (semantic)
        # var explained by Ring = (ss_res_x - ss_res) / ss_tot  (Ring after Reaction)
        # var explained by Reaction = (ss_res_r - ss_res) / ss_tot (Reaction after Ring)
        # Both | after-other = joint - sum (overlap ignored)
        var_ring_after_x = max(0, (ss_res_x - ss_res) / ss_tot)
        var_x_after_ring = max(0, (ss_res_r - ss_res) / ss_tot)
        var_ring_unique = var_ring_after_x
        var_x_unique = var_x_after_ring
        var_shared = max(0, (1 - ss_res / ss_tot) - var_ring_unique - var_x_unique)
        var_residual = ss_res / ss_tot
        rows.append({"descriptor": desc,
                     "N": len(sub2d),
                     "Ring_explained_unique": float(var_ring_unique),
                     "Reaction_explained_unique": float(var_x_unique),
                     "Shared": float(var_shared),
                     "Residual": float(var_residual)})
        # also report full-main-effect
    vardec = pd.DataFrame(rows)
    vardec.to_csv(RXCROSS / "variance_decomposition.csv", index=False)

    # Stacked bar
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = ["L_HOMA", "L_nMCBO", "L_NICS"]
    r_share = vardec["Ring_explained_unique"].values
    x_share = vardec["Reaction_explained_unique"].values
    sh_share = vardec["Shared"].values
    res_share = vardec["Residual"].values
    bottoms = np.zeros(3)
    for vals, lab, color in [(r_share, "Ring unique", "#1f77b4"),
                             (x_share, "Reaction unique", "#ff7f0e"),
                             (sh_share, "Shared", "#888888"),
                             (res_share, "Residual", "#dddddd")]:
        ax.bar(labels, vals, bottom=bottoms, label=lab, color=color, edgecolor="white")
        bottoms += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of variance (descriptive)")
    ax.set_title("Variance decomposition of L\n(OLS, descriptive only — not causal)",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(RXCROSS / f"Fig6fB_variance_decomposition.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("\n[variance decomp]\n", vardec.to_string(index=False))


if __name__ == "__main__":
    main()
