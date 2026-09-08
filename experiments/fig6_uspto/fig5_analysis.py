#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.5 analysis: reaction-level aromaticity change across 4377 complete
reactant/product target-ring pairs (USPTO dearomatization corpus).

Pipeline:
  1. master table (01_master)
  2. global R vs P change + Delta distributions (02_global_change)
  3. descriptor coupling / pairwise correlations (03_descriptor_coupling)
  4. PCA (04_pca)
  5. discordant cases (05_discordant)
  6. reactant-product structural change (02_global_change / 08_statistics)
  7. ring family annotation (07_ring_family)
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
DATA = ROOT / "dearom_ring_pairs_A_tierA"
OUT = ROOT / "fig5_analysis"
COMPLETE = DATA / "ring_pair_aromaticity_predictions_complete.csv"
RINGPAIRS = DATA / "ring_pairs_ml.csv"
PRED = DATA / "ring_property_predictions.csv"
DISC = DATA / "descriptor_discordant_cases.csv"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8

OUTS = {
    "master": OUT / "01_master",
    "global": OUT / "02_global_change",
    "coupling": OUT / "03_descriptor_coupling",
    "pca": OUT / "04_pca",
    "disc": OUT / "05_discordant",
    "chemspace": OUT / "06_chemical_space",
    "rfam": OUT / "07_ring_family",
    "stat": OUT / "08_statistics",
    "main": OUT / "09_fig5_main",
    "si": OUT / "10_fig5_si",
    "logs": OUT / "logs",
}
for d in OUTS.values():
    d.mkdir(parents=True, exist_ok=True)

DESC = ["Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star"]
PAIRED = [
    ("HOMA", "HOMA_pred_reactant", "HOMA_pred_product", "Delta_HOMA"),
    ("nMCBO", "nMCBO_pred_reactant", "nMCBO_pred_product", "Delta_nMCBO"),
    ("NICS(1)zz", "NICS_1zz_pred_reactant", "NICS_1zz_pred_product", "Delta_NICS_star"),
]


def save_fig(fig, name, main=True):
    d = OUTS["main"] if main else OUTS["si"]
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(x, n_boot=5000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boot = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def parse_idx(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return [int(x) for x in str(s).replace(",", ";").split(";") if x.strip() and x.strip().lstrip("-").isdigit()]


# ---------------------------------------------------------------- 1. master
def build_master():
    c = pd.read_csv(COMPLETE, low_memory=False)
    rp = pd.read_csv(RINGPAIRS, low_memory=False, usecols=[
        "pair_id", "reaction_id",
        "reactant_component_smiles", "product_component_smiles",
        "reactant_component_smiles_mapped", "product_component_smiles_mapped",
        "target_ring_map_numbers", "target_ring_size",
        "lost_aromatic_atoms", "lost_aromatic_ring_edges",
        "new_sp3_ring_atoms", "new_stereocenters_on_ring",
        "reactant_ring_fragment_smiles", "product_ring_fragment_smiles",
        "exact_score",
    ])
    m = c.merge(rp, on="pair_id", how="left")
    print(f"[master] complete={len(c)} merged={len(m)} missing={m['reaction_id'].isna().sum()}")
    # keep only target columns in required order
    cols = ["pair_id", "reaction_id",
            "HOMA_pred_reactant", "HOMA_pred_product", "Delta_HOMA",
            "nMCBO_pred_reactant", "nMCBO_pred_product", "Delta_nMCBO",
            "NICS_1zz_pred_reactant", "NICS_1zz_pred_product", "Delta_NICS_star",
            "aromaticity_pattern",
            "reactant_component_smiles", "product_component_smiles",
            "reactant_component_smiles_mapped", "product_component_smiles_mapped",
            "target_ring_map_numbers", "target_ring_size",
            "lost_aromatic_atoms", "lost_aromatic_ring_edges",
            "new_sp3_ring_atoms", "new_stereocenters_on_ring",
            "reactant_ring_fragment_smiles", "product_ring_fragment_smiles",
            "exact_score"]
    m = m[cols]
    m.to_csv(OUTS["master"] / "reaction_aromaticity_fig5_master.csv", index=False)
    print(f"[master] wrote {len(m)} rows -> 01_master/reaction_aromaticity_fig5_master.csv")
    return m


# ---------------------------------------------------------------- 2. global change
def global_change(m):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    for ax, (name, rcol, pcol, dcol) in zip(axes, PAIRED):
        R = m[rcol].astype(float)
        P = m[pcol].astype(float)
        d = m[dcol].astype(float)  # unified convention: Delta>0 = aromaticity loss
        # paired Wilcoxon
        try:
            w = stats.wilcoxon(R, P)
            pval = float(w.pvalue)
        except Exception as e:
            pval = float("nan")
        med_diff = float(np.median(d))
        ci_lo, ci_hi = bootstrap_ci(d)
        # probability of superiority (paired effect size)
        p_sup = float((d > 0).mean())
        # rank-biserial for paired (Matched rank-biserial)
        ranks = stats.rankdata(np.abs(d))
        w_plus = float(np.sum(ranks[d > 0]))
        w_minus = float(np.sum(ranks[d < 0]))
        r_rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0
        rows.append({
            "descriptor": name, "N": int(len(d)),
            "median_reactant": float(np.median(R)), "median_product": float(np.median(P)),
            "mean_reactant": float(np.mean(R)), "mean_product": float(np.mean(P)),
            "median_delta": med_diff, "bootstrap95CI_median_delta_lo": ci_lo,
            "bootstrap95CI_median_delta_hi": ci_hi,
            "fraction_delta_gt0": float((d > 0).mean()),
            "wilcoxon_p": pval,
            "paired_rank_biserial": r_rb,
            "probability_of_superiority": p_sup,
        })
        # plot: violin + box + scatter
        data = [R.values, P.values]
        vp = ax.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#4C72B0" if b.get_paths() else "#4C72B0")
            b.set_alpha(0.35)
            b.set_edgecolor("#2b3f5f")
        bp = ax.boxplot(data, positions=[0, 1], widths=0.16, showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color="#C44E52", linewidth=1.4),
                        boxprops=dict(facecolor="white", edgecolor="#333333"),
                        whiskerprops=dict(color="#333333"),
                        capprops=dict(color="#333333"))
        rng = np.random.default_rng(7)
        for pos, vals in zip([0, 1], data):
            jx = rng.normal(pos, 0.06, size=len(vals))
            ax.scatter(jx, vals, s=1.5, alpha=0.12, color="#4C72B0", rasterized=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Reactant", "Product"])
        ax.set_ylabel(name)
        ax.set_title(f"{name}   (median Δ = {med_diff:.3f})", fontsize=8)
        ax.axvline(0.5, color="#999999", lw=0.6, ls="--")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Reactant vs Product target-ring aromaticity (N = {len(m)} pairs)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "Fig5b_reactant_vs_product")
    df = pd.DataFrame(rows)
    df.to_csv(OUTS["global"] / "paired_aromaticity_statistics.csv", index=False)
    print("\n[global] paired statistics:\n", df.to_string(index=False))

    # Delta distributions (Fig5c)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    for ax, name, dcol, frac in zip(axes, ["HOMA", "nMCBO", "NICS*"], DESC, ["99.6%", "98.3%", "99.4%"]):
        x = m[dcol].astype(float).values
        ax.hist(x, bins=60, density=True, alpha=0.55, color="#4C72B0")
        ax.axvline(0, color="#C44E52", lw=1.4)
        ax.axvline(np.median(x), color="#2b3f5f", lw=1.2, ls="--")
        ax.set_xlabel(dcol)
        ax.set_ylabel("density")
        ax.set_title(f"Δ {name}: {frac} with loss (median {np.median(x):.3f})", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Delta distributions (positive = aromaticity loss)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "Fig5c_delta_distributions")

    # save plot data
    gsum = {
        "N": int(len(m)),
        **{f"median_{c}": float(np.median(m[c])) for c in DESC},
        **{f"mean_{c}": float(np.mean(m[c])) for c in DESC},
        **{f"frac_gt0_{c}": float((m[c] > 0).mean()) for c in DESC},
        "ALL_LOSS_frac": float((m["aromaticity_pattern"] == "ALL_LOSS").mean()),
        "DISCORDANT_frac": float((m["aromaticity_pattern"] == "DISCORDANT").mean()),
        "ALL_GAIN_OR_NONLOSS_frac": float((m["aromaticity_pattern"] == "ALL_GAIN_OR_NONLOSS").mean()),
    }
    (OUTS["global"] / "global_delta_summary.json").write_text(json.dumps(gsum, indent=2))
    (OUTS["global"] / "global_delta_summary.csv").write_text(
        pd.DataFrame([gsum]).to_csv(index=False))
    return df


# ---------------------------------------------------------------- 3. coupling
def descriptor_coupling(m):
    X = m[DESC].astype(float)
    rows = []
    pairs = [("Delta_HOMA", "Delta_nMCBO"),
             ("Delta_HOMA", "Delta_NICS_star"),
             ("Delta_nMCBO", "Delta_NICS_star")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    for ax, (a, b) in zip(axes, pairs):
        x, y = X[a].values, X[b].values
        rp = stats.pearsonr(x, y)[0]
        rs = stats.spearmanr(x, y)[0]
        # bootstrap CI of pearson
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(x), size=(3000, len(x)))
        rr = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
        lo, hi = np.percentile(rr, [2.5, 97.5])
        hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", bins="log", mincnt=1)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log count", fontsize=6)
        ax.set_xlabel(a.replace("_", " "))
        ax.set_ylabel(b.replace("_", " "))
        ax.set_title(f"r = {rp:.3f}   ρ = {rs:.3f}", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        rows.append({"x": a, "y": b, "pearson_r": float(rp), "spearman_rho": float(rs),
                     "pearson_95CI_lo": float(lo), "pearson_95CI_hi": float(hi), "N": int(len(x))})
    fig.suptitle("Pairwise Delta descriptor coupling (N = 4377)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "Fig5d_descriptor_coupling")

    corr = pd.DataFrame(rows)
    corr.to_csv(OUTS["coupling"] / "descriptor_pairwise_correlations.csv", index=False)
    print("\n[coupling]\n", corr.to_string(index=False))

    # full Pearson/Spearman matrix for SI
    pm = X.corr(method="pearson"); sm = X.corr(method="spearman")
    pm.to_csv(OUTS["coupling"] / "pearson_matrix.csv")
    sm.to_csv(OUTS["coupling"] / "spearman_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, cm, title in zip(axes, [pm, sm], ["Pearson r", "Spearman ρ"]):
        im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([c.replace("Delta_", "") for c in cm.columns], fontsize=7)
        ax.set_yticklabels([c.replace("Delta_", "") for c in cm.index], fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, "S3_descriptor_correlation_matrix", main=False)

    # Fig5d left: concordance proportion bar
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    labels = ["ALL_LOSS", "DISCORDANT", "ALL_GAIN_OR_NONLOSS"]
    vals = [(m["aromaticity_pattern"] == "ALL_LOSS").sum(),
            (m["aromaticity_pattern"] == "DISCORDANT").sum(),
            (m["aromaticity_pattern"] == "ALL_GAIN_OR_NONLOSS").sum()]
    cols = ["#4C72B0", "#DD8452", "#C44E52"]
    fracs = [v / sum(vals) for v in vals]
    left = 0
    for lab, v, f, cc in zip(labels, vals, fracs, cols):
        ax.barh(0, v, left=left, color=cc, height=0.5, label=f"{lab} ({f*100:.1f}%)")
        left += v
    ax.set_xlim(0, sum(vals)); ax.set_yticks([]); ax.set_ylim(-0.4, 0.6)
    ax.legend(loc="center right", bbox_to_anchor=(1.28, 0.5), fontsize=7, frameon=False)
    ax.set_title(f"Descriptor concordance (N={len(m)})", fontsize=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "Fig5d_left_concordance_bar")
    return corr


# ---------------------------------------------------------------- 4. PCA
def run_pca(m):
    X = m[DESC].astype(float)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    pca = PCA().fit(Z)
    scores = pca.transform(Z)
    loadings = pca.components_.T
    sc_df = pd.DataFrame({
        "pair_id": m["pair_id"], "reaction_id": m["reaction_id"],
        "aromaticity_pattern": m["aromaticity_pattern"],
        "PC1": scores[:, 0], "PC2": scores[:, 1], "PC3": scores[:, 2],
        "z_HOMA": Z[:, 0], "z_nMCBO": Z[:, 1], "z_NICS": Z[:, 2],
        **{c: m[c].values for c in DESC},
    })
    sc_df.to_csv(OUTS["pca"] / "descriptor_delta_pca_scores.csv", index=False)
    ld_df = pd.DataFrame(loadings, index=DESC,
                         columns=[f"PC{i+1}" for i in range(3)])
    ld_df.to_csv(OUTS["pca"] / "descriptor_delta_pca_loadings.csv")
    evr = pca.explained_variance_ratio_
    print("\n[pca] explained variance:", evr)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    is_disc = (m["aromaticity_pattern"] == "DISCORDANT").values
    ax.scatter(scores[~is_disc, 0], scores[~is_disc, 1], s=6, alpha=0.35, color="#9aa5ad",
               label="ALL_LOSS", rasterized=True)
    ax.scatter(scores[is_disc, 0], scores[is_disc, 1], s=12, alpha=0.9, color="#C44E52",
               marker="x", label="DISCORDANT")
    # loading arrows
    scale = max(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max()) * 1.15
    for k, nm in enumerate(DESC):
        lx, ly = loadings[k, 0], loadings[k, 1]
        ax.arrow(0, 0, lx * scale, ly * scale, color="#2b3f5f", width=0.004,
                 head_width=0.05, length_includes_head=True)
        ax.text(lx * scale * 1.08, ly * scale * 1.08,
                nm.replace("Delta_", ""), color="#2b3f5f", fontsize=8)
    ax.axhline(0, color="#cccccc", lw=0.6); ax.axvline(0, color="#cccccc", lw=0.6)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("PCA of standardized Delta descriptors", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "Fig5e_pca")
    (OUTS["pca"] / "pca_explained_variance.json").write_text(
        json.dumps({"PC1": float(evr[0]), "PC2": float(evr[1]), "PC3": float(evr[2])}, indent=2))
    return sc_df, ld_df, evr


# ---------------------------------------------------------------- 5. discordant
def discordant_analysis(m):
    d = pd.read_csv(DISC, low_memory=False)
    d = d.merge(m[["pair_id", "reaction_id", "reactant_component_smiles",
                    "product_component_smiles", "target_ring_size",
                    "reactant_ring_fragment_smiles"]], on="pair_id", how="left")
    # type: which descriptors are <= 0 (reversed / non-loss)
    def dtype(row):
        neg = [c for c in DESC if row[c] <= 0]
        if not neg:
            return "ALL_LOSS_LE"  # shouldn't happen
        return "_".join(c.replace("Delta_", "") for c in neg) + "_NEGATIVE_ONLY"
    d["discordance_type"] = d.apply(dtype, axis=1)
    # strength on standardized scale (use global mean/std from full corpus)
    X = m[DESC].astype(float)
    mu = X.mean(); sd = X.std()
    for c in DESC:
        d[f"z_{c}"] = (d[c].astype(float) - mu[c]) / sd[c]
    zcols = [f"z_{c}" for c in DESC]
    d["discordance_strength"] = d[zcols].max(axis=1) - d[zcols].min(axis=1)
    d["discordance_strength_std"] = d[zcols].std(axis=1)
    d.to_csv(OUTS["disc"] / "discordant_cases_annotated.csv", index=False)
    tc = d["discordance_type"].value_counts().reset_index()
    tc.columns = ["discordance_type", "count"]
    tc.to_csv(OUTS["disc"] / "discordance_type_counts.csv", index=False)
    print("\n[discordant] type counts:\n", tc.to_string(index=False))
    top = d.sort_values("discordance_strength", ascending=False).head(50)
    top_out = top[["pair_id", "reaction_id", "reactant_component_smiles",
                   "product_component_smiles", "reactant_ring_fragment_smiles",
                   "target_ring_size", "Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star",
                   "discordance_type", "discordance_strength"]].copy()
    top_out.columns = ["pair_id", "reaction_id", "Reactant_SMILES", "Product_SMILES",
                       "target_ring", "target_ring_size", "Delta_HOMA", "Delta_nMCBO",
                       "Delta_NICS_star", "discordance_type", "discordance_strength"]
    top_out.to_csv(OUTS["disc"] / "top_50_descriptor_discordant_cases.csv", index=False)
    return d


# ---------------------------------------------------------------- 6. structural change + initial aromaticity
def structural_change(m):
    rows = []
    def mfp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    for _, r in m.iterrows():
        fr = mfp(r["reactant_component_smiles"])
        fp = mfp(r["product_component_smiles"])
        if fr is None or fp is None:
            rows.append({"pair_id": r["pair_id"], "Tanimoto_RP": np.nan,
                         "StructuralChange": np.nan, "structural_valid": False})
            continue
        t = DataStructs.TanimotoSimilarity(fr, fp)
        rows.append({"pair_id": r["pair_id"], "Tanimoto_RP": float(t),
                     "StructuralChange": float(1 - t), "structural_valid": True,
                     "HOMA_R": r["HOMA_pred_reactant"], "Delta_HOMA": r["Delta_HOMA"],
                     "nMCBO_R": r["nMCBO_pred_reactant"], "Delta_nMCBO": r["Delta_nMCBO"],
                     "NICS_R": r["NICS_1zz_pred_reactant"], "Delta_NICS_star": r["Delta_NICS_star"]})
    sc = pd.DataFrame(rows)
    sc.to_csv(OUTS["stat"] / "reactant_product_structural_change.csv", index=False)
    ok = sc[sc["structural_valid"] & sc["StructuralChange"].notna()]
    print(f"\n[structural] valid={len(ok)}")
    for c, rc in [("Delta_HOMA", "HOMA_R"), ("Delta_nMCBO", "nMCBO_R"), ("Delta_NICS_star", "NICS_R")]:
        rp = stats.pearsonr(ok["StructuralChange"], ok[c])[0]
        rs = stats.spearmanr(ok["StructuralChange"], ok[c])[0]
        print(f"  StructuralChange vs {c}: pearson={rp:.3f} spearman={rs:.3f}")

    # initial aromaticity vs delta + residual analysis
    init_rows = []
    for name, rcol, pcol, dcol in [("HOMA", "HOMA_pred_reactant", "HOMA_pred_product", "Delta_HOMA"),
                                    ("nMCBO", "nMCBO_pred_reactant", "nMCBO_pred_product", "Delta_nMCBO"),
                                    ("NICS", "NICS_1zz_pred_reactant", "NICS_1zz_pred_product", "Delta_NICS_star")]:
        R = m[rcol].astype(float); P = m[pcol].astype(float); D = m[dcol].astype(float)
        # A_R vs Delta (mathematically coupled) and A_R vs A_P
        r_delta = stats.pearsonr(R, D)[0]
        r_rp = stats.pearsonr(R, P)[0]
        # residual of A_P ~ A_R (OLS), correlate residual with structural change where available
        b = np.polyfit(R, P, 1)
        resid = P - (b[0] * R + b[1])
        if len(ok):
            r_res_struct = stats.pearsonr(ok["StructuralChange"], resid.reindex(ok.index))[0]
        else:
            r_res_struct = np.nan
        init_rows.append({"descriptor": name, "pearson_R_vs_Delta": float(r_delta),
                          "pearson_R_vs_P": float(r_rp),
                          "OLS_Ap_vs_Ar_slope": float(b[0]), "OLS_Ap_vs_Ar_intercept": float(b[1]),
                          "pearson_residual_vs_structural_change": float(r_res_struct)})
    ini = pd.DataFrame(init_rows)
    ini.to_csv(OUTS["stat"] / "initial_aromaticity_analysis.csv", index=False)
    print("\n[initial aromaticity]\n", ini.to_string(index=False))
    return sc


# ---------------------------------------------------------------- 7. ring family
def ring_family(m):
    # Use reactant_component_smiles_mapped + target_ring_map_numbers to get target atoms
    def mfp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    def classify(mol, target_idx):
        target = set(target_idx)
        if not target:
            return "unknown"
        size = len(target)
        syms = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in target)
        nC = syms.get("C", 0); nN = syms.get("N", 0); nO = syms.get("O", 0); nS = syms.get("S", 0)
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in target)
        # fused? does any target atom belong to another SSSR ring?
        sssr = list(Chem.GetSymmSSSR(mol))
        is_fused = False
        fused_6c = fused_5c = 0
        for ring in sssr:
            rset = set(ring)
            if rset == target:
                continue
            if rset & target:
                is_fused = True
                if all(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in ring):
                    if len(ring) == 6:
                        fused_6c += 1
                    elif len(ring) == 5:
                        fused_5c += 1
        # ring heteroatom
        has_N = nN > 0; has_O = nO > 0; has_S = nS > 0
        # OH directly attached to ring carbon
        has_oh = False
        for i in target:
            a = mol.GetAtomWithIdx(i)
            if a.GetSymbol() == "C":
                for nb in a.GetNeighbors():
                    if nb.GetSymbol() == "O" and nb.GetIdx() not in target:
                        has_oh = True
        # benzyl-type / N-substituted
        if size == 6 and nC == 6 and not has_N and not has_O and not has_S and not is_fused:
            return "phenol-type" if has_oh else "benzene"
        if size == 6 and nC == 6 and not has_N and not has_O and not has_S and is_fused and fused_6c:
            return "naphthol-type" if has_oh else "naphthalene"
        if size == 5 and nC == 4 and nN == 1 and not has_O and not has_S:
            return "indole" if is_fused else "pyrrole"
        if size == 5 and nC == 4 and nO == 1 and not has_N and not has_S:
            return "benzofuran" if is_fused else "furan"
        if size == 5 and nC == 4 and nS == 1 and not has_N and not has_O:
            return "benzothiophene" if is_fused else "thiophene"
        if size == 6 and nC == 5 and nN == 1 and not has_O and not has_S:
            if is_fused:
                # distinguish quinoline (N in ring) vs isoquinoline via fragment-smiles N adjacency
                # crude: N ortho to fusion vs para
                return "quinoline_or_isoquinoline"
            return "pyridine"
        if size == 5 and nN == 1 and not has_O and not has_S:
            return "indole" if is_fused else "pyrrole"
        if size == 6 and has_N and not has_O and not has_S:
            return "quinoline_or_isoquinoline" if is_fused else "other N-heteroarene"
        if size == 6 and has_O and not has_N and not has_S:
            return "other O-heteroarene"
        if size == 6 and has_S and not has_N and not has_O:
            return "other S-heteroarene"
        if has_N and not has_O and not has_S:
            return "other N-heteroarene"
        if has_O and not has_S:
            return "other O-heteroarene"
        if has_S:
            return "other S-heteroarene"
        if nC == size:
            return "other carbocycle"
        return "unknown"

    rows = []
    n_fail = 0
    for _, r in m.iterrows():
        try:
            mol = Chem.MolFromSmiles(r["reactant_component_smiles_mapped"])
            maps = set(int(x) for x in str(r["target_ring_map_numbers"]).replace(";", " ").split() if x.strip().isdigit())
            if mol is None or not maps:
                raise ValueError("parse fail")
            target_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() in maps]
            fam = classify(mol, target_idx)
            # radius-2 local fingerprint for chemical space later
            local = set(target_idx)
            frontier = set(target_idx)
            for _ in range(2):
                new = set()
                for i in frontier:
                    new.update(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
                new -= local
                local |= new
                frontier = new
            frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(local),
                                                canonical=True, isomericSmiles=True)
            frag = Chem.MolFromSmiles(frag_smi)
            fp = mfp(frag_smi) if frag is not None else None
            rows.append({"pair_id": r["pair_id"], "ring_family": fam,
                         "ring_size": len(target_idx),
                         "local_fragment_smiles": frag_smi,
                         "local_fp": fp.ToBitString() if fp is not None else ""})
        except Exception:
            n_fail += 1
            rows.append({"pair_id": r["pair_id"], "ring_family": "unknown",
                         "ring_size": np.nan, "local_fragment_smiles": "",
                         "local_fp": ""})
    rf = pd.DataFrame(rows)
    rf.to_csv(OUTS["rfam"] / "ring_family_annotations.csv", index=False)
    cnt = rf["ring_family"].value_counts().reset_index()
    cnt.columns = ["ring_family", "count"]
    cnt.to_csv(OUTS["rfam"] / "ring_family_counts.csv", index=False)
    print(f"\n[ring family] annotated={len(rf)} fail={n_fail}")
    print(cnt.to_string(index=False))

    # ALL_LOSS fraction per family (for report)
    mf = m.merge(rf[["pair_id", "ring_family"]], on="pair_id", how="left")
    per = mf.groupby("ring_family").apply(
        lambda g: pd.Series({"N": len(g), "ALL_LOSS_frac": (g["aromaticity_pattern"] == "ALL_LOSS").mean(),
                             "median_Delta_HOMA": g["Delta_HOMA"].median(),
                             "median_Delta_nMCBO": g["Delta_nMCBO"].median(),
                             "median_Delta_NICS_star": g["Delta_NICS_star"].median()}),
        include_groups=False).reset_index()
    per.to_csv(OUTS["rfam"] / "ring_family_delta_summary.csv", index=False)
    print(per.to_string(index=False))
    return rf, cnt


def main():
    m = build_master()
    global_change(m)
    descriptor_coupling(m)
    sc_df, ld_df, evr = run_pca(m)
    d = discordant_analysis(m)
    structural_change(m)
    ring_family(m)
    print("\nALL FIG5 ANALYSIS STEPS COMPLETE")


if __name__ == "__main__":
    main()
