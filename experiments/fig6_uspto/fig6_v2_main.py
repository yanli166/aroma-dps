#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Panels B, C, D, E, F, EXPLORATORY.

Inputs:
  dearom_ring_pairs_A_tierA/ring_pair_aromaticity_predictions_complete.csv  (4,377 target pairs)
  dearom_ring_pairs_A_tierA/ring_pairs_ml.csv                             (4,443 tierA pair metadata)
  fig6_analysis_v2/PANEL_B_TARGET_SPECTATOR/spectator_ring_pair_aromaticity_predictions.csv

Outputs: under fig6_analysis_v2/<PANEL>/<stats.csv|figure.{svg,pdf,png}|tables.csv|...>
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem import rdDepictor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
TIERA = ROOT / "dearom_ring_pairs_A_tierA"
ALLRINGS = ROOT / "dearom_ring_pairs_A"
SPEC = ROOT / "fig6_analysis_v2" / "PANEL_B_TARGET_SPECTATOR"
V2 = ROOT / "fig6_analysis_v2"
OUT = {p.name: V2 / p.name for p in [
    V2 / "PANEL_B_TARGET_SPECTATOR",
    V2 / "PANEL_C_GLOBAL_LOSS",
    V2 / "PANEL_D_RING_FAMILY",
    V2 / "PANEL_E_STRUCTURAL_CHANGE",
    V2 / "PANEL_F_DISCORDANT",
    V2 / "SI",
    V2 / "REPORT",
]}
for d in OUT.values():
    d.mkdir(parents=True, exist_ok=True)

DESC = ["L_HOMA", "L_nMCBO", "L_NICS"]
COLOR = {"L_HOMA": "#1f77b4", "L_nMCBO": "#2ca02c", "L_NICS": "#9467bd"}
DESC_FILE = ["Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star"]
PAIRED_COLS = list(zip(DESC, DESC_FILE))

RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------- load + canonicalise
def load_data():
    complete = pd.read_csv(TIERA / "ring_pair_aromaticity_predictions_complete.csv", low_memory=False)
    rp = pd.read_csv(TIERA / "ring_pairs_ml.csv", low_memory=False)
    spec_pairs = pd.read_csv(SPEC / "spectator_ring_pair_aromaticity_predictions.csv", low_memory=False)
    spec_samples = pd.read_csv(SPEC / "spectator_ring_property_ml_samples_fixed.csv",
                               low_memory=False, usecols=["pair_id", "reaction_id"])
    spec_pairs = spec_pairs.merge(spec_samples.drop_duplicates("pair_id"), on="pair_id", how="left")
    spec_rp = pd.read_csv(ALLRINGS / "ring_pairs_ml.csv", low_memory=False,
                          usecols=["pair_id", "stage2_tier", "decision_reason"])
    spec_pairs = spec_pairs.merge(spec_rp, on="pair_id", how="left")

    # rename canonical losses: L_X = A_R - A_P (L_NICS uses opposite sign convention)
    # Predict CSV already stores Delta_HOMA = HOMA_R - HOMA_P (loss positive).
    # For NICS, predict CSV uses Delta_NICS_star = NICS_P - NICS_R (loss positive).
    complete = complete.rename(columns={
        "HOMA_pred_reactant": "A_HOMA_R", "HOMA_pred_product": "A_HOMA_P",
        "nMCBO_pred_reactant": "A_nMCBO_R", "nMCBO_pred_product": "A_nMCBO_P",
        "NICS_1zz_pred_reactant": "A_NICS_R", "NICS_1zz_pred_product": "A_NICS_P",
        "Delta_HOMA": "L_HOMA", "Delta_nMCBO": "L_nMCBO", "Delta_NICS_star": "L_NICS",
    })
    spec_pairs = spec_pairs.rename(columns={
        "HOMA_pred_reactant": "A_HOMA_R", "HOMA_pred_product": "A_HOMA_P",
        "nMCBO_pred_reactant": "A_nMCBO_R", "nMCBO_pred_product": "A_nMCBO_P",
        "NICS_1zz_pred_reactant": "A_NICS_R", "NICS_1zz_pred_product": "A_NICS_P",
        "Delta_HOMA": "L_HOMA", "Delta_nMCBO": "L_nMCBO", "Delta_NICS_star": "L_NICS",
    })

    # attach reaction_id + target ring metadata to complete
    rp_tierA = rp[rp["pair_id"].isin(complete["pair_id"])][[
        "pair_id", "reaction_id", "target_ring_size", "target_ring_map_numbers",
        "reactant_component_smiles", "product_component_smiles",
        "reactant_component_smiles_mapped", "product_component_smiles_mapped",
        "reactant_ring_atom_indices", "product_ring_atom_indices",
        "reactant_radius2_atom_indices", "product_radius2_atom_indices",
    ]].copy()
    complete = complete.merge(rp_tierA, on="pair_id", how="left")

    # spectator metadata
    spec_meta = pd.read_csv(ALLRINGS / "ring_pairs_ml.csv", low_memory=False,
                            usecols=["pair_id", "reaction_id", "target_ring_size",
                                     "target_ring_map_numbers",
                                     "reactant_component_smiles_mapped",
                                     "product_component_smiles_mapped",
                                     "reactant_ring_atom_indices",
                                     "product_ring_atom_indices"])
    spec_pairs = spec_pairs.merge(spec_meta, on="pair_id", how="left",
                                  suffixes=("", "_meta"))
    print(f"[load] target complete={len(complete)} spectator pairs={len(spec_pairs)}")
    return complete, spec_pairs


# ---------------------------------------------------------------- cluster bootstrap utilities
def compute_reaction_gap(target_df, spec_df, value_col, reaction_col="reaction_id"):
    """d_i = median(L_target | reaction_i) - median(L_spectator | reaction_i).

    Returns sorted reaction ids and the per-reaction gap vector d_i.
    """
    t_by_r = target_df.dropna(subset=[value_col]).groupby(reaction_col)[value_col].apply(list).to_dict()
    s_by_r = spec_df.dropna(subset=[value_col]).groupby(reaction_col)[value_col].apply(list).to_dict()
    common = sorted(set(t_by_r) & set(s_by_r))
    d_i = np.array([np.median(t_by_r[r]) - np.median(s_by_r[r]) for r in common])
    return common, d_i


def cluster_bootstrap_diff(target_df, spec_df, value_col, reaction_col="reaction_id",
                           n_boot=5000, seed=RANDOM_STATE):
    """Reaction-clustered bootstrap of the *median* of the per-reaction gap.

    d_i   = median(L_target|rxn_i) - median(L_spectator|rxn_i)
    d_hat = median_i(d_i)
    Bootstrap resamples the reaction units (cluster = reaction_id) and
    recomputes median(d_i*) each iteration; CI is the 2.5/97.5 percentile of
    that distribution, centred on d_hat.
    """
    rng = np.random.default_rng(seed)
    common, d_i = compute_reaction_gap(target_df, spec_df, value_col, reaction_col)
    if len(d_i) == 0:
        return None
    d_hat = float(np.median(d_i))
    boot = np.empty(n_boot)
    n = len(d_i)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.median(d_i[idx])
    return {
        "n_reactions": len(d_i),
        "d_hat_median": d_hat,
        "mean_d_i": float(np.mean(d_i)),
        "n_reactions_with_target_greater": int(np.sum(d_i > 0)),
        "frac_reactions_target_greater": float(np.mean(d_i > 0)),
        "boot_CI95_lo": float(np.percentile(boot, 2.5)),
        "boot_CI95_hi": float(np.percentile(boot, 97.5)),
        "frac_boot_positive": float((boot > 0).mean()),
    }


def cliffs_delta(x, y):
    """Cliff's delta (effect size), P(x>y) - P(x<y), via Mann-Whitney U rank sum.

    Standard formula (no tie correction):
        Ux = sum(rank_x) - nx*(nx+1)/2
        delta = 2*Ux/(nx*ny) - 1
    Range [-1, +1]. +1 when every x > every y.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    rx = ranks[:nx]
    Ux = rx.sum() - nx * (nx + 1) / 2.0
    delta = 2.0 * Ux / (nx * ny) - 1.0
    return float(delta)


# ---------------------------------------------------------------- ring family classification
def classify_ring_family(mol, target_idx):
    """Rule-based, returns (family, confidence, heteroatom_pattern, is_fused, neighbor_ring_types,
    classification_rule)."""
    if mol is None or not target_idx:
        return ("unknown", "low", "NA", False, "", "parse_failed")
    target = set(target_idx)
    size = len(target)
    syms = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in target)
    nC = syms.get("C", 0); nN = syms.get("N", 0); nO = syms.get("O", 0); nS = syms.get("S", 0)
    arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in target)
    hetero_pattern = "".join(sorted([f"{s}{n}" for s, n in syms.items() if s != "C" and n > 0]))

    sssr = list(Chem.GetSymmSSSR(mol))
    fused_6c = fused_5c = 0
    is_fused = False
    neighbor_ring_types = []
    for ring in sssr:
        rset = set(ring)
        if rset == target:
            continue
        if rset & target:
            is_fused = True
            syms_nb = Counter(mol.GetAtomWithIdx(a).GetSymbol() for a in ring)
            nb_size = len(ring)
            nb_pat = "".join(sorted([f"{s}{c}" for s, c in syms_nb.items() if s != "C" and c > 0]))
            nb_arom = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
            t = f"{nb_size}{'arom' if nb_arom else 'sp3'}-{nb_pat or 'C'}"
            neighbor_ring_types.append(t)
            if all(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in ring):
                if nb_size == 6: fused_6c += 1
                if nb_size == 5: fused_5c += 1

    has_N = nN > 0; has_O = nO > 0; has_S = nS > 0
    has_OH_on_ring = False
    has_NH2_on_ring = False
    for i in target:
        a = mol.GetAtomWithIdx(i)
        for nb in a.GetNeighbors():
            if nb.GetIdx() in target:
                continue
            sym = nb.GetSymbol()
            if sym == "O" and nb.GetTotalNumHs() > 0:
                has_OH_on_ring = True
            if sym == "N" and nb.GetTotalNumHs() > 0 and nb.GetFormalCharge() == 0:
                has_NH2_on_ring = True

    rule = ""
    conf = "low"

    if size == 6 and nC == 6 and not has_N and not has_O and not has_S:
        if is_fused and fused_6c >= 1:
            fam = "naphthol-type" if has_OH_on_ring else "naphthalene-type"
            rule = "6C-ring, fused, 6C neighbor"
        else:
            fam = "phenol-type" if has_OH_on_ring else "benzene-type"
            rule = "6C-ring, isolated, no heteroatom"
        conf = "high"
    elif size == 5 and nC == 4 and nN == 1 and not has_O and not has_S:
        fam = "indole-type" if is_fused else "pyrrole-type"
        rule = "5-ring, 4C+1N"
        conf = "high" if not (is_fused and nN > 1) else "low"
    elif size == 5 and nC == 4 and nO == 1 and not has_N and not has_S:
        fam = "benzofuran-type" if is_fused else "furan-type"
        rule = "5-ring, 4C+1O"
        conf = "high"
    elif size == 5 and nC == 4 and nS == 1 and not has_N and not has_O:
        fam = "benzothiophene-type" if is_fused else "thiophene-type"
        rule = "5-ring, 4C+1S"
        conf = "high"
    elif size == 6 and nC == 5 and nN == 1 and not has_O and not has_S:
        if is_fused:
            # N position relative to fusion decides quinoline vs isoquinoline
            n_atom_idx = [i for i in target if mol.GetAtomWithIdx(i).GetSymbol() == "N"][0]
            fusion_atoms = []
            for ring in sssr:
                rset = set(ring)
                if rset & target and rset != target:
                    fusion_atoms.extend(rset & target)
            fusions_set = set(fusion_atoms)
            # count bonds between N and fusion atoms within the target
            n_dist_to_fusion = min(
                len(Chem.GetShortestPath(mol, n_atom_idx, f)) if f != n_atom_idx else 0
                for f in fusions_set
            )
            fam = "isoquinoline-type" if n_dist_to_fusion == 1 else "quinoline-type"
            rule = f"6-ring, 5C+1N, fused, N-fusion_distance={n_dist_to_fusion}"
        else:
            fam = "pyridine-type"
            rule = "6-ring, 5C+1N, isolated"
        conf = "high"
    elif size in (5, 6) and has_N and not has_O and not has_S:
        fam = "other N-heteroarene"
        rule = f"{size}-ring, N-containing, pattern={hetero_pattern}"
        conf = "medium"
    elif size in (5, 6) and has_O and not has_N and not has_S:
        fam = "other O-heteroarene"
        rule = f"{size}-ring, O-containing"
        conf = "medium"
    elif size in (5, 6) and has_S and not has_N and not has_O:
        fam = "other S-heteroarene"
        rule = f"{size}-ring, S-containing"
        conf = "medium"
    elif nC == size:
        fam = "other carbocycle"
        rule = f"{size}-ring, all-C"
        conf = "medium" if size not in (5, 6) else "low"
    else:
        fam = "unknown"
        rule = f"{size}-ring, mixed hetero pattern={hetero_pattern}"
        conf = "low"
    return (fam, conf, hetero_pattern, is_fused, ";".join(neighbor_ring_types), rule)


def build_ring_family_assignment(complete):
    # original (unmapped) reaction SMILES per reaction_id, for lookup convenience
    tierA = pd.read_csv(ROOT / "dearom_stage2_full" / "tier_A_exact.csv",
                        low_memory=False, usecols=["reaction_id", "original_reaction"])
    rxn_smiles = dict(zip(tierA["reaction_id"].astype(str), tierA["original_reaction"]))
    rows = []
    fail = 0
    for _, r in complete.iterrows():
        mol = Chem.MolFromSmiles(r["reactant_component_smiles_mapped"])
        try:
            maps = set(int(x) for x in str(r["target_ring_map_numbers"]).replace(";", " ").split() if x.strip().isdigit())
            target_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() in maps] if mol else []
        except Exception:
            target_idx = []
        fam, conf, hp, is_fused, nb_types, rule = classify_ring_family(mol, target_idx)
        if fam == "unknown":
            fail += 1
        rows.append({
            "pair_id": r["pair_id"], "reaction_id": r["reaction_id"],
            "reaction_smiles": rxn_smiles.get(str(r["reaction_id"]), ""),
            "ring_family": fam, "ring_size": r["target_ring_size"],
            "heteroatom_pattern": hp, "is_fused": is_fused,
            "neighbor_ring_types": nb_types, "classification_rule": rule,
            "confidence": conf,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT["PANEL_D_RING_FAMILY"] / "ring_family_assignment.csv", index=False)
    print(f"[ring_family] n={len(out)} unknown={fail} confidence distribution: {out['confidence'].value_counts().to_dict()}")
    return out


# ---------------------------------------------------------------- Panel B (target vs spectator)
def panel_b(complete, spec_pairs, ring_fam):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, desc in zip(axes, DESC):
        # single source of truth for the per-reaction gap d_i and its median d_hat
        common_rxns, d_i = compute_reaction_gap(complete, spec_pairs, desc)
        t_by_r = complete.dropna(subset=[desc]).groupby("reaction_id")[desc].apply(list).to_dict()
        s_by_r = spec_pairs.dropna(subset=[desc]).groupby("reaction_id")[desc].apply(list).to_dict()
        target_v = np.array([float(np.median(t_by_r[r])) for r in common_rxns])
        spect_v = np.array([float(np.median(s_by_r[r])) for r in common_rxns])
        d_hat = float(np.median(d_i))
        cd = cliffs_delta(target_v, spect_v)
        boot = cluster_bootstrap_diff(
            complete, spec_pairs, desc, n_boot=5000)
        try:
            w = stats.wilcoxon(target_v, spect_v)
            p = float(w.pvalue)
        except Exception:
            p = float("nan")
        rows.append({
            "descriptor": desc,
            "n_reactions_with_both": len(common_rxns),
            "n_target_pairs": len(complete),
            "n_spectator_pairs_total_rows": len(spec_pairs),
            "n_spectator_pairs_nonnull": int(spec_pairs[desc].notna().sum()),
            "median_target": float(np.median(target_v)),
            "median_spectator": float(np.median(spect_v)),
            "d_hat_median_paired_diff": d_hat,
            "mean_paired_diff": float(np.mean(d_i)),
            "n_reactions_target_greater": boot["n_reactions_with_target_greater"],
            "frac_reactions_target_greater": boot["frac_reactions_target_greater"],
            "boot_CI95_lo": boot["boot_CI95_lo"], "boot_CI95_hi": boot["boot_CI95_hi"],
            "cliffs_delta": cd,
            "wilcoxon_p": p,
        })
        # paired dot / slope plot: left spectator, right target
        rng = np.random.default_rng(RANDOM_STATE)
        order = rng.permutation(len(d_i))
        sample_idx = order[: min(len(order), 1500)]
        for i in sample_idx:
            ax.plot([0, 1], [spect_v[i], target_v[i]], color="#cccccc", lw=0.4, alpha=0.5,
                    zorder=1)
        ax.scatter(np.zeros(len(sample_idx)) + rng.normal(0, 0.04, len(sample_idx)),
                   spect_v[sample_idx], s=4, alpha=0.55, color="#7f7f7f", zorder=2)
        ax.scatter(np.ones(len(sample_idx)) + rng.normal(0, 0.04, len(sample_idx)),
                   target_v[sample_idx], s=6, alpha=0.55, color=COLOR[desc], zorder=2)
        ax.boxplot([spect_v[sample_idx], target_v[sample_idx]],
                   positions=[0, 1], widths=0.3, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor="#333"),
                   medianprops=dict(color="#C44E52", lw=1.4),
                   showfliers=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Spectator\nring(s)", "Target\nring"])
        ax.set_ylabel(desc)
        ax.set_title(
            f"{desc}: target − spectator\nd̂ (median) = {d_hat:.3f}, "
            f"Cliff's δ = {cd:.3f}\n"
            f"{boot['n_reactions_with_target_greater']}/{len(common_rxns)} reactions with d_i > 0",
            fontsize=7.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Fig.6b — Is aromaticity loss localized to the target ring?",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_B_TARGET_SPECTATOR"] / f"Fig6b_target_vs_spectator.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(OUT["PANEL_B_TARGET_SPECTATOR"] / "fig6b_target_spectator_stats.csv", index=False)
    print("\n[panel B]\n", df.to_string(index=False))
    return df


# ---------------------------------------------------------------- Panel C (global loss)
def panel_c(complete):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, desc in zip(axes, DESC):
        x = complete[desc].astype(float).values
        # raincloud = half-violin + jittered dots + box
        ax.fill_betweenx(np.linspace(x.min(), x.max(), 200),
                         ax.get_xlim()[0], ax.get_xlim()[0],
                         alpha=0)
        # left half-violin (mirror)
        v = ax.violinplot([x], positions=[0], widths=0.7,
                          showextrema=False, vert=True)
        for b in v["bodies"]:
            xs_m, ys = b.get_paths()[0].vertices.T
            # mirror to left
            xs_m = -xs_m
            b.get_paths()[0].vertices = np.column_stack([xs_m + 0.0, ys])
            b.set_facecolor(COLOR[desc]); b.set_alpha(0.45)
            b.set_edgecolor("#333333")
        ax.set_xlim(-1.0, 1.5)
        # jittered dots
        rng = np.random.default_rng(RANDOM_STATE)
        sample = rng.choice(x, size=min(len(x), 1500), replace=False)
        ax.scatter(np.zeros(len(sample)) + rng.normal(0.55, 0.05, len(sample)),
                   sample, s=4, alpha=0.45, color=COLOR[desc])
        # box
        ax.boxplot([x], positions=[0.55], widths=0.18, vert=True,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor="#333"),
                   medianprops=dict(color="#C44E52", lw=1.4))
        ax.axhline(0, color="#666666", lw=1.0, ls="--", alpha=0.7)
        ax.set_xticks([0.55])
        ax.set_xticklabels([desc])
        ax.set_title(f"{desc}\nmedian = {np.median(x):.3f},  fraction >0 = {(x>0).mean()*100:.1f}%",
                      fontsize=7.5)
        ax.set_xlim(-0.9, 1.2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        q75, q25 = np.percentile(x, [75, 25])
        rows.append({"descriptor": desc,
                     "N": len(x),
                     "median": float(np.median(x)),
                     "IQR": float(q75 - q25),
                     "Q1": float(q25), "Q3": float(q75),
                     "mean": float(np.mean(x)),
                     "fraction_loss_positive": float((x > 0).mean())})
    fig.suptitle("Fig.6c — Global aromaticity-loss landscape (target rings, N=4377)",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_C_GLOBAL_LOSS"] / f"Fig6c_global_loss.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT["PANEL_C_GLOBAL_LOSS"] / "fig6c_global_loss_stats.csv", index=False)

    # stacked bar ALL_THREE_LOSS vs DISCORDANT vs ALL_GAIN
    pat = complete["aromaticity_pattern"].value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    cats = ["ALL_LOSS", "DISCORDANT", "ALL_GAIN_OR_NONLOSS"]
    fracs = [pat.get(c, 0.0) for c in cats]
    cols = ["#4C72B0", "#DD8452", "#C44E52"]
    left = 0
    for c, f, col in zip(cats, fracs, cols):
        ax.barh(0, f, left=left, color=col, height=0.45)
        ax.text(left + f / 2, 0, f"{f * 100:.1f}%", ha="center", va="center",
                fontsize=8.5, color="white", weight="bold")
        left += f
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_title("Descriptor concordance (N=4377)", fontsize=9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_C_GLOBAL_LOSS"] / f"Fig6c_concordance_bar.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("\n[panel C]\n", df.to_string(index=False))
    return df


# ---------------------------------------------------------------- Panel D (ring family)
def panel_d(complete, ring_fam):
    main_families = ["benzene-type", "phenol-type", "naphthalene-type", "naphthol-type",
                     "pyrrole-type", "indole-type", "furan-type", "benzofuran-type",
                     "thiophene-type", "benzothiophene-type",
                     "pyridine-type", "quinoline-type", "isoquinoline-type",
                     "other N-heteroarene", "other O-heteroarene",
                     "other S-heteroarene", "other carbocycle"]
    df = complete.merge(ring_fam[["pair_id", "ring_family", "confidence", "is_fused",
                                  "heteroatom_pattern"]], on="pair_id", how="left")

    # D1 raw distributions
    fam_keep = df.groupby("ring_family").filter(lambda g: len(g) >= 30)["ring_family"].unique().tolist()
    fam_keep = [f for f in main_families if f in fam_keep]
    if not fam_keep:
        fam_keep = df["ring_family"].value_counts().head(8).index.tolist()

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
    for ax, desc, color in zip(axes, DESC, [COLOR[d] for d in DESC]):
        data = [df.loc[df["ring_family"] == f, desc].dropna().values for f in fam_keep]
        vp = ax.violinplot(data, positions=range(len(fam_keep)),
                           widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(color); b.set_alpha(0.45); b.set_edgecolor("#333")
        bp = ax.boxplot(data, positions=range(len(fam_keep)), widths=0.18,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor="white", edgecolor="#333"),
                        medianprops=dict(color="#C44E52", lw=1.4))
        rng = np.random.default_rng(RANDOM_STATE)
        for i, vals in enumerate(data):
            sub = rng.choice(vals, size=min(len(vals), 200), replace=False)
            ax.scatter(np.zeros(len(sub)) + i + rng.normal(0, 0.06, len(sub)),
                       sub, s=3, alpha=0.4, color=color)
        ax.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.7)
        ax.set_xticks(range(len(fam_keep)))
        ax.set_xticklabels(fam_keep, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(desc)
        ax.set_title(f"{desc} — distribution per ring family (n ≥ 30)",
                     fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Fig.6d1 — Ring-family-dependent aromaticity response (raw units)",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_D_RING_FAMILY"] / f"Fig6d1_raw_family_distributions.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # D2 z-robust heatmap (rows = families, cols = descriptors)
    rows = []
    for fam in fam_keep:
        sub = df[df["ring_family"] == fam]
        rec = {"ring_family": fam, "n": len(sub)}
        for desc in DESC:
            rec[f"median_{desc}"] = float(sub[desc].median()) if len(sub) else float("nan")
        rows.append(rec)
    fam_summary = pd.DataFrame(rows)

    z_rows = []
    global_med = {d: float(np.median(df[d])) for d in DESC}
    global_mad = {d: float(1.4826 * np.median(np.abs(df[d] - global_med[d]))) for d in DESC}
    for _, r in fam_summary.iterrows():
        z = {"ring_family": r["ring_family"], "n": int(r["n"])}
        for d in DESC:
            m = r[f"median_{d}"]
            z[f"z_{d}"] = (m - global_med[d]) / global_mad[d] if global_mad[d] > 0 else 0.0
        z_rows.append(z)
    zr = pd.DataFrame(z_rows)

    # Heatmap
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    M = zr[["z_L_HOMA", "z_L_nMCBO", "z_L_NICS"]].values
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["L_HOMA", "L_nMCBO", "L_NICS"], fontsize=8)
    ax.set_yticks(range(len(zr))); ax.set_yticklabels(
        [f"{r}  (n={n})" for r, n in zip(zr["ring_family"], zr["n"])], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=("white" if abs(M[i, j]) > vmax * 0.55 else "black"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("z_robust = (median − global_median) / (1.4826·global_MAD)", fontsize=7.5)
    ax.set_title("Fig.6d2 — Family multidimensional response fingerprint\n"
                 "(>0 = above-average aromaticity loss)", fontsize=8.5)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_D_RING_FAMILY"] / f"Fig6d2_family_fingerprint_heatmap.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    fam_summary.to_csv(OUT["PANEL_D_RING_FAMILY"] / "ring_family_descriptor_summary.csv", index=False)
    zr.to_csv(OUT["PANEL_D_RING_FAMILY"] / "ring_family_z_robust.csv", index=False)

    # contact sheets per family
    contact_dir = OUT["PANEL_D_RING_FAMILY"] / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)
    for fam in fam_keep:
        sub = df[df["ring_family"] == fam].sample(min(len(df[df["ring_family"] == fam]), 24),
                                                  random_state=RANDOM_STATE)
        mols = []
        for smi in sub["reactant_component_smiles"]:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                mols.append(m)
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(220, 160),
                                       legends=[fam] * len(mols))
            img.save(contact_dir / f"family_{fam.replace('/', '_')}.png")
    print("[panel D] families in main fig:", fam_keep, "n samples", len(df))
    return df, fam_summary, zr


# ---------------------------------------------------------------- Panel E (Structural change)
def panel_e(complete):
    def fp(s):
        m = Chem.MolFromSmiles(s) if pd.notna(s) else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    rows = []
    for _, r in complete.iterrows():
        fr = fp(r["reactant_component_smiles"])
        fp_ = fp(r["product_component_smiles"])
        if fr is None or fp_ is None:
            rows.append({"pair_id": r["pair_id"], "Tanimoto": np.nan,
                         "StructuralChange": np.nan})
            continue
        t = DataStructs.TanimotoSimilarity(fr, fp_)
        rows.append({"pair_id": r["pair_id"], "Tanimoto": float(t),
                     "StructuralChange": float(1 - t)})
    sc_df = pd.DataFrame(rows)
    full = complete.merge(sc_df, on="pair_id", how="left")
    full.to_csv(OUT["PANEL_E_STRUCTURAL_CHANGE"] / "structural_change_vs_loss.csv", index=False)

    # Try LOWESS (statsmodels if installed, else linear)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        HAVE_LOWESS = True
    except ImportError:
        HAVE_LOWESS = False

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    rows_stats = []
    for ax, desc in zip(axes, DESC):
        sub = full.dropna(subset=["StructuralChange", desc]).copy()
        x = sub["StructuralChange"].values; y = sub[desc].values
        hb = ax.hexbin(x, y, gridsize=50, cmap="Blues", bins="log", mincnt=1)
        if HAVE_LOWESS:
            try:
                smoothed = lowess(y, x, frac=0.3, return_sorted=True)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color="#C44E52", lw=1.8,
                        label="LOWESS")
            except Exception:
                pass
        ax.axhline(0, color="#666", lw=0.8, ls="--")
        ax.set_xlabel("Structural change  (1 − Tanimoto R/P)")
        ax.set_ylabel(desc)
        rp = stats.pearsonr(x, y)[0]
        rs = stats.spearmanr(x, y)[0]
        # bootstrap pearson CI
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.integers(0, len(x), size=(3000, len(x)))
        rr = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
        lo, hi = np.percentile(rr, [2.5, 97.5])
        ax.set_title(f"{desc}: Pearson r = {rp:.3f} (95% CI [{lo:.3f}, {hi:.3f}])\n"
                     f"Spearman ρ = {rs:.3f}", fontsize=7.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        rows_stats.append({"descriptor": desc, "N": int(len(x)),
                           "pearson_r": float(rp), "spearman_rho": float(rs),
                           "pearson_CI95_lo": float(lo), "pearson_CI95_hi": float(hi)})
        fig.colorbar(hb, ax=ax, fraction=0.046)
    fig.suptitle("Fig.6e — Global structural perturbation is a poor proxy for local aromaticity loss",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_E_STRUCTURAL_CHANGE"] / f"Fig6e_structural_decoupling.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows_stats).to_csv(
        OUT["PANEL_E_STRUCTURAL_CHANGE"] / "fig6e_stats.csv", index=False)
    print("\n[panel E]\n", pd.DataFrame(rows_stats).to_string(index=False))
    return full, rows_stats


# ---------------------------------------------------------------- Panel F (discordant)
def panel_f(complete, ring_fam):
    disc = complete[complete["aromaticity_pattern"] == "DISCORDANT"].copy()
    # enrichment against background
    bg = complete
    rows = []
    for fam in ring_fam["ring_family"].value_counts().index:
        n_d = int((disc["pair_id"].isin(ring_fam[ring_fam["ring_family"] == fam]["pair_id"])).sum())
        n_b = int((ring_fam["ring_family"] == fam).sum())
        if n_b == 0:
            continue
        # Fisher exact: 2x2 [n_d in fam, n_d not in fam] vs [n_b in fam, n_b not in fam]
        # actually use simpler: odds ratio with Haldane correction
        a = n_d; b = len(disc) - n_d
        c = n_b - n_d; d = len(bg) - n_b - (len(disc) - n_d)
        # odds = (a*d)/(b*c) with +0.5
        if min(b, c) == 0:
            or_ = float("inf"); p = float("nan")
        else:
            or_ = (a * d) / (b * c)
            try:
                _, p = stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                p = float("nan")
        rows.append({"variable": "ring_family", "category": fam,
                     "n_disc": n_d, "n_total": n_b,
                     "frac_disc": n_d / max(n_b, 1),
                     "frac_background": len(disc) / len(bg),
                     "odds_ratio": or_, "fisher_p": p})
    # fused
    rf = ring_fam.copy()
    for val in [True, False]:
        n_d = int((disc["pair_id"].isin(rf[rf["is_fused"] == val]["pair_id"])).sum())
        n_b = int((rf["is_fused"] == val).sum())
        a = n_d; b = len(disc) - n_d
        c = n_b - n_d; d = len(bg) - n_b - (len(disc) - n_d)
        if min(b, c) == 0:
            or_ = float("inf"); p = float("nan")
        else:
            or_ = (a * d) / (b * c)
            try:
                _, p = stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                p = float("nan")
        rows.append({"variable": "is_fused", "category": str(val),
                     "n_disc": n_d, "n_total": n_b,
                     "frac_disc": n_d / max(n_b, 1),
                     "frac_background": len(disc) / len(bg),
                     "odds_ratio": or_, "fisher_p": p})
    # ring_size
    for sz in [5, 6, 7]:
        n_d = int(disc.merge(ring_fam[ring_fam["ring_size"] == sz], on="pair_id").shape[0])
        n_b = int((ring_fam["ring_size"] == sz).sum())
        a = n_d; b = len(disc) - n_d
        c = n_b - n_d; d = len(bg) - n_b - (len(disc) - n_d)
        if min(b, c) == 0:
            or_ = float("inf"); p = float("nan")
        else:
            or_ = (a * d) / (b * c)
            try:
                _, p = stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                p = float("nan")
        rows.append({"variable": "ring_size", "category": str(sz),
                     "n_disc": n_d, "n_total": n_b,
                     "frac_disc": n_d / max(n_b, 1),
                     "frac_background": len(disc) / len(bg),
                     "odds_ratio": or_, "fisher_p": p})
    enr = pd.DataFrame(rows)
    # BH FDR
    if "fisher_p" in enr.columns:
        valid = enr["fisher_p"].notna()
        p = enr.loc[valid, "fisher_p"].values
        if len(p):
            order = np.argsort(p)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(p) + 1)
            fdr = p * len(p) / ranks
            fdr = np.minimum.accumulate(fdr[order[::-1]])[::-1]
            fdr_full = np.full(len(enr), np.nan)
            fdr_full[valid.values] = fdr
            enr["fdr_BH"] = np.clip(fdr_full, 0, 1)
    enr.to_csv(OUT["PANEL_F_DISCORDANT"] / "discordant_enrichment.csv", index=False)

    # discordance strength = max stdz desc - min stdz desc
    X = complete[DESC].astype(float)
    mu = X.mean(); sd = X.std()
    disc_z = disc.copy()
    for d in DESC:
        disc_z[f"z_{d}"] = (disc_z[d] - mu[d]) / sd[d]
    zcols = [f"z_{d}" for d in DESC]
    disc_z["discordance_strength"] = disc_z[zcols].max(axis=1) - disc_z[zcols].min(axis=1)
    disc_z["discordance_strength_std"] = disc_z[zcols].std(axis=1)
    disc_z.to_csv(OUT["PANEL_F_DISCORDANT"] / "discordant_cases_annotated.csv", index=False)

    top = disc_z.sort_values("discordance_strength", ascending=False).head(6)
    top.to_csv(OUT["PANEL_F_DISCORDANT"] / "discordant_top_cases.csv", index=False)

    # render top-6 R/P pairs
    fig, axes = plt.subplots(6, 2, figsize=(7.2, 18.0))
    for i, (_, r) in enumerate(top.iterrows()):
        mr = Chem.MolFromSmiles(r["reactant_component_smiles"])
        mp = Chem.MolFromSmiles(r["product_component_smiles"])
        rdDepictor.Compute2DCoords(mr); rdDepictor.Compute2DCoords(mp)
        axr, axp = axes[i, 0], axes[i, 1]
        img_r = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(220, 180)
        img_r.drawOptions().addAtomIndices = False
        img_r.DrawMolecule(mr); img_r.FinishDrawing()
        img_p = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(220, 180)
        img_p.drawOptions().addAtomIndices = False
        img_p.DrawMolecule(mp); img_p.FinishDrawing()
        import io
        from PIL import Image
        axr.imshow(Image.open(io.BytesIO(img_r.GetDrawingText())))
        axp.imshow(Image.open(io.BytesIO(img_p.GetDrawingText())))
        axr.axis("off"); axp.axis("off")
        axr.set_title(f"Reactant  ({r['pair_id']})", fontsize=7)
        axp.set_title(f"Product", fontsize=7)
        if i == 0:
            axr.set_xlabel(f"L_HOMA={r['L_HOMA']:.2f}  L_nMCBO={r['L_nMCBO']:.3f}  L_NICS={r['L_NICS']:.1f}", fontsize=7)
    fig.suptitle("Fig.6f-A — Top-6 descriptor-discordant cases (R→P)", fontsize=10, y=0.995)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["PANEL_F_DISCORDANT"] / f"Fig6fA_discordant_cases.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    # discordance type breakdown
    def dtype(row):
        neg = [d for d in DESC if row[d] <= 0]
        return ",".join(d.replace("L_", "") for d in neg) + "_NEG"
    disc["discordance_type"] = disc.apply(dtype, axis=1)
    disc["discordance_type"].value_counts().to_csv(
        OUT["PANEL_F_DISCORDANT"] / "discordance_type_counts.csv")
    print("\n[panel F] n_discordant =", len(disc))
    print(disc["discordance_type"].value_counts())
    return enr, disc_z


# ---------------------------------------------------------------- Exploratory
def exploratory(complete):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, d, r_col in zip(axes, DESC,
                            [("A_HOMA_R", "L_HOMA"), ("A_nMCBO_R", "L_nMCBO"),
                             ("A_NICS_R", "L_NICS")]):
        sub = complete[[r_col[0], r_col[1]]].dropna()
        x = sub[r_col[0]].astype(float); y = sub[r_col[1]].astype(float)
        ax.hexbin(x, y, gridsize=50, cmap="Blues", bins="log", mincnt=1)
        rp = stats.pearsonr(x, y)[0]
        ax.set_xlabel(f"reactant {r_col[0]}"); ax.set_ylabel(d)
        ax.set_title(f"{d} vs reactant aromaticity: r = {rp:.3f}", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        rows.append({"descriptor": d, "pearson_R_vs_L": float(rp)})
    fig.suptitle("Initial aromaticity vs aromaticity loss", fontsize=10, y=1.04)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["SI"] / f"S11_initial_aromaticity_vs_loss.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT["SI"] / "initial_aromaticity_vs_loss.csv", index=False)

    # PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(complete[DESC].fillna(0).values)
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    sc = pca.transform(X)
    pca_df = pd.DataFrame(sc[:, :3], columns=["PC1", "PC2", "PC3"])
    pca_df.insert(0, "pair_id", complete["pair_id"].values)
    pca_df["aromaticity_pattern"] = complete["aromaticity_pattern"].values
    pca_df.to_csv(OUT["SI"] / "descriptor_loss_pca_scores.csv", index=False)
    pd.DataFrame(pca.components_.T, index=DESC, columns=[f"PC{i+1}" for i in range(3)]
                 ).to_csv(OUT["SI"] / "descriptor_loss_pca_loadings.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    is_disc = (complete["aromaticity_pattern"].values == "DISCORDANT")
    ax.scatter(sc[~is_disc, 0], sc[~is_disc, 1], s=4, alpha=0.3, color="#9aa5ad",
               label="ALL_LOSS", rasterized=True)
    ax.scatter(sc[is_disc, 0], sc[is_disc, 1], s=14, alpha=0.9, color="#C44E52",
               marker="x", label="DISCORDANT")
    scale = max(np.abs(sc[:, 0]).max(), np.abs(sc[:, 1]).max()) * 1.1
    for k, name in enumerate(DESC):
        lx, ly = pca.components_[0, k], pca.components_[1, k]
        ax.arrow(0, 0, lx * scale, ly * scale, color="#2b3f5f", width=0.004,
                 head_width=0.05, length_includes_head=True)
        ax.text(lx * scale * 1.08, ly * scale * 1.08,
                name.replace("L_", ""), color="#2b3f5f", fontsize=8)
    ax.axhline(0, color="#cccccc", lw=0.6); ax.axvline(0, color="#cccccc", lw=0.6)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("PCA of standardized L vector", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT["SI"] / f"S12_pca_loss_vector.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    (OUT["SI"] / "pca_loss_explained_variance.json").write_text(json.dumps({
        "PC1": float(evr[0]), "PC2": float(evr[1]), "PC3": float(evr[2])}, indent=2))
    print("[exploratory] PCA evr:", evr)


# ---------------------------------------------------------------- main
def main():
    complete, spec_pairs = load_data()
    ring_fam = build_ring_family_assignment(complete)
    panel_b(complete, spec_pairs, ring_fam)
    panel_c(complete)
    panel_d(complete, ring_fam)
    panel_e(complete)
    panel_f(complete, ring_fam)
    exploratory(complete)
    print("ALL FIG6 MAIN PANELS DONE")


if __name__ == "__main__":
    main()
