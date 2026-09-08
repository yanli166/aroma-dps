#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.5a workflow schematic + SI supplementary figures (S1/S2/S6/S7/S8).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
DATA = ROOT / "dearom_ring_pairs_A_tierA"
OUT = ROOT / "fig5_analysis"
MAIN = OUT / "09_fig5_main"
SI = OUT / "10_fig5_si"
DISC = OUT / "05_discordant"
MASTER = OUT / "01_master"
STAT = OUT / "08_statistics"
for d in (MAIN, SI):
    d.mkdir(parents=True, exist_ok=True)

DESC = ["Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star"]


def save(fig, name, main=True):
    d = MAIN if main else SI
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_idx(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return [int(x) for x in str(s).replace(",", ";").split(";") if x.strip()]


def mol_image(smi, highlight_idx, size=(180, 160), caption=""):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    d.drawOptions().addAtomIndices = False
    opts = d.drawOptions()
    opts.bondLineWidth = 2
    # highlight ring with same color
    colors = [(0.75, 0.85, 1.0)] * len(highlight_idx)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=highlight_idx, highlightAtomColors=colors,
        highlightBonds=None)
    d.FinishDrawing()
    import io
    from PIL import Image
    png = d.GetDrawingText()
    img = Image.open(io.BytesIO(png))
    return img


def fig5a():
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axis("off")
    stages = [
        ("USPTO_STEREO\n1,002,915 reactions", "#dde7f0"),
        ("Fast structural screening\n(Stage-1)", "#cfe3f3"),
        ("Exact atom-mapped\nStage-2 validation", "#bfd8ec"),
        ("4,306 high-confidence\nreactions", "#aed0e5"),
        ("4,443 target-ring\npairs", "#9cc7de"),
        ("ML aromaticity\nprediction", "#8abdd7"),
        ("4,377 complete\nR/P pairs", "#78b3d0"),
        ("ΔA vector\n(ΔHOMA, ΔnMCBO, ΔNICS*)", "#66a8c8"),
    ]
    xs = np.linspace(0.03, 0.97, len(stages))
    for (txt, col), x in zip(stages, xs):
        b = FancyBboxPatch((x - 0.058, 0.52), 0.116, 0.30,
                           boxstyle="round,pad=0.01", fc=col, ec="#4a6a8a", lw=1.2)
        ax.add_patch(b)
        ax.text(x, 0.67, txt, ha="center", va="center", fontsize=7.2, color="#1a2733",
                linespacing=1.3)
        if x < xs[-1]:
            ar = FancyArrowPatch((x + 0.058, 0.67), (x + 0.118, 0.67),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="#4a6a8a", lw=1.4)
            ax.add_patch(ar)
    # legend note
    ax.text(0.5, 0.16, "46 incomplete pairs excluded (one side lacked a valid model prediction)",
            ha="center", fontsize=7, style="italic", color="#555555")

    # representative reaction on the right side (same atom-mapped ring highlighted)
    # choose a simple, interpretable example
    master = pd.read_csv(MASTER / "reaction_aromaticity_fig5_master.csv", low_memory=False)
    row = master.iloc[0]
    rmap = set(int(x) for x in str(row["target_ring_map_numbers"]).replace(";", " ").split() if x.strip().isdigit())
    rmol = Chem.MolFromSmiles(row["reactant_component_smiles_mapped"])
    pmol = Chem.MolFromSmiles(row["product_component_smiles_mapped"])
    r_idx = [a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum() in rmap]
    p_idx = [a.GetIdx() for a in pmol.GetAtoms() if a.GetAtomMapNum() in rmap]
    img_r = mol_image(row["reactant_component_smiles"], r_idx)
    img_p = mol_image(row["product_component_smiles"], p_idx)

    # left inset: reaction
    ax2 = fig.add_axes([0.03, -0.02, 0.5, 0.5])
    ax2.axis("off")
    if img_r is not None and img_p is not None:
        # draw both
        axr = fig.add_axes([0.05, -0.10, 0.24, 0.55]); axr.axis("off")
        axr.imshow(img_r)
        axr.set_title("Reactant\n(aromatic target ring)", fontsize=6.5, color="#1a2733")
        axp = fig.add_axes([0.31, -0.10, 0.24, 0.55]); axp.axis("off")
        axp.imshow(img_p)
        axp.set_title("Product\n(dearomatized target ring)", fontsize=6.5, color="#1a2733")
        ax2.text(0.0, 0.0, "", fontsize=6)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.55, 0.62, "→", fontsize=18, ha="center", va="center", color="#4a6a8a")
    ax2.text(0.58, 0.28, "same atom-mapped ring", fontsize=7, color="#C44E52",
             ha="center")
    fig.tight_layout()
    save(fig, "Fig5a_workflow")


def fig5a_v2():
    """Simpler, self-contained schematic without RDKit-drawn molecules (robust)."""
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.axis("off")
    stages = [
        ("USPTO_STEREO\n1,002,915 reactions", "#dde7f0"),
        ("Fast structural\nscreening (Stage-1)", "#cfe3f3"),
        ("Exact atom-mapped\nStage-2 validation", "#bfd8ec"),
        ("4,306 high-confidence\nreactions", "#aed0e5"),
        ("4,443 target-ring\npairs", "#9cc7de"),
        ("ML aromaticity\nprediction", "#8abdd7"),
        ("4,377 complete\nR/P pairs", "#78b3d0"),
        ("ΔA vector\n(ΔHOMA, ΔnMCBO, ΔNICS*)", "#66a8c8"),
    ]
    xs = np.linspace(0.035, 0.965, len(stages))
    for (txt, col), x in zip(stages, xs):
        b = FancyBboxPatch((x - 0.056, 0.42), 0.112, 0.34,
                           boxstyle="round,pad=0.01", fc=col, ec="#4a6a8a", lw=1.2)
        ax.add_patch(b)
        ax.text(x, 0.59, txt, ha="center", va="center", fontsize=7.0, color="#1a2733",
                linespacing=1.3)
        if x < xs[-1]:
            ar = FancyArrowPatch((x + 0.056, 0.59), (x + 0.112, 0.59),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#4a6a8a", lw=1.4)
            ax.add_patch(ar)
    ax.text(0.5, 0.10, "46 incomplete pairs excluded (one side lacked a valid model prediction)",
            ha="center", fontsize=7, style="italic", color="#555555")
    ax.text(0.5, 0.0,
            "representative reaction: aromatic target ring → dearomatized target ring (same atom-mapped ring)",
            ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout()
    save(fig, "Fig5a_workflow")


def si_figures():
    # S1: full R/P distributions already in Fig5b (keep separate SI copy)
    master = pd.read_csv(MASTER / "reaction_aromaticity_fig5_master.csv", low_memory=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, name, rcol, pcol in zip(axes,
                                    ["HOMA", "nMCBO", "NICS(1)zz"],
                                    ["HOMA_pred_reactant", "nMCBO_pred_reactant", "NICS_1zz_pred_reactant"],
                                    ["HOMA_pred_product", "nMCBO_pred_product", "NICS_1zz_pred_product"]):
        ax.hist(master[rcol], bins=80, alpha=0.55, density=True, color="#4C72B0",
                label="Reactant")
        ax.hist(master[pcol], bins=80, alpha=0.55, density=True, color="#DD8452",
                label="Product")
        ax.set_xlabel(name); ax.set_ylabel("density")
        ax.legend(fontsize=6.5, frameon=False)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("S1: Reactant vs Product distributions", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "S1_reactant_product_distributions", main=False)

    # S2: delta histograms with KDE
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, c in zip(axes, DESC):
        x = master[c].astype(float).dropna()
        ax.hist(x, bins=80, density=True, alpha=0.5, color="#4C72B0")
        try:
            k = gaussian_kde(x)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, k(xs), color="#2b3f5f", lw=1.3)
        except Exception:
            pass
        ax.axvline(0, color="#C44E52", lw=1.2)
        ax.set_xlabel(c); ax.set_ylabel("density")
        ax.set_title(f"{c}   >0: {(x>0).mean()*100:.1f}%", fontsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("S2: Delta distributions with KDE", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "S2_delta_histograms", main=False)

    # S7: prediction failure / 46 incomplete pairs summary
    pred = pd.read_csv(DATA / "ring_property_predictions.csv", low_memory=False)
    n_err = (pred["prediction_status"] == "ERROR").sum()
    incomplete = []
    for pid, g in pred.groupby("pair_id"):
        if (g["prediction_status"] == "PASS").sum() < 2:
            incomplete.append(pid)
    master_ids = set(master["pair_id"])
    pd.DataFrame({"pair_id": incomplete,
                  "in_master_4377": [p in master_ids for p in incomplete]}
                 ).to_csv(STAT / "incomplete_pairs_excluded.csv", index=False)
    print(f"[S7] prediction errors={n_err}, incomplete pairs={len(incomplete)}")
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.axis("off")
    txt = (f"Prediction failure summary (S7)\n\n"
           f"total sample predictions: {len(pred)}\n"
           f"failed (UFF/charge/special S): {n_err}\n"
           f"incomplete R/P pairs excluded: {len(incomplete)}\n"
           f"pairs remaining in Fig.5 corpus: {len(master)}")
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9, family="monospace")
    fig.tight_layout()
    save(fig, "S7_prediction_failure_summary", main=False)

    # S6: top-50 discordant structures - render grid of 50 mols
    top = pd.read_csv(DISC / "top_50_descriptor_discordant_cases.csv", low_memory=False)
    mols = []
    legends = []
    for _, r in top.iterrows():
        mols.append(Chem.MolFromSmiles(r["Reactant_SMILES"]))
        legends.append(r["discordance_type"][:18])
    img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(260, 180),
                               legends=legends)
    img.save(SI / "S6_top50_discordant_structures.png")
    print("[S6] top-50 discordant grid saved")

    # S8: ring family counts bar
    cnt = pd.read_csv(OUT / "07_ring_family/ring_family_counts.csv")
    cnt = cnt.sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(cnt["ring_family"], cnt["count"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("n target-ring pairs")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "S8_ring_family_counts", main=False)


def main():
    fig5a_v2()
    si_figures()
    print("Fig5a + SI done")


if __name__ == "__main__":
    main()
