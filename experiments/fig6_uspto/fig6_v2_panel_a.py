#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Panel A pipeline schematic (SVG / PDF / PNG 600 dpi).
Plus a paired .json with the numeric anchors used in the figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
OUT = ROOT / "fig6_analysis_v2" / "PANEL_A_PIPELINE"
OUT.mkdir(parents=True, exist_ok=True)

ANCHORS = {
    "USPTO_STEREO_reactions": 1002915,
    "Tier_A_reactions": 4306,
    "Tier_A_target_ring_pairs": 4443,
    "complete_RP_pairs": 4377,
    "spectator_rings_predicted": 5592,
    "spectator_rings_input": 5722,
    "all_ring_pairs": 10221,
}

(OUT / "anchors.json").write_text(json.dumps(ANCHORS, indent=2))


def make_fig():
    fig, ax = plt.subplots(figsize=(11.6, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    stages_top = [
        ("USPTO_STEREO\n1,002,915 reactions", "#dde7f0"),
        ("High-recall\nring-aware screening", "#cfe3f3"),
        ("Atom-mapped\ntarget-ring tracking", "#bfd8ec"),
        ("4,306 Tier-A\nreactions", "#aed0e5"),
        ("4,443 target-ring\npairs", "#9cc7de"),
        ("Frozen RC-GNN\nprediction (3 tasks)", "#8abdd7"),
        ("4,377 complete\ntarget-ring pairs", "#78b3d0"),
        ("Multidimensional\nL = (L_HOMA, L_nMCBO, L_NICS)", "#66a8c8"),
    ]
    n = len(stages_top)
    xs = np.linspace(0.04, 0.96, n)
    for (txt, col), x in zip(stages_top, xs):
        b = FancyBboxPatch((x - 0.057, 0.56), 0.114, 0.32,
                           boxstyle="round,pad=0.012", fc=col, ec="#4a6a8a", lw=1.2)
        ax.add_patch(b)
        ax.text(x, 0.72, txt, ha="center", va="center", fontsize=7.5,
                color="#1a2733", linespacing=1.25)
        if x < xs[-1]:
            ar = FancyArrowPatch((x + 0.057, 0.72), (x + 0.113, 0.72),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#4a6a8a", lw=1.4)
            ax.add_patch(ar)

    # Bottom branch: all rings branch
    ax.text(0.18, 0.40, "5,722 spectator-ring\ncandidates (R + B + C tiers)",
            ha="center", va="center", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0e4d8",
                      ec="#a07f5b", lw=1.0))
    ax.text(0.42, 0.40, "frozen RC-GNN prediction\n(internal control dataset)",
            ha="center", va="center", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0e4d8",
                      ec="#a07f5b", lw=1.0))
    ax.text(0.66, 0.40, "5,592 complete spectator\nR/P pairs (re-predicted)",
            ha="center", va="center", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0e4d8",
                      ec="#a07f5b", lw=1.0))
    # Connectors from top stage 4 (4,306 Tier-A) down to all-rings branch
    for x_top, x_bot in [(xs[3], 0.18), (xs[3], 0.42), (0.42, 0.66)]:
        ar = FancyArrowPatch((x_top, 0.56), (x_bot, 0.48),
                             arrowstyle="-|>", mutation_scale=10,
                             color="#a07f5b", lw=1.0,
                             connectionstyle="arc3,rad=0.18")
        ax.add_patch(ar)

    # Legend
    ax.text(0.5, 0.10, "orange branch = internal control dataset (spectator rings)",
            ha="center", fontsize=7.0, color="#a07f5b", style="italic")
    ax.text(0.5, 0.02, "ML enables systematic, reaction-scale aromaticity analysis that would be "
            "computationally burdensome by routine quantum-chemical workflows.",
            ha="center", fontsize=7.0, color="#444444", style="italic")
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"Fig6a_pipeline.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Fig6a_pipeline saved")


if __name__ == "__main__":
    make_fig()
