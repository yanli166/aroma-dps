#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6 v2 — Montage: Fig6_CORE_candidate and Fig6_ENHANCED_candidate.

CORE (6 panels):
  a) pipeline
  b) target vs spectator
  c) global loss raincloud
  d) ring-family z_robust heatmap
  e) structural decoupling
  f) ring-family explanatory strength (CV R2 + eps2)
  (discordant top-6 cases moved to SI)

ENHANCED (6 panels):
  a) pipeline
  b) target vs spectator
  c) global loss raincloud
  d) Ring × Reaction z_robust heatmap
  e) controlled slices
  f) variance decomposition
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
V2 = ROOT / "fig6_analysis_v2"

PA = V2 / "PANEL_A_PIPELINE" / "Fig6a_pipeline.png"
PB = V2 / "PANEL_B_TARGET_SPECTATOR" / "Fig6b_target_vs_spectator.png"
PC = V2 / "PANEL_C_GLOBAL_LOSS" / "Fig6c_global_loss.png"
PD1 = V2 / "PANEL_D_RING_FAMILY" / "Fig6d2_family_fingerprint_heatmap.png"
PE = V2 / "PANEL_E_STRUCTURAL_CHANGE" / "Fig6e_structural_decoupling.png"
PF = V2 / "PANEL_F_DISCORDANT" / "Fig6f_ring_family_cv_r2.png"

PE_RX = V2 / "VARIANCE_DECOMPOSITION_OPTIONAL" / "Fig6_RingReaction_zrobust_heatmap.png"
PE_SLICE = V2 / "VARIANCE_DECOMPOSITION_OPTIONAL" / "Fig6_slice_A_fixed_reaction.png"
PE_VARDEC = V2 / "VARIANCE_DECOMPOSITION_OPTIONAL" / "Fig6fB_variance_decomposition.png"


def load_img(p):
    if not p.exists():
        print("missing", p); return None
    return Image.open(p)


def make_montage(panels_with_labels, out_path, figsize):
    fig = plt.figure(figsize=figsize)
    n = len(panels_with_labels)
    for i, (img, label) in enumerate(panels_with_labels):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=60, va="center",
                      weight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(h_pad=1.0)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    core = [
        (load_img(PA),  "a"),
        (load_img(PB),  "b"),
        (load_img(PC),  "c"),
        (load_img(PD1), "d"),
        (load_img(PE),  "e"),
        (load_img(PF),  "f"),
    ]
    make_montage(core, V2 / "Fig6_CORE_candidate", figsize=(11.0, 36.0))

    enhanced = [
        (load_img(PA),        "a"),
        (load_img(PB),        "b"),
        (load_img(PC),        "c"),
        (load_img(PE_RX),     "d"),
        (load_img(PE_SLICE),  "e"),
        (load_img(PE_VARDEC), "f"),
    ]
    make_montage(enhanced, V2 / "Fig6_ENHANCED_candidate", figsize=(11.0, 36.0))
    print("montages written")


if __name__ == "__main__":
    main()
