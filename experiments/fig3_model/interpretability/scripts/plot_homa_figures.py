# -*- coding: utf-8 -*-
"""
Step 6: HOMA merged 重训模型性能图 (固定 collet a/b test)
  Figure 1: HOMA test parity (密度散点, 无拟合线)
  Figure 2: 旧最优模型 vs Merged 重训 在同一 test 上的指标对比

用法: python3 plot_homa_figures.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

from matplotlib import font_manager
for fpath in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if os.path.exists(fpath):
        try:
            font_manager.fontManager.fontadd(fpath)
        except Exception:
            pass
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "predictions")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLOR = "#E76F51"


def main():
    # ---- parity ----
    df = pd.read_csv(os.path.join(PRED_DIR, "HOMA_test_predictions.csv"))
    yt, yp = df["y_true"].values, df["y_pred"].values
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    spr = spearmanr(yt, yp).correlation

    fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=160)
    xy = np.vstack([yt, yp])
    z = gaussian_kde(xy)(xy)
    order = np.argsort(z)
    sc = ax.scatter(yt[order], yp[order], c=z[order], s=14, cmap="viridis",
                    alpha=0.8, edgecolor="none")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    pad = 0.05 * (lims[1] - lims[0])
    ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", lw=1.2, alpha=0.6)
    ax.set_xlim(lims[0] - pad, lims[1] + pad)
    ax.set_ylim(lims[0] - pad, lims[1] + pad)
    ax.set_xlabel("True HOMA", fontsize=13)
    ax.set_ylabel("Predicted HOMA", fontsize=13)
    ax.set_title("HOMA · Merged 重训 (collet a/b + lunci10)\n固定 collet a/b test",
                 fontsize=12.5, fontweight="bold")
    ax.text(0.04, 0.96, f"$R^2$={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\n"
                         f"$\\rho$={spr:.3f}\nn={len(df)}",
            transform=ax.transAxes, va="top", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="0.6"))
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Density", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "HOMA_merged_test_parity.png")
    plt.savefig(p1, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close()
    print("[saved]", p1)

    # ---- old vs merged ----
    old = json.load(open(os.path.join(RESULT_DIR, "old_baseline_metrics.json")))["HOMA"]
    new = json.load(open(os.path.join(RESULT_DIR, "HOMA_metrics.json")))["test"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=160)
    labels = ["旧最优模型\n(仅 collet 训练)", "Merged 重训\n(collet a/b + lunci10)"]
    for ax, (key, tlabel, fmt_) in zip(axes, [("mae", "MAE (越低越好)", ".4f"),
                                               ("r2", "$R^2$ (越高越好)", ".4f")]):
        vals = [old[key], new[key]]
        bars = ax.bar([0, 1], vals, width=0.5, color=["#999999", COLOR],
                      alpha=0.9, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    format(v, fmt_), ha="center", va="bottom", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(tlabel, fontsize=12)
        ax.set_title(f"固定 collet a/b test · {tlabel}", fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("HOMA: 旧最优模型 vs Merged 重训模型", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "HOMA_old_vs_merged_metrics.png")
    plt.savefig(p2, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close()
    print("[saved]", p2)


if __name__ == "__main__":
    main()
