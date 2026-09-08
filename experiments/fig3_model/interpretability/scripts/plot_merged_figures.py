# -*- coding: utf-8 -*-
"""
Step 4: 绘图 — merged 重训模型在固定 collet a/b test 上的表现
  Figure 1: 三任务 test 真值-预测 parity (散点密度 + 指标标注)
  Figure 2: 旧最优模型 vs merged 重训模型 在同一 test 上的指标对比

用法: python3 plot_merged_figures.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_absolute_error

from matplotlib import font_manager
for fpath in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if os.path.exists(fpath):
        try:
            font_manager.fontManager.addfont(fpath)
        except Exception:
            pass
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "predictions")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
TASK_LABELS = {"HOMA": "HOMA", "NICS_1zz": "NICS(1)zz", "MBCO": "MBCO"}
TASK_COLORS = {"HOMA": "#E76F51", "NICS_1zz": "#3568C0", "MBCO": "#2A9D8F"}
# 单位/范围 (用于散点图方框范围)
AX_RANGE = {
    "HOMA": (None, None),
    "NICS_1zz": (None, None),
    "MBCO": (None, None),
}


def scatter_parity(ax, y_true, y_pred, color, label):
    """密度着色散点 (不画拟合线)"""
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        order = np.argsort(z)
        sc = ax.scatter(y_true[order], y_pred[order], c=z[order], s=10,
                        cmap="viridis", alpha=0.75, edgecolor="none")
    except Exception:
        sc = ax.scatter(y_true, y_pred, s=10, color=color, alpha=0.6,
                        edgecolor="none")
    lims = [np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.max([ax.get_xlim()[1], ax.get_ylim()[1]])]
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.7, zorder=0)
    return sc


def main():
    # ---- 读取 merged 重训 test 预测 ----
    merged_pred = {}
    for t in TASKS:
        merged_pred[t] = pd.read_csv(os.path.join(PRED_DIR, f"{t}_test_predictions.csv"))

    # ============ Figure 1: parity ============
    fig = plt.figure(figsize=(17, 5.2), dpi=150)
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)
    for idx, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[idx])
        df = merged_pred[t]
        yt, yp = df["y_true"].values, df["y_pred"].values
        sc = scatter_parity(ax, yt, yp, TASK_COLORS[t], t)
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        from scipy.stats import spearmanr
        spr = spearmanr(yt, yp).correlation
        ax.text(0.04, 0.97,
                f"$R^2$={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\n"
                f"$\\rho$={spr:.3f}\nn={len(df)}",
                transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="0.6"))
        ax.set_xlabel("True", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Prediction", fontsize=12)
        ax.set_title(f"{TASK_LABELS[t]} (merged 重训)", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Density", fontsize=9)

    fig.suptitle("Merged 训练 (collet a/b + lunci10) · Stage 6 最优配置 · 固定 collet test",
                 fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(FIG_DIR, "merged_test_parity.png"),
                bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print("[saved]", os.path.join(FIG_DIR, "merged_test_parity.png"))

    # ============ Figure 2: old vs new ============
    old_met = json.load(open(os.path.join(RESULT_DIR, "old_baseline_metrics.json")))
    new_met = json.load(open(os.path.join(RESULT_DIR, "HOMA_metrics.json")))  # placeholder

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    x = np.arange(len(TASKS)); width = 0.32
    # MAE panel
    ax = axes[0]
    old_mae = [old_met[t]["mae"] for t in TASKS]
    new_mae = []
    for t in TASKS:
        m = json.load(open(os.path.join(RESULT_DIR, f"{t}_metrics.json")))
        new_mae.append(m["test"]["mae"])
    b1 = ax.bar(x - width / 2, old_mae, width, label="旧最优模型 (仅 collet 训练)",
                color="#999999", alpha=0.9, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, new_mae, width, label="Merged 重训 (collet a/b + lunci10)",
                color=[TASK_COLORS[t] for t in TASKS], alpha=0.9, edgecolor="black", linewidth=0.5)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=12)
    ax.set_ylabel("Test MAE (越低越好)", fontsize=12)
    ax.set_title("固定 collet a/b test 上的 MAE", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # R2 panel
    ax = axes[1]
    old_r2 = [old_met[t]["r2"] for t in TASKS]
    new_r2 = []
    for t in TASKS:
        m = json.load(open(os.path.join(RESULT_DIR, f"{t}_metrics.json")))
        new_r2.append(m["test"]["r2"])
    b1 = ax.bar(x - width / 2, old_r2, width, label="旧最优模型 (仅 collet 训练)",
                color="#999999", alpha=0.9, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, new_r2, width, label="Merged 重训 (collet a/b + lunci10)",
                color=[TASK_COLORS[t] for t in TASKS], alpha=0.9, edgecolor="black", linewidth=0.5)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=12)
    ax.set_ylabel("Test $R^2$ (越高越好)", fontsize=12)
    ax.set_title("固定 collet a/b test 上的 $R^2$", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "old_vs_merged_metrics.png"),
                bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print("[saved]", os.path.join(FIG_DIR, "old_vs_merged_metrics.png"))

    # ============ 汇总表 csv ============
    rows = []
    for t in TASKS:
        m = json.load(open(os.path.join(RESULT_DIR, f"{t}_metrics.json")))
        rows.append({"task": t, "model": "merged",
                     "r2": m["test"]["r2"], "mae": m["test"]["mae"],
                     "rmse": m["test"]["rmse"], "spearman": m["test"]["spearman"],
                     "best_epoch": m["best_epoch"], "n_test": m["n_test"],
                     "n_collet_train": m["n_collet_train"], "n_l10_train": m["n_l10_train"]})
        rows.append({"task": t, "model": "old_baseline",
                     "r2": old_met[t]["r2"], "mae": old_met[t]["mae"],
                     "rmse": old_met[t]["rmse"], "spearman": old_met[t]["spearman"],
                     "best_epoch": None, "n_test": old_met[t]["n_test"],
                     "n_collet_train": None, "n_l10_train": None})
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "new_vs_old_metrics_summary.csv"), index=False)
    print("[saved]", os.path.join(RESULT_DIR, "new_vs_old_metrics_summary.csv"))


if __name__ == "__main__":
    main()
