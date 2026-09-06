"""
在原始散点图基础上，用颜色深浅表示每个点的局部密度。
使用 gaussian_kde 对每个点的 2D 密度估计，颜色从浅到深。
"""
import os
import numpy as np
import pandas as pd
import matplotlib

# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

RESULTS_DIR = '_PROJ_ROOT/three_task_results'
OUTPUT_DIR = '_PROJ_ROOT/gnn_label_complete/results/three_task_eval/enhanced_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = [
    ('homa_nics_original', 'HOMA (NICS Original)'),
    ('homa_collet_0702', 'HOMA (Collet 0702)'),
    ('mbco_lunci2', 'MBCO (Lunci2)'),
]


def nice_scatter(true, pred, title, save_path, cmap='magma'):
    true = np.array(true).astype(float)
    pred = np.array(pred).astype(float)

    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    # 计算每个点的 2D 密度
    xy = np.vstack([true, pred])
    kde = gaussian_kde(xy)
    density = kde(xy)

    # 按密度排序绘制：低密度点先画，高密度点后画覆盖，视觉效果更好
    idx = np.argsort(density)

    fig, ax_main = plt.subplots(figsize=(9, 9), dpi=250)

    # 背景
    ax_main.set_facecolor('white')

    # 散点：颜色深浅表示局部密度
    sc = ax_main.scatter(true[idx], pred[idx], c=density[idx],
                         cmap=cmap, s=15, alpha=0.85,
                         edgecolors='none', zorder=3)

    # 对角线
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    pad = 0.02 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    ax_main.plot(lims, lims, 'r-', linewidth=1.8, alpha=0.85, zorder=4,
                 label='Perfect prediction (y=x)')

    # 拟合线
    z = np.polyfit(true, pred, 1)
    x_fit = np.linspace(lims[0], lims[1], 200)
    ax_main.plot(x_fit, np.poly1d(z)(x_fit), '--', color='#3498db',
                 linewidth=1.5, alpha=0.8, zorder=4,
                 label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_xlabel('True Values', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
    ax_main.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # colorbar
    cbar = fig.colorbar(sc, ax=ax_main, shrink=0.65, pad=0.02, aspect=35)
    cbar.set_label('Local Density', fontsize=12, fontweight='bold')
    cbar.outline.set_linewidth(0.5)

    # 统计信息框
    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nN = {len(true):,}'
    ax_main.text(0.97, 0.03, stats_text, transform=ax_main.transAxes,
                 fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor='black', alpha=0.92, linewidth=1.2),
                 family='monospace', fontweight='bold')

    ax_main.legend(loc='upper left', fontsize=11, framealpha=0.92,
                   edgecolor='gray', fancybox=True)
    ax_main.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, zorder=1)
    ax_main.tick_params(axis='both', labelsize=12)

    # === 上下 marginal rug 显示分布 ===
    # 上方
    ax_top = ax_main.inset_axes([0, 1.01, 1, 0.12], sharex=ax_main)
    ax_top.hist(true, bins=80, color='#2c3e50', alpha=0.5, density=True)
    ax_top.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax_top.set_ylabel('True', fontsize=9, rotation=0, labelpad=10)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    # 右侧
    ax_right = ax_main.inset_axes([1.01, 0, 0.12, 1], sharey=ax_main)
    ax_right.hist(pred, bins=80, color='#c0392b', alpha=0.5, density=True, orientation='horizontal')
    ax_right.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax_right.set_xlabel('Pred', fontsize=9, labelpad=10)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {save_path}")


def nice_scatter_with_subdensity(true, pred, title, save_path):
    """备用：带对角线subplot展示不同区域的密度"""
    nice_scatter(true, pred, title, save_path, cmap='plasma')


def main():
    for task_id, task_label in TASKS:
        task_dir = os.path.join(RESULTS_DIR, task_id)
        print(f"\n=== {task_label} ===")

        for set_name in ['test', 'train']:
            df = pd.read_csv(os.path.join(task_dir, f'{set_name}_set_predictions.csv'))
            true, pred = df['true'].values, df['pred'].values
            n = len(true)

            title = f'{task_label}\n{set_name.upper()} Set (N={n:,})'
            out = os.path.join(OUTPUT_DIR, f'{task_id}_{set_name}_nice_density.png')
            nice_scatter(true, pred, title, out, cmap='magma')

            # 再生成一个 plasma 色板版本
            out2 = os.path.join(OUTPUT_DIR, f'{task_id}_{set_name}_nice_density_plasma.png')
            nice_scatter(true, pred, title, out2, cmap='plasma')

    # 三任务对比 - nice density 版
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=220)
    for i, (task_id, task_label) in enumerate(TASKS):
        df = pd.read_csv(os.path.join(RESULTS_DIR, task_id, 'test_set_predictions.csv'))
        true, pred = df['true'].values, df['pred'].values
        r2 = r2_score(true, pred)
        mae = float(np.abs(true - pred).mean())
        rmse = float(np.sqrt(((true - pred) ** 2).mean()))

        xy = np.vstack([true, pred])
        kde = gaussian_kde(xy)
        density = kde(xy)
        idx = np.argsort(density)

        ax = axes[i]
        sc = ax.scatter(true[idx], pred[idx], c=density[idx], cmap='magma',
                        s=12, alpha=0.85, edgecolors='none')
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        pad = 0.02 * (lims[1] - lims[0])
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, 'r-', linewidth=1.5, alpha=0.85)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_label}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label('Density', fontsize=10)

    fig.suptitle('Three-Task Test Set Comparison - Point Density Colored Scatter',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, 'three_task_nice_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n三任务对比图保存: {save_path}")


if __name__ == '__main__':
    main()
