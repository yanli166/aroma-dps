"""
生成美观的密度散点图 - 反映点的疏密分布
包含: 六边形密度图 + 边缘分布 + 2D等高线 + 残差图
"""
import os
import sys
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

PROJ_ROOT = _PROJ_ROOT
RESULTS_DIR = os.path.join(PROJ_ROOT, 'three_task_results')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'gnn_label_complete', 'results', 'three_task_eval')

TASKS = [
    ('homa_nics_original', 'HOMA (NICS Original)', '#1f77b4'),
    ('homa_collet_0702', 'HOMA (Collet 0702)', '#ff7f0e'),
    ('mbco_lunci2', 'MBCO (Lunci2)', '#2ca02c'),
]


def plot_density_scatter(true, pred, title, color, save_path, subtitle=''):
    """主图：六边形密度 + 边缘直方图 + 对角线 + 统计标注"""
    true = np.array(true)
    pred = np.array(pred)
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    fig = plt.figure(figsize=(10, 10), dpi=200)
    gs = gridspec.GridSpec(4, 4, hspace=0.1, wspace=0.1)

    # 主图区域
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # 上方边缘直方图
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # 右侧边缘直方图
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # --- 主图：六边形密度 ---
    # 自定义色图：白→浅色→深色
    cmap = LinearSegmentedColormap.from_list('density',
        ['#ffffff', '#e8f0fe', '#c6dbef', '#9ecae1', '#6baed6',
         '#4292c6', '#2171b5', '#08519c', '#08306b'])

    hb = ax_main.hexbin(true, pred, gridsize=40, cmap=cmap, mincnt=1,
                        edgecolors='none', alpha=0.9)

    # 对角线
    lims = [min(true.min(), pred.min()) - 0.02 * abs(true.min()),
            max(true.max(), pred.max()) + 0.02 * abs(true.max())]
    ax_main.plot(lims, lims, 'r-', linewidth=2, alpha=0.8, zorder=5)
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    # 回归线（虚线）
    if len(true) > 2:
        z = np.polyfit(true, pred, 1)
        p_fit = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax_main.plot(x_fit, p_fit(x_fit), '--', color='#e74c3c',
                     linewidth=1.5, alpha=0.7, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # colorbar
    cb = fig.colorbar(hb, ax=ax_main, shrink=0.7, pad=0.02, aspect=30)
    cb.set_label('Point Count', fontsize=11)

    ax_main.set_xlabel('True Values', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')

    # 统计标注框
    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nN = {len(true)}'
    ax_main.text(0.97, 0.03, stats_text, transform=ax_main.transAxes,
                 fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='gray', alpha=0.9),
                 family='monospace')

    # 标题
    full_title = title
    if subtitle:
        full_title += f'\n{subtitle}'
    ax_main.set_title(full_title, fontsize=14, fontweight='bold', pad=15)

    # --- 上方边缘直方图 ---
    ax_top.hist(true, bins=50, color=color, alpha=0.6, edgecolor='white',
                linewidth=0.3, density=True)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(labelbottom=False)

    # --- 右侧边缘直方图 ---
    ax_right.hist(pred, bins=50, color=color, alpha=0.6, edgecolor='white',
                  linewidth=0.3, density=True, orientation='horizontal')
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)

    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {save_path}")


def plot_kde_density(true, pred, title, save_path):
    """2D KDE 等高线密度图 + 散点底层"""
    true = np.array(true)
    pred = np.array(pred)
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)

    # 底层：散点（低透明度）
    ax.scatter(true, pred, alpha=0.15, s=8, c='navy', zorder=1)

    # 上层：KDE 等高线
    xy = np.vstack([true, pred])
    try:
        kde = gaussian_kde(xy)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        xrange = np.linspace(lims[0] - 0.05 * abs(lims[0]),
                             lims[1] + 0.05 * abs(lims[1]), 200)
        yrange = np.linspace(lims[0] - 0.05 * abs(lims[0]),
                             lims[1] + 0.05 * abs(lims[1]), 200)
        Xi, Yi = np.meshgrid(xrange, yrange)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        levels = np.linspace(Zi.min(), Zi.max(), 15)
        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='YlOrRd',
                              alpha=0.6, zorder=2)
        ax.contour(Xi, Yi, Zi, levels=levels[:8], colors='darkred',
                   linewidths=0.5, alpha=0.4, zorder=3)
        cb = fig.colorbar(contour, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label('Density', fontsize=11)
    except Exception as e:
        print(f"  KDE 跳过: {e}")

    # 对角线
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r-', linewidth=2.5, alpha=0.9, zorder=5,
            label='Perfect Prediction (y=x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('True Values', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')
    ax.set_title(f'{title} - KDE Density Plot\n'
                 f'R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, N={len(true)}',
                 fontsize=14, fontweight='bold')

    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9),
            family='monospace')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {save_path}")


def plot_residual(true, pred, title, save_path):
    """残差图 + 误差分布"""
    true = np.array(true)
    pred = np.array(pred)
    residual = pred - true
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200,
                              gridspec_kw={'width_ratios': [2, 1]})

    # 左图：残差 vs 真实值（散点+密度色）
    ax1 = axes[0]
    scatter = ax1.scatter(true, residual, c=np.abs(residual), cmap='RdYlBu_r',
                          alpha=0.6, s=12, edgecolors='none', zorder=2)
    ax1.axhline(y=0, color='red', linewidth=2, linestyle='-', alpha=0.8, zorder=5)
    # 拟合趋势线
    if len(true) > 10:
        z = np.polyfit(true, residual, 1)
        p_fit = np.poly1d(z)
        x_fit = np.linspace(true.min(), true.max(), 100)
        ax1.plot(x_fit, p_fit(x_fit), '--', color='black', linewidth=1.5,
                 alpha=0.6, label=f'Trend: slope={z[0]:.4f}')
        ax1.legend(fontsize=10, framealpha=0.9)

    ax1.set_xlabel('True Values', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Residual (Pred - True)', fontsize=13, fontweight='bold')
    ax1.set_title(f'{title} - Residual Analysis', fontsize=14, fontweight='bold')
    cb1 = fig.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.02)
    cb1.set_label('|Residual|', fontsize=11)
    ax1.grid(True, alpha=0.2)

    # 右图：残差直方图 + 正态拟合
    ax2 = axes[1]
    n, bins, patches = ax2.hist(residual, bins=60, color='steelblue',
                                 edgecolor='white', linewidth=0.3, density=True, alpha=0.7)
    # 正态分布拟合
    mu, sigma = np.mean(residual), np.std(residual)
    x_norm = np.linspace(bins[0], bins[-1], 100)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.4f}, σ={sigma:.4f}')
    ax2.axvline(x=0, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Residual', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    # 添加统计信息
    stats_text = (f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\n'
                  f'Mean Residual = {mu:.4f}\nMax |Residual| = {np.max(np.abs(residual)):.4f}')
    fig.text(0.98, 0.02, stats_text, fontsize=11, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9),
             family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {save_path}")


def plot_combined(true, pred, title, color, save_path):
    """组合图：密度散点 + KDE等高线 + 边缘分布 + 残差（2x2布局）"""
    true = np.array(true)
    pred = np.array(pred)
    residual = pred - true
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=200)
    fig.suptitle(f'{title} - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)

    # === 左上：六边形密度 + 边缘直方图 ===
    ax1 = axes[0, 0]
    cmap = LinearSegmentedColormap.from_list('density',
        ['#ffffff', '#deebf7', '#9ecae1', '#4292c6', '#08519c', '#08306b'])
    hb = ax1.hexbin(true, pred, gridsize=35, cmap=cmap, mincnt=1,
                    edgecolors='none', alpha=0.9)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax1.plot(lims, lims, 'r-', linewidth=2, alpha=0.8)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_title('Hexbin Density Scatter', fontsize=13, fontweight='bold')
    cb = fig.colorbar(hb, ax=ax1, shrink=0.75, pad=0.02)
    cb.set_label('Count', fontsize=10)
    ax1.text(0.97, 0.03, f'R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}',
             transform=ax1.transAxes, fontsize=11, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
             family='monospace')
    ax1.grid(True, alpha=0.15)

    # === 右上：KDE 等高线 ===
    ax2 = axes[0, 1]
    ax2.scatter(true, pred, alpha=0.1, s=5, c='navy')
    xy = np.vstack([true, pred])
    try:
        kde = gaussian_kde(xy)
        xrange = np.linspace(lims[0], lims[1], 150)
        yrange = np.linspace(lims[0], lims[1], 150)
        Xi, Yi = np.meshgrid(xrange, yrange)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        levels = np.linspace(Zi.min(), Zi.max(), 12)
        c = ax2.contourf(Xi, Yi, Zi, levels=levels, cmap='YlOrRd', alpha=0.6)
        ax2.contour(Xi, Yi, Zi, levels=levels[:6], colors='darkred',
                    linewidths=0.5, alpha=0.3)
        cb2 = fig.colorbar(c, ax=ax2, shrink=0.75, pad=0.02)
        cb2.set_label('Density', fontsize=10)
    except:
        pass
    ax2.plot(lims, lims, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_title('KDE Contour Density', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.15)

    # === 左下：残差散点 ===
    ax3 = axes[1, 0]
    sc = ax3.scatter(true, residual, c=np.abs(residual), cmap='RdYlBu_r',
                     alpha=0.6, s=10, edgecolors='none')
    ax3.axhline(y=0, color='red', linewidth=2, alpha=0.8)
    # 趋势线
    if len(true) > 10:
        z = np.polyfit(true, residual, 1)
        x_fit = np.linspace(true.min(), true.max(), 100)
        ax3.plot(x_fit, np.polyval(z, x_fit), '--', color='black',
                 linewidth=1.5, alpha=0.5, label=f'Trend slope={z[0]:.4f}')
        ax3.legend(fontsize=10)
    ax3.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residual (Pred - True)', fontsize=12, fontweight='bold')
    ax3.set_title('Residual vs True Value', fontsize=13, fontweight='bold')
    cb3 = fig.colorbar(sc, ax=ax3, shrink=0.75, pad=0.02)
    cb3.set_label('|Residual|', fontsize=10)
    ax3.grid(True, alpha=0.15)

    # === 右下：残差分布直方图 ===
    ax4 = axes[1, 1]
    ax4.hist(residual, bins=60, color=color, edgecolor='white',
             linewidth=0.3, density=True, alpha=0.7)
    mu, sigma = np.mean(residual), np.std(residual)
    x_norm = np.linspace(residual.min(), residual.max(), 100)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
    ax4.plot(x_norm, y_norm, 'r-', linewidth=2,
             label=f'Normal: μ={mu:.4f}, σ={sigma:.4f}')
    ax4.axvline(x=0, color='red', linewidth=1.5, linestyle='--', alpha=0.4)
    ax4.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {save_path}")


def main():
    # 同时确保输出目录存在
    enhanced_dir = os.path.join(OUTPUT_DIR, 'enhanced_plots')
    os.makedirs(enhanced_dir, exist_ok=True)

    for task_id, task_label, color in TASKS:
        task_dir = os.path.join(RESULTS_DIR, task_id)
        print(f"\n=== {task_label} ===")

        # 测试集
        test_df = pd.read_csv(os.path.join(task_dir, 'test_set_predictions.csv'))
        train_df = pd.read_csv(os.path.join(task_dir, 'train_set_predictions.csv'))

        for set_name, df in [('test', test_df), ('train', train_df)]:
            true = df['true'].values
            pred = df['pred'].values
            n = len(true)
            subtitle = f'{set_name.upper()} Set (N={n})'

            # 1. 六边形密度 + 边缘直方图
            plot_density_scatter(
                true, pred, task_label, color,
                os.path.join(enhanced_dir, f'{task_id}_{set_name}_hexbin.png'),
                subtitle)

            # 2. KDE 等高线密度图
            plot_kde_density(
                true, pred, f'{task_label} ({set_name})',
                os.path.join(enhanced_dir, f'{task_id}_{set_name}_kde.png'))

            # 3. 残差分析图
            plot_residual(
                true, pred, f'{task_label} ({set_name})',
                os.path.join(enhanced_dir, f'{task_id}_{set_name}_residual.png'))

            # 4. 组合大图（2x2）
            plot_combined(
                true, pred, f'{task_label} ({set_name})', color,
                os.path.join(enhanced_dir, f'{task_id}_{set_name}_combined.png'))

    # 汇总对比图：三个任务的测试集散点并排
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
    for i, (task_id, task_label, color) in enumerate(TASKS):
        task_dir = os.path.join(RESULTS_DIR, task_id)
        df = pd.read_csv(os.path.join(task_dir, 'test_set_predictions.csv'))
        true, pred = df['true'].values, df['pred'].values
        r2 = r2_score(true, pred)
        mae = float(np.abs(true - pred).mean())
        rmse = float(np.sqrt(((true - pred) ** 2).mean()))

        ax = axes[i]
        cmap = LinearSegmentedColormap.from_list('d',
            ['#ffffff', '#deebf7', '#9ecae1', '#4292c6', '#08519c', '#08306b'])
        hb = ax.hexbin(true, pred, gridsize=35, cmap=cmap, mincnt=1, edgecolors='none')
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r-', linewidth=2, alpha=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True Values', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')
        ax.set_title(f'{task_label}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}',
                     fontsize=13, fontweight='bold')
        cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label('Count', fontsize=10)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Three-Task Test Set Comparison - Density Scatter', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(enhanced_dir, 'three_task_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n三任务对比图保存: {save_path}")

    print(f"\n全部完成! 增强图保存到: {enhanced_dir}")


if __name__ == '__main__':
    main()
