"""
立体程度 vs 芳香性 散点图
- X = 立体描述符 (RPD, PBF, Fsp3, Cremer-Pople Q, Asphericity)
- Y = 芳香性描述符 (HOMA, MBCO, NICS-ZZ)
- Color = 环类型 (type1) / 反应类型 (type3)
- Shape = 反应物(o) vs 产物(^)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.linewidth': 0.8,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5, 'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

XLSX_PATH = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
OUTPUT_DIR = '/home/ubuntu/aroma-dps-code/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 配色 ──────────────────────────────────────────────────────────
RING_COLORS = {
    '呋喃': '#E64B35', '萘': '#4DBBD5', '萘2': '#00A087',
    '吲哚': '#3C5488', '吡咯': '#F39B7F', '苯酚': '#8491B4',
    '萘酚': '#91D1C2',
}
TYPE3_COLORS = {
    '光催化': '#3C5488', '氢化': '#00A087', '烯丙基去芳构化': '#F39B7F',
}
TYPE3_ORDER = ['光催化', '氢化', '烯丙基去芳构化']
RING_ORDER  = ['呋喃', '萘', '萘2', '吲哚', '吡咯', '苯酚', '萘酚']

MARKERS = {'反应物': 'o', '产物': '^'}
MARKER_LABELS = {'反应物': 'Reactant (o)', '产物': 'Product (△)'}


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12, linestyle='--')


def scatter_panel(ax, df, x_col, y_col, xlabel, ylabel, title,
                  color_col='type1', color_map=None, color_order=None,
                  show_legend=True, fit_by_group=False):
    """通用散点面板: X=stereo, Y=aromaticity, color=group, shape=反应物/产物"""
    if color_map is None:
        color_map = RING_COLORS
    if color_order is None:
        color_order = RING_ORDER

    for cat in color_order:
        sub = df[df[color_col] == cat]
        if len(sub) == 0:
            continue
        for t2 in ['反应物', '产物']:
            sub2 = sub[sub['type2'] == t2]
            if len(sub2) == 0:
                continue
            ax.scatter(sub2[x_col], sub2[y_col],
                       s=22, alpha=0.55,
                       color=color_map.get(cat, '#888'),
                       marker=MARKERS[t2],
                       edgecolors='white', linewidth=0.3, zorder=3)

    # 零参考线 (NICS 或 HOMA)
    if 'NICS' in y_col:
        ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.5)
    if 'HOMA' in y_col or 'MBCO' in y_col:
        ax.axhline(y=0.5, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.3)

    # 总体拟合
    valid = df.dropna(subset=[x_col, y_col])
    if len(valid) >= 8:
        r, p = stats.pearsonr(valid[x_col], valid[y_col])
        slope, intercept, _, _, _ = stats.linregress(valid[x_col], valid[y_col])
        x_fit = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                linewidth=1.2, alpha=0.5, zorder=2)
        ax.text(0.03, 0.03, f'r = {r:.3f}\np = {p:.1e}',
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9, linewidth=0.6))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    style_ax(ax)


def make_legend(ax, color_map, color_order, title='', loc='best'):
    """创建双色编码图例 (颜色+形状)"""
    color_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(c, '#888'),
               markersize=8, label=c)
        for c in color_order
    ]
    shape_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label=MARKER_LABELS['反应物']),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label=MARKER_LABELS['产物']),
    ]
    leg1 = ax.legend(handles=color_handles, title=title, loc='upper left',
                     frameon=True, framealpha=0.9, edgecolor='#ccc',
                     fontsize=8, title_fontsize=8.5,
                     handletextpad=0.3, borderpad=0.4)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, loc='upper right',
                     frameon=True, framealpha=0.9, edgecolor='#ccc',
                     fontsize=8, handletextpad=0.3, borderpad=0.4)


# ════════════════════════════════════════════════════════════════
#  图 1: 全局 — 立体程度 vs 芳香性 (color=环类型, shape=反应前后)
# ════════════════════════════════════════════════════════════════
def fig_global_stereo_vs_aroma(df):
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # 行 = 芳香性指标, 列 = 立体指标
    y_metrics = [
        ('HOMA',     'HOMA'),
        ('MBCO',     'MBCO'),
        ('NICS_ZZ',  'NICS-ZZ (ppm)'),
    ]
    x_metrics = [
        ('Ring_RPD',            'Ring RPD (Å)'),
        ('Mol_PBF',             'Mol PBF (Å)'),
        ('Fsp3',                'Fsp³'),
    ]

    for i, (yc, yl) in enumerate(y_metrics):
        for j, (xc, xl) in enumerate(x_metrics):
            ax = axes[i][j]
            panel = f'{"ABC"[i]}{"123"[j]})'
            scatter_panel(ax, df, xc, yc, xl, yl,
                          title=f'{panel} {yl} vs {xl}',
                          color_col='type1', color_map=RING_COLORS,
                          color_order=RING_ORDER)
            if i == 0 and j == 0:
                make_legend(ax, RING_COLORS, RING_ORDER, title='Ring Family')

    fig.suptitle('Stereo Degree vs Aromaticity (All Data)\n'
                 'Color = Ring Family,  Shape = ○ Reactant / △ Product',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig14_global_stereo_vs_aroma.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig14: Global Stereo vs Aroma ✓')


# ════════════════════════════════════════════════════════════════
#  图 2: 全局 — 补充立体指标 (Q, Asphericity, NPR1)
# ════════════════════════════════════════════════════════════════
def fig_global_supplementary_stereo(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    y_metrics = [
        ('HOMA',    'HOMA'),
        ('NICS_ZZ', 'NICS-ZZ (ppm)'),
    ]
    x_metrics = [
        ('Ring_Cremer_Pople_Q', 'Cremer-Pople Q (Å)'),
        ('Asphericity',         'Asphericity'),
        ('NPR1',                'NPR1 (I₁/I₃)'),
    ]

    for i, (yc, yl) in enumerate(y_metrics):
        for j, (xc, xl) in enumerate(x_metrics):
            ax = axes[i][j]
            panel = f'{"AB"[i]}{"123"[j]})'
            scatter_panel(ax, df, xc, yc, xl, yl,
                          title=f'{panel} {yl} vs {xl}',
                          color_col='type1', color_map=RING_COLORS,
                          color_order=RING_ORDER)
            if i == 0 and j == 0:
                make_legend(ax, RING_COLORS, RING_ORDER, title='Ring Family')

    fig.suptitle('Supplementary: Stereo Descriptors vs Aromaticity\n'
                 'Color = Ring Family,  Shape = ○ Reactant / △ Product',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig15_global_supplementary_stereo.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig15: Supplementary Stereo vs Aroma ✓')


# ════════════════════════════════════════════════════════════════
#  图 3: 吲哚固定 — 立体 vs 芳香性 (color=type3, shape=反应前后)
# ════════════════════════════════════════════════════════════════
def fig_indole_stereo_vs_aroma(df):
    indole = df[df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    y_metrics = [
        ('HOMA',    'HOMA'),
        ('MBCO',    'MBCO'),
        ('NICS_ZZ', 'NICS-ZZ (ppm)'),
    ]
    x_metrics = [
        ('Ring_RPD', 'Ring RPD (Å)'),
        ('Mol_PBF',  'Mol PBF (Å)'),
        ('Fsp3',     'Fsp³'),
    ]

    for i, (yc, yl) in enumerate(y_metrics):
        for j, (xc, xl) in enumerate(x_metrics):
            ax = axes[i][j]
            panel = f'{"ABC"[i]}{"123"[j]})'
            scatter_panel(ax, indole, xc, yc, xl, yl,
                          title=f'{panel} {yl} vs {xl}',
                          color_col='type3', color_map=TYPE3_COLORS,
                          color_order=TYPE3_ORDER)
            if i == 0 and j == 0:
                make_legend(ax, TYPE3_COLORS, TYPE3_ORDER, title='Reaction Type')

    fig.suptitle('Indole: Stereo Degree vs Aromaticity by Reaction Type\n'
                 'Color = Reaction Type,  Shape = ○ Reactant / △ Product',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig16_indole_stereo_vs_aroma.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig16: Indole Stereo vs Aroma ✓')


# ════════════════════════════════════════════════════════════════
#  图 4: 吲哚 — 补充立体指标 (Q, Asphericity, NPR1)
# ════════════════════════════════════════════════════════════════
def fig_indole_supplementary_stereo(df):
    indole = df[df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    y_metrics = [
        ('HOMA',    'HOMA'),
        ('NICS_ZZ', 'NICS-ZZ (ppm)'),
    ]
    x_metrics = [
        ('Ring_Cremer_Pople_Q', 'Cremer-Pople Q (Å)'),
        ('Asphericity',         'Asphericity'),
        ('NPR1',                'NPR1 (I₁/I₃)'),
    ]

    for i, (yc, yl) in enumerate(y_metrics):
        for j, (xc, xl) in enumerate(x_metrics):
            ax = axes[i][j]
            panel = f'{"AB"[i]}{"123"[j]})'
            scatter_panel(ax, indole, xc, yc, xl, yl,
                          title=f'{panel} {yl} vs {xl}',
                          color_col='type3', color_map=TYPE3_COLORS,
                          color_order=TYPE3_ORDER)
            if i == 0 and j == 0:
                make_legend(ax, TYPE3_COLORS, TYPE3_ORDER, title='Reaction Type')

    fig.suptitle('Indole: Supplementary Stereo Descriptors vs Aromaticity\n'
                 'Color = Reaction Type,  Shape = ○ Reactant / △ Product',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig17_indole_supplementary_stereo.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig17: Indole Supplementary Stereo ✓')


# ════════════════════════════════════════════════════════════════
#  图 5: 重点大图 — RPD vs HOMA (全局, 按反应类型分面)
# ════════════════════════════════════════════════════════════════
def fig_focus_rpd_homa(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, t3 in zip(axes, TYPE3_ORDER):
        sub = df[df['type3'] == t3]
        for t1 in RING_ORDER:
            sub2 = sub[sub['type1'] == t1]
            if len(sub2) == 0:
                continue
            for t2 in ['反应物', '产物']:
                sub3 = sub2[sub2['type2'] == t2]
                if len(sub3) == 0:
                    continue
                ax.scatter(sub3['Ring_RPD'], sub3['HOMA'],
                           s=26, alpha=0.6,
                           color=RING_COLORS.get(t1, '#888'),
                           marker=MARKERS[t2],
                           edgecolors='white', linewidth=0.3, zorder=3)

        # 拟合
        valid = sub.dropna(subset=['Ring_RPD', 'HOMA'])
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid['Ring_RPD'], valid['HOMA'])
            slope, intercept, _, _, _ = stats.linregress(valid['Ring_RPD'], valid['HOMA'])
            x_fit = np.linspace(valid['Ring_RPD'].min(), valid['Ring_RPD'].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                    linewidth=1.3, alpha=0.6)
            ax.text(0.03, 0.03, f'r = {r:.3f}, p = {p:.1e}\nn = {len(valid)}',
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9, linewidth=0.6))

        ax.axhline(y=0.5, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.3)
        ax.set_xlabel('Ring RPD (Å)')
        ax.set_ylabel('HOMA')
        ax.set_title(f'{t3} (n={len(sub)})', fontsize=12, fontweight='bold')
        style_ax(ax)

    # 全局图例
    color_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RING_COLORS[c],
               markersize=8, label=c) for c in RING_ORDER
    ]
    shape_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Reactant (○)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Product (△)'),
    ]
    fig.legend(handles=color_handles + shape_handles,
               loc='lower center', ncol=len(RING_ORDER) + 2,
               frameon=True, framealpha=0.9, edgecolor='#ccc',
               fontsize=9, title='Ring Family / Reactant vs Product',
               title_fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Ring RPD vs HOMA — by Reaction Type\n'
                 '(Color = Ring Family,  Shape = ○ Reactant / △ Product)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'Fig18_focus_RPD_vs_HOMA.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig18: Focus RPD vs HOMA ✓')


# ════════════════════════════════════════════════════════════════
#  图 6: 重点大图 — PBF vs HOMA (全局, 按反应类型分面)
# ════════════════════════════════════════════════════════════════
def fig_focus_pbf_homa(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, t3 in zip(axes, TYPE3_ORDER):
        sub = df[df['type3'] == t3]
        for t1 in RING_ORDER:
            sub2 = sub[sub['type1'] == t1]
            if len(sub2) == 0:
                continue
            for t2 in ['反应物', '产物']:
                sub3 = sub2[sub2['type2'] == t2]
                if len(sub3) == 0:
                    continue
                ax.scatter(sub3['Mol_PBF'], sub3['HOMA'],
                           s=26, alpha=0.6,
                           color=RING_COLORS.get(t1, '#888'),
                           marker=MARKERS[t2],
                           edgecolors='white', linewidth=0.3, zorder=3)

        valid = sub.dropna(subset=['Mol_PBF', 'HOMA'])
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid['Mol_PBF'], valid['HOMA'])
            slope, intercept, _, _, _ = stats.linregress(valid['Mol_PBF'], valid['HOMA'])
            x_fit = np.linspace(valid['Mol_PBF'].min(), valid['Mol_PBF'].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                    linewidth=1.3, alpha=0.6)
            ax.text(0.03, 0.03, f'r = {r:.3f}, p = {p:.1e}\nn = {len(valid)}',
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9, linewidth=0.6))

        ax.axhline(y=0.5, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.3)
        ax.set_xlabel('Mol PBF (Å)')
        ax.set_ylabel('HOMA')
        ax.set_title(f'{t3} (n={len(sub)})', fontsize=12, fontweight='bold')
        style_ax(ax)

    color_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RING_COLORS[c],
               markersize=8, label=c) for c in RING_ORDER
    ]
    shape_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Reactant (○)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Product (△)'),
    ]
    fig.legend(handles=color_handles + shape_handles,
               loc='lower center', ncol=len(RING_ORDER) + 2,
               frameon=True, framealpha=0.9, edgecolor='#ccc',
               fontsize=9, title='Ring Family / Reactant vs Product',
               title_fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Mol PBF vs HOMA — by Reaction Type\n'
                 '(Color = Ring Family,  Shape = ○ Reactant / △ Product)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'Fig19_focus_PBF_vs_HOMA.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig19: Focus PBF vs HOMA ✓')


# ════════════════════════════════════════════════════════════════
#  图 7: 吲哚 — RPD vs HOMA 按 type3 分面 (重点大图)
# ════════════════════════════════════════════════════════════════
def fig_indole_focus_rpd_homa(df):
    indole = df[df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, t3 in zip(axes, TYPE3_ORDER):
        sub = indole[indole['type3'] == t3]
        for t2 in ['反应物', '产物']:
            sub2 = sub[sub['type2'] == t2]
            if len(sub2) == 0:
                continue
            ax.scatter(sub2['Ring_RPD'], sub2['HOMA'],
                       s=30, alpha=0.6,
                       color=TYPE3_COLORS.get(t3, '#888'),
                       marker=MARKERS[t2],
                       edgecolors='white', linewidth=0.3, zorder=3,
                       label=f'{t2} (n={len(sub2)})')

        valid = sub.dropna(subset=['Ring_RPD', 'HOMA'])
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid['Ring_RPD'], valid['HOMA'])
            slope, intercept, _, _, _ = stats.linregress(valid['Ring_RPD'], valid['HOMA'])
            x_fit = np.linspace(valid['Ring_RPD'].min(), valid['Ring_RPD'].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                    linewidth=1.3, alpha=0.6)
            ax.text(0.03, 0.03, f'r = {r:.3f}, p = {p:.1e}\nn = {len(valid)}',
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9, linewidth=0.6))

        ax.axhline(y=0.5, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=0.7, alpha=0.3)
        ax.set_xlabel('Ring RPD (Å)')
        ax.set_ylabel('HOMA')
        ax.set_title(f'{t3} (n={len(sub)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, frameon=True, framealpha=0.9,
                  edgecolor='#ccc', loc='upper right')
        style_ax(ax)

    fig.suptitle('Indole: Ring RPD vs HOMA — by Reaction Type\n'
                 '(Shape = ○ Reactant / △ Product)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig20_indole_focus_RPD_vs_HOMA.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig20: Indole Focus RPD vs HOMA ✓')


# ════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('立体程度 vs 芳香性 散点图')
    print('=' * 60)

    df = pd.read_excel(XLSX_PATH)
    print(f"数据: {len(df)} 行")

    print('\n生成图表:')
    # 全局: 主力 3×3 矩阵
    fig_global_stereo_vs_aroma(df)
    fig_global_supplementary_stereo(df)

    # 吲哚专题
    fig_indole_stereo_vs_aroma(df)
    fig_indole_supplementary_stereo(df)

    # 重点大图 (按反应类型分面)
    fig_focus_rpd_homa(df)
    fig_focus_pbf_homa(df)
    fig_indole_focus_rpd_homa(df)

    print(f'\n完成! → {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
