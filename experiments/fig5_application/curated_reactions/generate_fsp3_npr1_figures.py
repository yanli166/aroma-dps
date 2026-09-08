"""
Fsp3 & NPR1 vs Aromaticity — standalone clean scatter plots
No fit lines, clear legends, English labels.
"""
import os, warnings
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
    'font.size': 11, 'axes.linewidth': 0.9,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

XLSX = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
OUT  = '/home/ubuntu/aroma-dps-code/figures'
os.makedirs(OUT, exist_ok=True)

RING_COLORS = {
    'Furan':'#E64B35', 'Naphthalene':'#4DBBD5', 'Naphthalene-2':'#00A087',
    'Indole':'#3C5488', 'Pyrrole':'#F39B7F', 'Phenol':'#8491B4',
    'Naphthol':'#91D1C2',
}
TYPE3_COLORS = {
    'Photocatalysis':'#3C5488', 'Hydrogenation':'#00A087', 'Allylic Dearom.':'#E64B35',
}
TYPE3_ORDER = ['Photocatalysis', 'Hydrogenation', 'Allylic Dearom.']
RING_ORDER  = ['Furan', 'Naphthalene', 'Naphthalene-2', 'Indole', 'Pyrrole', 'Phenol', 'Naphthol']

MARKERS = {'Reactant': 'o', 'Product': '^'}
MARKER_SIZE = 28

CN_TO_EN_T1 = {'呋喃':'Furan','萘':'Naphthalene','萘2':'Naphthalene-2','吲哚':'Indole',
               '吡咯':'Pyrrole','苯酚':'Phenol','萘酚':'Naphthol'}
CN_TO_EN_T3 = {'光催化':'Photocatalysis', '氢化':'Hydrogenation', '烯丙基去芳构化':'Allylic Dearom.'}
CN_TO_EN_T2 = {'反应物':'Reactant', '产物':'Product'}


def load_data():
    df = pd.read_excel(XLSX)
    df['type1_en'] = df['type1'].map(CN_TO_EN_T1)
    df['type3_en'] = df['type3'].map(CN_TO_EN_T3)
    df['type2_en'] = df['type2'].map(CN_TO_EN_T2)
    return df


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.1, linestyle='--')


def make_dual_legend(ax, color_map, color_order, color_title):
    """Two-part legend: color = category, shape = Reactant/Product"""
    color_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map[c],
               markersize=9, markeredgecolor='white', markeredgewidth=0.4, label=c)
        for c in color_order
    ]
    shape_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
               markersize=9, markeredgecolor='white', markeredgewidth=0.4, label='Reactant'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='gray',
               markersize=9, markeredgecolor='white', markeredgewidth=0.4, label='Product'),
    ]
    leg1 = ax.legend(handles=color_handles, title=color_title, loc='upper left',
                     frameon=True, framealpha=0.95, edgecolor='#ccc',
                     fontsize=9.5, title_fontsize=10,
                     handletextpad=0.4, borderpad=0.5, labelspacing=0.5)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, title='Stage', loc='upper right',
                     frameon=True, framealpha=0.95, edgecolor='#ccc',
                     fontsize=9.5, title_fontsize=10,
                     handletextpad=0.4, borderpad=0.5, labelspacing=0.5)


def scatter_panel(ax, df, x_col, y_col, xlabel, ylabel, title,
                  color_col, color_map, color_order, color_title,
                  show_legend=False, show_stats=True):
    """Clean scatter: no fit line, optional Pearson r in corner"""
    for cat in color_order:
        sub = df[df[color_col] == cat]
        if len(sub) == 0:
            continue
        for stage in ['Reactant', 'Product']:
            sub2 = sub[sub['type2_en'] == stage]
            if len(sub2) == 0:
                continue
            ax.scatter(sub2[x_col], sub2[y_col],
                       s=MARKER_SIZE, alpha=0.55,
                       color=color_map.get(cat, '#888'),
                       marker=MARKERS[stage],
                       edgecolors='white', linewidth=0.4, zorder=3)

    # Reference lines only (no fit line)
    if 'HOMA' in y_col or 'MBCO' in y_col:
        ax.axhline(y=0.5, color='#bbb', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.axhline(y=0, color='#bbb', linestyle=':', linewidth=0.8, alpha=0.3)
    if 'NICS' in y_col:
        ax.axhline(y=0, color='#bbb', linestyle=':', linewidth=0.8, alpha=0.4)

    # Pearson r (text only, no line)
    if show_stats:
        valid = df.dropna(subset=[x_col, y_col])
        if len(valid) >= 8:
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            txt = f'r = {r:.3f}\np = {p:.1e}' if p < 0.001 else f'r = {r:.3f}\np = {p:.3f}'
            ax.text(0.97, 0.03, txt, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9.5,
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor='#ccc', alpha=0.9, linewidth=0.7))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=10)
    style_ax(ax)

    if show_legend:
        make_dual_legend(ax, color_map, color_order, color_title)


# ════════════════════════════════════════════════════════════════
#  1. Global: Fsp3 vs Aromaticity (color = ring family)
# ════════════════════════════════════════════════════════════════
def fig_fsp3_global(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    panels = [
        ('Fsp3', 'HOMA',    'Fsp\u00b3', 'HOMA',          '(A) HOMA vs Fsp\u00b3'),
        ('Fsp3', 'MBCO',    'Fsp\u00b3', 'MBCO',          '(B) MBCO vs Fsp\u00b3'),
        ('Fsp3', 'NICS_ZZ', 'Fsp\u00b3', 'NICS-ZZ (ppm)', '(C) NICS-ZZ vs Fsp\u00b3'),
    ]

    for ax, (xc, yc, xl, yl, title) in zip(axes, panels):
        scatter_panel(ax, df, xc, yc, xl, yl, title,
                      'type1_en', RING_COLORS, RING_ORDER, 'Ring Family',
                      show_legend=(xc == 'Fsp3' and yc == 'HOMA'))

    fig.suptitle('Fraction of sp\u00b3 Carbons (Fsp\u00b3) vs Aromaticity \u2014 All Data\n'
                 'Color = Ring Family    Shape: \u25CB Reactant  /  \u25B3 Product',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, 'Clean_Fig01_Fsp3_vs_aroma_global.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig01: Fsp3 Global done')


# ════════════════════════════════════════════════════════════════
#  2. Global: NPR1 vs Aromaticity (color = ring family)
# ════════════════════════════════════════════════════════════════
def fig_npr1_global(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    panels = [
        ('NPR1', 'HOMA',    'NPR1 (I\u2081/I\u2083)', 'HOMA',          '(A) HOMA vs NPR1'),
        ('NPR1', 'NICS_ZZ', 'NPR1 (I\u2081/I\u2083)', 'NICS-ZZ (ppm)', '(B) NICS-ZZ vs NPR1'),
    ]

    for ax, (xc, yc, xl, yl, title) in zip(axes, panels):
        scatter_panel(ax, df, xc, yc, xl, yl, title,
                      'type1_en', RING_COLORS, RING_ORDER, 'Ring Family',
                      show_legend=(yc == 'HOMA'))

    fig.suptitle('NPR1 (Normalized Principal Moment) vs Aromaticity \u2014 All Data\n'
                 'Color = Ring Family    Shape: \u25CB Reactant  /  \u25B3 Product',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT, 'Clean_Fig02_NPR1_vs_aroma_global.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig02: NPR1 Global done')


# ════════════════════════════════════════════════════════════════
#  3. Indole: Fsp3 vs Aromaticity (color = reaction type)
# ════════════════════════════════════════════════════════════════
def fig_fsp3_indole(df):
    indole = df[df['type1'] == '吲哚']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    panels = [
        ('Fsp3', 'HOMA',    'Fsp\u00b3', 'HOMA',          '(A) HOMA vs Fsp\u00b3'),
        ('Fsp3', 'MBCO',    'Fsp\u00b3', 'MBCO',          '(B) MBCO vs Fsp\u00b3'),
        ('Fsp3', 'NICS_ZZ', 'Fsp\u00b3', 'NICS-ZZ (ppm)', '(C) NICS-ZZ vs Fsp\u00b3'),
    ]

    for ax, (xc, yc, xl, yl, title) in zip(axes, panels):
        scatter_panel(ax, indole, xc, yc, xl, yl, title,
                      'type3_en', TYPE3_COLORS, TYPE3_ORDER, 'Reaction Type',
                      show_legend=(yc == 'HOMA'))

    fig.suptitle('Indole: Fsp\u00b3 vs Aromaticity by Reaction Type\n'
                 'Color = Reaction Type    Shape: \u25CB Reactant  /  \u25B3 Product',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, 'Clean_Fig03_Fsp3_vs_aroma_indole.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig03: Fsp3 Indole done')


# ════════════════════════════════════════════════════════════════
#  4. Indole: NPR1 vs Aromaticity (color = reaction type)
# ════════════════════════════════════════════════════════════════
def fig_npr1_indole(df):
    indole = df[df['type1'] == '吲哚']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    panels = [
        ('NPR1', 'HOMA',    'NPR1 (I\u2081/I\u2083)', 'HOMA',          '(A) HOMA vs NPR1'),
        ('NPR1', 'NICS_ZZ', 'NPR1 (I\u2081/I\u2083)', 'NICS-ZZ (ppm)', '(B) NICS-ZZ vs NPR1'),
    ]

    for ax, (xc, yc, xl, yl, title) in zip(axes, panels):
        scatter_panel(ax, indole, xc, yc, xl, yl, title,
                      'type3_en', TYPE3_COLORS, TYPE3_ORDER, 'Reaction Type',
                      show_legend=(yc == 'HOMA'))

    fig.suptitle('Indole: NPR1 vs Aromaticity by Reaction Type\n'
                 'Color = Reaction Type    Shape: \u25CB Reactant  /  \u25B3 Product',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT, 'Clean_Fig04_NPR1_vs_aroma_indole.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig04: NPR1 Indole done')


# ════════════════════════════════════════════════════════════════
#  5. Global: NPR1 vs NPR2 shape space (color = ring, shape = R/P)
#     + HOMA as size or annotation
# ════════════════════════════════════════════════════════════════
def fig_shape_space_global(df):
    fig, ax = plt.subplots(figsize=(10, 9))

    # Reference triangle
    ax.fill([0, 1, 0], [0, 1, 1], alpha=0.03, color='blue')
    ax.plot([0, 1, 0, 0], [0, 1, 1, 0], 'k-', linewidth=1.2, alpha=0.5)
    ax.text(0.05, 0.30, 'Sphere\n(3D)', fontsize=11, ha='center', style='italic', color='#3C5488', alpha=0.7)
    ax.text(0.72, 0.88, 'Disc\n(2D)',  fontsize=11, ha='center', style='italic', color='#00A087', alpha=0.7)
    ax.text(0.72, 0.12, 'Rod\n(1D)',  fontsize=11, ha='center', style='italic', color='#E64B35', alpha=0.7)

    for cat in RING_ORDER:
        sub = df[df['type1_en'] == cat]
        for stage in ['Reactant', 'Product']:
            sub2 = sub[sub['type2_en'] == stage]
            if len(sub2) == 0:
                continue
            ax.scatter(sub2['NPR1'], sub2['NPR2'],
                       s=MARKER_SIZE, alpha=0.55,
                       color=RING_COLORS.get(cat, '#888'),
                       marker=MARKERS[stage],
                       edgecolors='white', linewidth=0.4, zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('NPR1 (I\u2081/I\u2083)', fontsize=13)
    ax.set_ylabel('NPR2 (I\u2082/I\u2083)', fontsize=13)
    ax.set_title('Molecular Shape Space: NPR1 vs NPR2 \u2014 All Data',
                 fontsize=14, fontweight='bold', loc='left', pad=12)
    ax.set_aspect('equal')
    style_ax(ax)
    make_dual_legend(ax, RING_COLORS, RING_ORDER, 'Ring Family')

    plt.tight_layout()
    path = os.path.join(OUT, 'Clean_Fig05_shape_space_global.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig05: Shape Space Global done')


# ════════════════════════════════════════════════════════════════
#  6. Indole: NPR1 vs NPR2 shape space (color = reaction type)
# ════════════════════════════════════════════════════════════════
def fig_shape_space_indole(df):
    indole = df[df['type1'] == '吲哚']
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.fill([0, 1, 0], [0, 1, 1], alpha=0.03, color='blue')
    ax.plot([0, 1, 0, 0], [0, 1, 1, 0], 'k-', linewidth=1.2, alpha=0.5)
    ax.text(0.05, 0.30, 'Sphere\n(3D)', fontsize=11, ha='center', style='italic', color='#3C5488', alpha=0.7)
    ax.text(0.72, 0.88, 'Disc\n(2D)',  fontsize=11, ha='center', style='italic', color='#00A087', alpha=0.7)
    ax.text(0.72, 0.12, 'Rod\n(1D)',  fontsize=11, ha='center', style='italic', color='#E64B35', alpha=0.7)

    for cat in TYPE3_ORDER:
        sub = indole[indole['type3_en'] == cat]
        for stage in ['Reactant', 'Product']:
            sub2 = sub[sub['type2_en'] == stage]
            if len(sub2) == 0:
                continue
            ax.scatter(sub2['NPR1'], sub2['NPR2'],
                       s=30, alpha=0.6,
                       color=TYPE3_COLORS.get(cat, '#888'),
                       marker=MARKERS[stage],
                       edgecolors='white', linewidth=0.4, zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('NPR1 (I\u2081/I\u2083)', fontsize=13)
    ax.set_ylabel('NPR2 (I\u2082/I\u2083)', fontsize=13)
    ax.set_title('Indole: Molecular Shape Space by Reaction Type',
                 fontsize=14, fontweight='bold', loc='left', pad=12)
    ax.set_aspect('equal')
    style_ax(ax)
    make_dual_legend(ax, TYPE3_COLORS, TYPE3_ORDER, 'Reaction Type')

    plt.tight_layout()
    path = os.path.join(OUT, 'Clean_Fig06_shape_space_indole.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Clean Fig06: Shape Space Indole done')


# ════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('Fsp3 & NPR1 Standalone Figures')
    print('=' * 60)

    df = load_data()
    print(f'Data: {len(df)} rows')

    print('\nGenerating:')
    fig_fsp3_global(df)
    fig_npr1_global(df)
    fig_fsp3_indole(df)
    fig_npr1_indole(df)
    fig_shape_space_global(df)
    fig_shape_space_indole(df)

    print(f'\nDone! -> {OUT}/')


if __name__ == '__main__':
    main()
