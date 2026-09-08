"""
Clean standalone: Fsp3 vs NICS-ZZ and NPR1 vs NICS-ZZ
Single large panel each, no fit line, clear legend outside plot area.
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
    'font.size': 12, 'axes.linewidth': 1.0,
    'axes.labelsize': 15, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

XLSX = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
OUT  = '/home/ubuntu/aroma-dps-code/figures'

RING_COLORS = {
    'Furan':'#E64B35', 'Naphthalene':'#4DBBD5', 'Naphthalene-2':'#00A087',
    'Indole':'#3C5488', 'Pyrrole':'#F39B7F', 'Phenol':'#8491B4',
    'Naphthol':'#91D1C2',
}
RING_ORDER = ['Furan','Naphthalene','Naphthalene-2','Indole','Pyrrole','Phenol','Naphthol']
CN_TO_EN = {'呋喃':'Furan','萘':'Naphthalene','萘2':'Naphthalene-2','吲哚':'Indole',
            '吡咯':'Pyrrole','苯酚':'Phenol','萘酚':'Naphthol'}
CN_TO_EN_T2 = {'反应物':'Reactant', '产物':'Product'}

MARKERS = {'Reactant': 'o', 'Product': '^'}
PT_SIZE = 200


def load():
    df = pd.read_excel(XLSX)
    df['ring'] = df['type1'].map(CN_TO_EN)
    df['stage'] = df['type2'].map(CN_TO_EN_T2)
    return df


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.1, linestyle='--')


def draw_scatter(ax, df, x_col, xlabel):
    for ring in RING_ORDER:
        sub = df[df['ring'] == ring]
        for stage in ['Reactant', 'Product']:
            s = sub[sub['stage'] == stage]
            if len(s) == 0:
                continue
            ax.scatter(s[x_col], s['NICS_ZZ'],
                       s=PT_SIZE, alpha=0.35,
                       color=RING_COLORS[ring],
                       marker=MARKERS[stage],
                       edgecolors='white', linewidth=0.8, zorder=3)

    ax.axhline(y=0, color='#bbb', linestyle=':', linewidth=1, alpha=0.4)

    # Pearson r in corner (small, unobtrusive)
    valid = df.dropna(subset=[x_col, 'NICS_ZZ'])
    if len(valid) >= 8:
        r, p = stats.pearsonr(valid[x_col], valid['NICS_ZZ'])
        txt = f'r = {r:.3f},  p = {p:.1e}' if p < 0.001 else f'r = {r:.3f},  p = {p:.3f}'
        ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9, linewidth=0.7))

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel('NICS-ZZ (ppm)', fontsize=15)
    style(ax)


def make_legend_below(fig, x_col):
    """Place legends below the plot so they don't overlap data"""
    # Color legend
    color_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RING_COLORS[r],
               markersize=13, markeredgecolor='white', markeredgewidth=0.6, label=r)
        for r in RING_ORDER
    ]
    shape_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
               markersize=13, markeredgecolor='white', markeredgewidth=0.6, label='Reactant'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='gray',
               markersize=13, markeredgecolor='white', markeredgewidth=0.6, label='Product'),
    ]

    leg1 = fig.legend(handles=color_handles, title='Ring Family',
                      loc='lower center', ncol=len(RING_ORDER),
                      frameon=True, framealpha=0.95, edgecolor='#ccc',
                      fontsize=12, title_fontsize=13,
                      handletextpad=0.5, borderpad=0.7, labelspacing=0.5,
                      columnspacing=1.5,
                      bbox_to_anchor=(0.5, 0.07))
    fig.add_artist(leg1)
    fig.legend(handles=shape_handles, title='Stage',
               loc='lower center', ncol=2,
               frameon=True, framealpha=0.95, edgecolor='#ccc',
               fontsize=12, title_fontsize=13,
               handletextpad=0.5, borderpad=0.7,
               columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.005))


# ════════════════════════════════════════════════════════════════
def main():
    df = load()
    print(f'Data: {len(df)} rows')

    # ── Fig 1: Fsp3 vs NICS-ZZ ──
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_scatter(ax, df, 'Fsp3', 'Fsp\u00b3')
    ax.set_title('NICS-ZZ vs Fsp\u00b3  (All Data)',
                 fontsize=16, fontweight='bold', loc='left', pad=14)
    make_legend_below(fig, 'Fsp3')
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.18)
    path = os.path.join(OUT, 'Single_Fsp3_vs_NICS.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Fsp3 vs NICS -> {path}')

    # ── Fig 2: NPR1 vs NICS-ZZ ──
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_scatter(ax, df, 'NPR1', 'NPR1 (I\u2081/I\u2083)')
    ax.set_title('NICS-ZZ vs NPR1  (All Data)',
                 fontsize=16, fontweight='bold', loc='left', pad=14)
    make_legend_below(fig, 'NPR1')
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.18)
    path = os.path.join(OUT, 'Single_NPR1_vs_NICS.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  NPR1 vs NICS -> {path}')

    print('Done!')


if __name__ == '__main__':
    main()
