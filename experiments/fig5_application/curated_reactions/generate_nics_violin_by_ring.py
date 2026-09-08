"""
Single figure: NICS-ZZ Violin + Box + Strip by Ring Family
All ring families in one plot, Reactant vs Product side by side.
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
    'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 10, 'ytick.labelsize': 12,
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

C_R = '#4DBBD5'
C_P = '#E64B35'


def load():
    df = pd.read_excel(XLSX)
    df['ring'] = df['type1'].map(CN_TO_EN)
    df['stage'] = df['type2'].map(CN_TO_EN_T2)
    return df


def main():
    df = load()

    # Only ring families with data
    ring_counts = df['ring'].value_counts()
    rings = [r for r in RING_ORDER if ring_counts.get(r, 0) > 0]
    n_rings = len(rings)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Each ring gets two adjacent positions: Reactant (left), Product (right)
    group_width = 0.8
    step = 1.5  # spacing between ring groups

    for gi, ring in enumerate(rings):
        sub = df[df['ring'] == ring]
        r_vals = sub[sub['stage'] == 'Reactant']['NICS_ZZ'].dropna().values
        p_vals = sub[sub['stage'] == 'Product']['NICS_ZZ'].dropna().values

        base = gi * step
        pos_r = base - 0.2
        pos_p = base + 0.2
        ring_color = RING_COLORS[ring]

        for vals, pos, stage_color, stage in [
            (r_vals, pos_r, C_R, 'Reactant'),
            (p_vals, pos_p, C_P, 'Product'),
        ]:
            if len(vals) == 0:
                continue

            # Violin
            if len(vals) >= 3:
                parts = ax.violinplot([vals], positions=[pos], widths=0.35,
                                      showmeans=False, showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(stage_color)
                    pc.set_alpha(0.2)
                    pc.set_edgecolor('none')

            # Box
            bp = ax.boxplot([vals], positions=[pos], widths=0.15,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color='#555', linewidth=0.9),
                            capprops=dict(color='#555', linewidth=0.9))
            for patch in bp['boxes']:
                patch.set_facecolor(stage_color)
                patch.set_alpha(0.8)
                patch.set_edgecolor('white')

            # Strip
            jitter = np.random.normal(0, 0.04, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       s=18, alpha=0.4, color=stage_color,
                       edgecolors='none', zorder=4)

        # Ring label below
        ax.text(base, ax.get_ylim()[0] - 3, ring, ha='center', va='top',
                fontsize=11, fontweight='bold', color=ring_color)

    # Axis
    ax.set_xticks([])
    ax.set_ylabel('NICS-ZZ (ppm)', fontsize=14)
    ax.set_title('NICS-ZZ by Ring Family: Reactant vs Product',
                 fontsize=15, fontweight='bold', loc='left', pad=14)
    ax.axhline(y=0, color='#aaa', linestyle=':', linewidth=1, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='y', alpha=0.12, linestyle='--')

    # Legend (below plot)
    stage_handles = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=C_R,
               markersize=12, label=f'Reactant (n={len(df[df["stage"]=="Reactant"])})'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=C_P,
               markersize=12, label=f'Product (n={len(df[df["stage"]=="Product"])})'),
    ]
    ax.legend(handles=stage_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=True, framealpha=0.95, edgecolor='#ccc',
              fontsize=12, title='Stage', title_fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(OUT, 'Single_NICS_violin_by_ring.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Done -> {path}')


if __name__ == '__main__':
    main()
