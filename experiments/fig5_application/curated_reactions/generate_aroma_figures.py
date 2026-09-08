"""
Aromaticity Reactant vs Product — Publication-quality figures
Pure aromaticity focus, no stereo descriptors.

Figures:
  1. Paired slope plot (dumbbell): HOMA / MBCO / NICS  — global
  2. Paired slope plot by reaction type
  3. Paired slope plot by ring family
  4. Indole-specific by reaction type
  5. Violin + box + strip composite
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.linewidth': 0.8,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

XLSX = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
OUT  = '/home/ubuntu/aroma-dps-code/figures'
os.makedirs(OUT, exist_ok=True)

# ── English labels for Chinese fields ────────────────────────────
TYPE3_EN = {'光催化': 'Photocatalysis', '氢化': 'Hydrogenation', '烯丙基去芳构化': 'Allylic Dearom.'}
TYPE1_EN = {'呋喃':'Furan','萘':'Naphthalene','萘2':'Naphthalene-2','吲哚':'Indole',
            '吡咯':'Pyrrole','苯酚':'Phenol','萘酚':'Naphthol'}
TYPE2_EN = {'反应物':'Reactant', '产物':'Product'}

TYPE3_ORDER = ['光催化', '氢化', '烯丙基去芳构化']
TYPE1_ORDER = ['呋喃', '萘', '萘2', '吲哚', '吡咯', '苯酚', '萘酚']

C_TYPE3 = {'光催化':'#3C5488', '氢化':'#00A087', '烯丙基去芳构化':'#E64B35'}
C_TYPE1 = {'呋喃':'#E64B35','萘':'#4DBBD5','萘2':'#00A087','吲哚':'#3C5488',
           '吡咯':'#F39B7F','苯酚':'#8491B4','萘酚':'#91D1C2'}
C_R = '#4DBBD5'   # reactant
C_P = '#E64B35'   # product


def load_paired():
    df = pd.read_excel(XLSX)
    pairs = []
    for t3 in df['type3'].unique():
        for t1 in df[df['type3']==t3]['type1'].unique():
            sub = df[(df['type3']==t3)&(df['type1']==t1)]
            rs = sub[sub['type2']=='反应物'].sort_values('New_ID')
            ps = sub[sub['type2']=='产物'].sort_values('New_ID')
            for i in range(min(len(rs), len(ps))):
                r, p = rs.iloc[i], ps.iloc[i]
                pairs.append({
                    'type3': t3, 'type1': t1,
                    'r_HOMA': r['HOMA'], 'p_HOMA': p['HOMA'],
                    'r_MBCO': r['MBCO'], 'p_MBCO': p['MBCO'],
                    'r_NICS': r['NICS_ZZ'], 'p_NICS': p['NICS_ZZ'],
                })
    return pd.DataFrame(pairs)


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.12, linestyle='--')


# ════════════════════════════════════════════════════════════════
#  Paired slope (dumbbell) plot
# ════════════════════════════════════════════════════════════════
def paired_slope_plot(ax, r_vals, p_vals, ylabel='', title='',
                      color_by=None, color_map=None, r_color=C_R, p_color=C_P,
                      show_pval=True):
    """
    Each pair: line from reactant (left) to product (right).
    Sorted by reactant value for clean visual.
    """
    n = len(r_vals)
    y_positions = np.arange(n)

    # Sort by reactant value descending
    order = np.argsort(-r_vals)
    r_sorted = r_vals[order]
    p_sorted = p_vals[order]

    if color_by is not None:
        cb_sorted = color_by[order]

    # Draw connecting lines
    for i in range(n):
        c = color_map.get(cb_sorted[i], '#aaa') if color_by is not None else '#bbb'
        ax.plot([0, 1], [r_sorted[i], p_sorted[i]], color=c, alpha=0.3, linewidth=0.8, zorder=1)

    # Draw points
    ax.scatter(np.zeros(n), r_sorted, s=22, color=r_color, edgecolors='white',
               linewidth=0.3, zorder=3, alpha=0.7, label='Reactant')
    ax.scatter(np.ones(n),  p_sorted, s=22, color=p_color, edgecolors='white',
               linewidth=0.3, zorder=3, alpha=0.7, label='Product')

    # Median lines
    ax.axhline(np.median(r_sorted), xmin=0, xmax=0.5, color=r_color, linestyle='--',
               linewidth=1.2, alpha=0.5)
    ax.axhline(np.median(p_sorted), xmin=0.5, xmax=1.0, color=p_color, linestyle='--',
               linewidth=1.2, alpha=0.5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Reactant', 'Product'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)

    if show_pval:
        try:
            _, pval = stats.wilcoxon(r_vals, p_vals)
            txt = f'p = {pval:.1e}' if pval < 0.001 else f'p = {pval:.3f}'
            ax.text(0.5, 0.02, txt, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                              edgecolor='#ccc', alpha=0.9))
        except:
            pass

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=8)
    style_ax(ax)


# ════════════════════════════════════════════════════════════════
#  Violin + box + strip composite
# ════════════════════════════════════════════════════════════════
def violin_box_strip(ax, r_vals, p_vals, ylabel='', title=''):
    """Violin + box + individual points for two groups"""
    from matplotlib.collections import PolyCollection

    data = [r_vals, p_vals]
    labels = ['Reactant', 'Product']

    # Violin
    parts = ax.violinplot(data, positions=[0, 1], widths=0.6, showmeans=False,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([C_R, C_P][i])
        pc.set_alpha(0.2)
        pc.set_edgecolor('none')

    # Box
    bp = ax.boxplot(data, positions=[0, 1], widths=0.25, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color='#555', linewidth=1),
                    capprops=dict(color='#555', linewidth=1))
    for patch, c in zip(bp['boxes'], [C_R, C_P]):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')

    # Strip
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.06, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.4,
                   color=[C_R, C_P][i], edgecolors='none', zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)

    try:
        _, pval = stats.wilcoxon(r_vals, p_vals)
        txt = f'p = {pval:.1e}' if pval < 0.001 else f'p = {pval:.3f}'
        ax.text(0.5, 0.97, txt, transform=ax.transAxes, ha='center', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                          edgecolor='#ccc', alpha=0.9))
    except:
        pass

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=8)
    style_ax(ax)


# ════════════════════════════════════════════════════════════════
#  Fig 1: Global paired slope plot (HOMA, MBCO, NICS)
# ════════════════════════════════════════════════════════════════
def fig1_global_paired_slope(paired):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', '(A) HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', '(B) MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', '(C) NICS-ZZ'),
    ]

    for ax, (rc, pc, yl, title) in zip(axes, metrics):
        paired_slope_plot(ax, paired[rc].values, paired[pc].values,
                          ylabel=yl, title=title)

    fig.suptitle('Aromaticity: Reactant vs Product (Paired Slope Plot)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT, 'Aroma_Fig01_global_paired_slope.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig01: Global Paired Slope done')


# ════════════════════════════════════════════════════════════════
#  Fig 2: Paired slope by reaction type (3 rows x 3 metrics)
# ════════════════════════════════════════════════════════════════
def fig2_paired_by_reaction_type(paired):
    fig, axes = plt.subplots(3, 3, figsize=(18, 20))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', 'HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', 'MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', 'NICS-ZZ'),
    ]

    for row, (rc, pc, yl, name) in enumerate(metrics):
        for col, t3 in enumerate(TYPE3_ORDER):
            ax = axes[row][col]
            sub = paired[paired['type3'] == t3]
            r_vals = sub[rc].values
            p_vals = sub[pc].values

            color_by = np.array([C_TYPE3.get(t, '#888') for t in sub['type3']])
            # Use single color per panel (all same type3)
            paired_slope_plot(ax, r_vals, p_vals,
                              ylabel=yl if col == 0 else '',
                              title=f'{"ABC"[row]}{col+1}) {name} - {TYPE3_EN[t3]} (n={len(sub)})')
            if col > 0:
                ax.set_ylabel('')

    fig.suptitle('Aromaticity Changes by Reaction Type (Paired Slope Plot)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Aroma_Fig02_by_reaction_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig02: By Reaction Type done')


# ════════════════════════════════════════════════════════════════
#  Fig 3: Paired slope by ring family
# ════════════════════════════════════════════════════════════════
def fig3_paired_by_ring(paired):
    # Only show ring families with >= 4 pairs
    ring_counts = paired['type1'].value_counts()
    rings = [r for r in TYPE1_ORDER if ring_counts.get(r, 0) >= 4]

    fig, axes = plt.subplots(len(rings), 3, figsize=(18, 4*len(rings)))
    if len(rings) == 1:
        axes = axes.reshape(1, -1)

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)'),
    ]

    for row, t1 in enumerate(rings):
        sub = paired[paired['type1'] == t1]
        for col, (rc, pc, yl) in enumerate(metrics):
            ax = axes[row][col]
            color_by = np.array([C_TYPE1.get(t1, '#888')] * len(sub))
            paired_slope_plot(ax, sub[rc].values, sub[pc].values,
                              ylabel=yl if col == 0 else '',
                              title=f'{TYPE1_EN[t1]} (n={len(sub)})' if col == 0 else '')
            if col > 0:
                ax.set_ylabel('')

    fig.suptitle('Aromaticity Changes by Ring Family (Paired Slope Plot)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Aroma_Fig03_by_ring_family.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig03: By Ring Family done')


# ════════════════════════════════════════════════════════════════
#  Fig 4: Indole-specific by reaction type
# ════════════════════════════════════════════════════════════════
def fig4_indole_by_type(paired):
    indole = paired[paired['type1'] == '吲哚']
    fig, axes = plt.subplots(3, 3, figsize=(18, 20))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', 'HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', 'MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', 'NICS-ZZ'),
    ]

    for row, (rc, pc, yl, name) in enumerate(metrics):
        for col, t3 in enumerate(TYPE3_ORDER):
            ax = axes[row][col]
            sub = indole[indole['type3'] == t3]
            if len(sub) < 2:
                ax.set_visible(False)
                continue
            paired_slope_plot(ax, sub[rc].values, sub[pc].values,
                              ylabel=yl if col == 0 else '',
                              title=f'{"ABC"[row]}{col+1}) {name} - {TYPE3_EN[t3]} (n={len(sub)})')
            if col > 0:
                ax.set_ylabel('')

    fig.suptitle('Indole: Aromaticity Changes by Reaction Type',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Aroma_Fig04_indole_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig04: Indole by Type done')


# ════════════════════════════════════════════════════════════════
#  Fig 5: Violin composite (global)
# ════════════════════════════════════════════════════════════════
def fig5_violin_composite(paired):
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', '(A) HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', '(B) MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', '(C) NICS-ZZ'),
    ]

    for ax, (rc, pc, yl, title) in zip(axes, metrics):
        violin_box_strip(ax, paired[rc].values, paired[pc].values,
                         ylabel=yl, title=title)

    fig.suptitle('Aromaticity Distribution: Reactant vs Product',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'Aroma_Fig05_violin_composite.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig05: Violin Composite done')


# ════════════════════════════════════════════════════════════════
#  Fig 6: Indole violin by reaction type
# ════════════════════════════════════════════════════════════════
def fig6_indole_violin_by_type(paired):
    indole = paired[paired['type1'] == '吲哚']
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', '(A) HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', '(B) MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', '(C) NICS-ZZ'),
    ]

    for ax, (rc, pc, yl, title) in zip(axes, metrics):
        # Create grouped violin: for each type3, show reactant and product side by side
        positions = []
        data_list = []
        colors_list = []
        labels_list = []

        for i, t3 in enumerate(TYPE3_ORDER):
            sub = indole[indole['type3'] == t3]
            if len(sub) < 2:
                continue
            # Reactant
            positions.append(i * 3)
            data_list.append(sub[rc].values)
            colors_list.append(C_TYPE3[t3])
            labels_list.append(f'{TYPE3_EN[t3]}\nR')
            # Product
            positions.append(i * 3 + 1)
            data_list.append(sub[pc].values)
            colors_list.append(C_TYPE3[t3])
            labels_list.append(f'P')

        # Plot violins
        parts = ax.violinplot(data_list, positions=positions, widths=0.8,
                              showmeans=False, showmedians=False, showextrema=False)
        for j, pc_body in enumerate(parts['bodies']):
            pc_body.set_facecolor(colors_list[j])
            pc_body.set_alpha(0.25 if j % 2 == 0 else 0.5)
            pc_body.set_edgecolor('none')

        # Box
        bp = ax.boxplot(data_list, positions=positions, widths=0.35,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', linewidth=1.2),
                        whiskerprops=dict(color='#555', linewidth=0.8),
                        capprops=dict(color='#555', linewidth=0.8))
        for patch, c in zip(bp['boxes'], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            patch.set_edgecolor('white')

        # Points
        for j, (vals, pos) in enumerate(zip(data_list, positions)):
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10, alpha=0.35,
                       color=colors_list[j], edgecolors='none', zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, fontsize=8.5)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        style_ax(ax)

    fig.suptitle('Indole: Aromaticity by Reaction Type (R = Reactant, P = Product)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'Aroma_Fig06_indole_violin_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig06: Indole Violin by Type done')


# ════════════════════════════════════════════════════════════════
#  Fig 7: Delta distribution (raincloud-style)
# ════════════════════════════════════════════════════════════════
def fig7_delta_distribution(paired):
    """Show delta values (reactant - product) as distribution + by type3"""
    paired = paired.copy()
    paired['dHOMA'] = paired['r_HOMA'] - paired['p_HOMA']
    paired['dMBCO'] = paired['r_MBCO'] - paired['p_MBCO']
    paired['dNICS'] = paired['p_NICS'] - paired['r_NICS']  # reversed for NICS

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    metrics = [
        ('dHOMA', 'dHOMA', '(A) dHOMA', True),
        ('dMBCO', 'dMBCO', '(B) dMBCO', True),
        ('dNICS', 'dNICS', '(C) dNICS', True),
    ]

    for ax, (col, yl, title, _) in zip(axes, metrics):
        # Half-violin on top, box below, points
        for i, t3 in enumerate(TYPE3_ORDER):
            sub = paired[paired['type3'] == t3]
            vals = sub[col].dropna().values
            if len(vals) < 2:
                continue

            pos = i
            c = C_TYPE3[t3]

            # Violin (half)
            if len(vals) >= 3:
                kde_x = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 50)
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(vals)
                    density = kde(kde_x)
                    # Scale to fit
                    density = density / density.max() * 0.3
                    ax.fill_betweenx(kde_x, pos, pos + density, color=c, alpha=0.3)
                except:
                    pass

            # Box
            bp = ax.boxplot([vals], positions=[pos], widths=0.2, patch_artist=True,
                            showfliers=False, vert=True,
                            medianprops=dict(color='white', linewidth=1.2),
                            whiskerprops=dict(color=c, linewidth=1),
                            capprops=dict(color=c, linewidth=1))
            for patch in bp['boxes']:
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
                patch.set_edgecolor('white')

            # Points
            jitter = np.random.normal(0, 0.04, len(vals))
            ax.scatter(np.full(len(vals), pos - 0.15) + jitter, vals,
                       s=16, alpha=0.5, color=c, edgecolors='none', zorder=4)

        ax.set_xticks(range(len(TYPE3_ORDER)))
        ax.set_xticklabels([TYPE3_EN[t] for t in TYPE3_ORDER], fontsize=10)
        ax.set_ylabel(yl)
        ax.axhline(y=0, color='#888', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')

        # KW test
        groups = [paired[paired['type3']==t][col].dropna().values for t in TYPE3_ORDER]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 2:
            try:
                _, pval = stats.kruskal(*groups)
                ax.text(0.97, 0.97, f'KW p = {pval:.3f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                                  edgecolor='#ccc', alpha=0.9))
            except:
                pass
        style_ax(ax)

    fig.suptitle('Aromaticity Loss Distribution by Reaction Type',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, 'Aroma_Fig07_delta_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig07: Delta Distribution done')


# ════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('Aromaticity Reactant vs Product — Figure Generation')
    print('=' * 60)

    paired = load_paired()
    print(f'Loaded {len(paired)} reaction pairs')

    print('\nGenerating figures:')
    fig1_global_paired_slope(paired)
    fig2_paired_by_reaction_type(paired)
    fig3_paired_by_ring(paired)
    fig4_indole_by_type(paired)
    fig5_violin_composite(paired)
    fig6_indole_violin_by_type(paired)
    fig7_delta_distribution(paired)

    print(f'\nDone! -> {OUT}/')


if __name__ == '__main__':
    main()
