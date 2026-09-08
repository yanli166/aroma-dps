"""
Publication-quality ring distribution figure
Replaces 05b_ring_distribution_pie.png with a top-journal aesthetic version

Panels:
  A. Donut chart: Target Ring Size distribution (with center label, grouped rare sizes)
  B. Box + strip plot: HOMA by ring size (aromaticity context)
  C. Scatter: NICS-ZZ vs HOMA colored by ring size (electronic property context)
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
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

DATASET_PATH = os.path.join(_PROJ_ROOT, "unified_models", "nics-nics1zz-out-no3.csv")
OUTPUT_DIR = os.path.join(_PROJ_ROOT, "unified_models", "0427_unified_results/chemical_space")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette (Nature/Science-inspired) ──────────────────────────────
# Distinct, harmonious colors for ring sizes
RING_COLORS = {
    3:  '#E64B35',  # red
    4:  '#4DBBD5',  # cyan
    5:  '#00A087',  # teal
    6:  '#3C5488',  # navy
    7:  '#F39B7F',  # peach
    8:  '#8491B4',  # muted purple
    9:  '#91D1C2',  # light teal
    10: '#DC0000',  # dark red
    18: '#7E6148',  # brown
    -1: '#B09C85',  # grey-brown for "Others"
}
# Fallback palette for any unexpected ring sizes
FALLBACK_PALETTE = ['#E64B35','#4DBBD5','#00A087','#3C5488','#F39B7F',
                    '#8491B4','#91D1C2','#DC0000','#7E6148','#B09C85']


def get_color(ring_size, idx=0):
    return RING_COLORS.get(ring_size, FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)])


def plot_panel_a_donut(ax, df):
    """Panel A: Donut chart of ring size distribution."""
    ring_counts = df['Ring_Size'].value_counts().sort_index()
    total = ring_counts.sum()

    # Group rare sizes (< 1%) into "Others"
    threshold = total * 0.01
    main_sizes = ring_counts[ring_counts >= threshold]
    rare_sizes = ring_counts[ring_counts < threshold]

    labels = []
    sizes = []
    colors = []

    for size, count in main_sizes.items():
        labels.append(f'{size}-membered')
        sizes.append(count)
        colors.append(get_color(size))

    if len(rare_sizes) > 0:
        rare_total = rare_sizes.sum()
        rare_detail = ', '.join([f'{s}({c})' for s, c in rare_sizes.items()])
        labels.append(f'Others\n({rare_detail})')
        sizes.append(rare_total)
        colors.append(get_color(-1))

    sizes = np.array(sizes)
    colors = np.array(colors)

    # Draw donut
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=1.5),
    )

    # Center label
    ax.text(0, 0.08, f'{total:,}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#333333')
    ax.text(0, -0.12, 'Ring\nEntries', ha='center', va='center',
            fontsize=10, color='#666666')

    # Legend with count + percentage
    legend_labels = []
    for i, (label, count) in enumerate(zip(labels, sizes)):
        pct = count / total * 100
        legend_labels.append(f'{label}  —  {count:,} ({pct:.1f}%)')

    leg = ax.legend(
        wedges, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9,
        labelspacing=0.8,
        handlelength=1.2,
        handleheight=1.2,
    )

    ax.set_title('(A)  Target Ring Size Distribution',
                 fontsize=13, fontweight='bold', pad=15, loc='left',
                 color='#222222')


def plot_panel_b_homa(ax, df):
    """Panel B: Box + strip plot of HOMA by ring size."""
    # Use only ring sizes with enough data
    ring_counts = df['Ring_Size'].value_counts()
    sizes_to_plot = sorted(ring_counts[ring_counts >= 10].index)

    box_data = []
    positions = []
    box_colors = []

    for i, size in enumerate(sizes_to_plot):
        vals = df[df['Ring_Size'] == size]['homa_value'].values
        box_data.append(vals)
        positions.append(i)
        box_colors.append(get_color(size))

    # Box plot
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='white', linewidth=1.5),
        whiskerprops=dict(color='#555555', linewidth=1),
        capprops=dict(color='#555555', linewidth=1),
    )
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('white')
        patch.set_linewidth(1)

    # Strip plot (jittered points) for sizes with manageable counts
    for i, size in enumerate(sizes_to_plot):
        vals = df[df['Ring_Size'] == size]['homa_value'].values
        if len(vals) > 300:
            # Sample for clarity
            idx = np.random.choice(len(vals), 300, replace=False)
            vals = vals[idx]
        jitter = np.random.normal(0, 0.08, len(vals))
        ax.scatter(
            positions[i] + jitter, vals,
            s=6, alpha=0.25, color=box_colors[i],
            edgecolors='none', zorder=3,
        )

    # Reference lines
    ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(y=0.5, color='#E64B35', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

    ax.text(len(sizes_to_plot) - 0.3, 0.6, 'HOMA = 0.5\n(borderline\naromaticity)',
            fontsize=7.5, color='#E64B35', alpha=0.7, va='bottom', ha='right')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{s}-ring' for s in sizes_to_plot])
    ax.set_xlabel('Ring Size', fontsize=11, labelpad=8)
    ax.set_ylabel('HOMA Value', fontsize=11, labelpad=8)

    # Add sample sizes on top
    for i, size in enumerate(sizes_to_plot):
        n = ring_counts[size]
        ax.text(i, ax.get_ylim()[1] + 0.5, f'n={n}',
                ha='center', va='bottom', fontsize=7.5, color='#666666')

    ax.set_title('(B)  Aromaticity (HOMA) by Ring Size',
                 fontsize=13, fontweight='bold', pad=15, loc='left',
                 color='#222222')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')


def plot_panel_c_nics_homa(ax, df):
    """Panel C: Scatter of NICS-ZZ vs HOMA colored by ring size."""
    ring_counts = df['Ring_Size'].value_counts()
    sizes_to_plot = sorted(ring_counts[ring_counts >= 10].index)

    # Subsample for visual clarity if needed
    plot_df = df[df['Ring_Size'].isin(sizes_to_plot)].copy()
    if len(plot_df) > 3000:
        plot_df = plot_df.sample(3000, random_state=42)

    for size in sizes_to_plot:
        subset = plot_df[plot_df['Ring_Size'] == size]
        ax.scatter(
            subset['Ring_NICS_ZZ_1'], subset['homa_value'],
            s=12, alpha=0.45, color=get_color(size),
            edgecolors='none', label=f'{size}-ring',
            zorder=3,
        )

    # Reference lines
    ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
    ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + 1, ylim[1] - 1, 'Aromatic\n(low NICS,\nhigh HOMA)',
            fontsize=7.5, color='#00A087', alpha=0.6,
            va='top', ha='left', style='italic')
    ax.text(xlim[1] - 1, ylim[0] + 1, 'Non-aromatic\n(high NICS,\nlow HOMA)',
            fontsize=7.5, color='#E64B35', alpha=0.6,
            va='bottom', ha='right', style='italic')

    # Pearson correlation
    r, p = stats.pearsonr(df['Ring_NICS_ZZ_1'], df['homa_value'])
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np < 0.001' if p < 0.001 else f'Pearson r = {r:.3f}\np = {p:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', linewidth=0.8, alpha=0.9))

    ax.set_xlabel('NICS-ZZ (ppm)', fontsize=11, labelpad=8)
    ax.set_ylabel('HOMA Value', fontsize=11, labelpad=8)

    ax.legend(
        loc='lower right',
        frameon=True, framealpha=0.9,
        edgecolor='#cccccc', fancybox=True,
        fontsize=8, labelspacing=0.4,
        handletextpad=0.3, borderpad=0.5,
        title='Ring Size', title_fontsize=8.5,
    )

    ax.set_title('(C)  NICS-ZZ vs HOMA Correlation',
                 fontsize=13, fontweight='bold', pad=15, loc='left',
                 color='#222222')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15, linestyle='--')


def main():
    print("=" * 60)
    print("Generating publication-quality ring distribution figure")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {len(df)} rows")

    # ── Create figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.1], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_panel_a_donut(ax_a, df)
    plot_panel_b_homa(ax_b, df)
    plot_panel_c_nics_homa(ax_c, df)

    # Main title
    fig.suptitle(
        'Ring Distribution & Aromaticity Analysis of the Dataset',
        fontsize=15, fontweight='bold', y=1.03, color='#222222',
    )

    # Use subplots_adjust instead of tight_layout (pie + legend doesn't play well)
    plt.subplots_adjust(left=0.04, right=0.97, bottom=0.12, top=0.88, wspace=0.35)

    out_path = os.path.join(OUTPUT_DIR, '05b_ring_distribution_pie.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
