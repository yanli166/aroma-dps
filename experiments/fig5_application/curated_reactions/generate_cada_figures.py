"""
CADA 芳香性变化与立体程度分析 — 顶刊级多面板图
基于 参考1.txt 的 19 张图方案 + 吲哚固定底物 type3 对比

输出目录: /home/ubuntu/aroma-dps-code/figures/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')

# ── 全局样式 ──────────────────────────────────────────────────────
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

XLSX_PATH = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
OUTPUT_DIR = '/home/ubuntu/aroma-dps-code/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 配色 ──────────────────────────────────────────────────────────
COLORS = {
    'reactant': '#4DBBD5',   # cyan
    'product':  '#E64B35',   # red
    '光催化':    '#3C5488',   # navy
    '氢化':      '#00A087',   # teal
    '烯丙基去芳构化': '#F39B7F',  # peach
}

RING_PALETTE = {
    '呋喃': '#E64B35', '萘': '#4DBBD5', '萘2': '#00A087',
    '吲哚': '#3C5488', '吡咯': '#F39B7F', '苯酚': '#8491B4',
    '萘酚': '#91D1C2',
}

TYPE3_ORDER = ['光催化', '氢化', '烯丙基去芳构化']
TYPE1_ORDER = ['呋喃', '萘', '萘2', '吲哚', '吡咯', '苯酚', '萘酚']

# 统一指标显示名
LABELS = {
    'HOMA': 'HOMA', 'MBCO': 'MBCO', 'NICS_ZZ': 'NICS-ZZ (ppm)',
    'Ring_RPD': 'Ring RPD (Å)', 'Mol_PBF': 'Mol PBF (Å)',
    'Fsp3': 'Fsp³', 'Ring_Cremer_Pople_Q': 'Cremer-Pople Q (Å)',
}

DELTA_LABELS = {
    'dHOMA': 'ΔHOMA', 'dMBCO': 'ΔMBCO', 'dNICS': 'ΔNICS (ppm)',
    'dRPD': 'ΔRPD (Å)', 'dPBF': 'ΔPBF (Å)', 'dFsp3': 'ΔFsp³',
}


# ════════════════════════════════════════════════════════════════
#  数据配对与 Δ 计算
# ════════════════════════════════════════════════════════════════
def load_and_pair():
    """读取数据，配对反应物-产物，计算所有 Δ 值"""
    df = pd.read_excel(XLSX_PATH)
    pairs = []

    for t3 in df['type3'].unique():
        for t1 in df[df['type3'] == t3]['type1'].unique():
            sub = df[(df['type3'] == t3) & (df['type1'] == t1)].copy()
            reactants = sub[sub['type2'] == '反应物'].sort_values('New_ID')
            products  = sub[sub['type2'] == '产物'].sort_values('New_ID')
            n = min(len(reactants), len(products))
            for i in range(n):
                r = reactants.iloc[i]
                p = products.iloc[i]
                pair = {
                    'Pair_ID': f'{t3}_{t1}_{i+1}',
                    'type3': t3, 'type1': t1,
                    'r_SMILES': r['SMILES'], 'p_SMILES': p['SMILES'],
                    'r_New_ID': r['New_ID'], 'p_New_ID': p['New_ID'],
                    'r_HOMA': r['HOMA'], 'p_HOMA': p['HOMA'],
                    'r_MBCO': r['MBCO'], 'p_MBCO': p['MBCO'],
                    'r_NICS': r['NICS_ZZ'], 'p_NICS': p['NICS_ZZ'],
                    'r_RPD': r['Ring_RPD'], 'p_RPD': p['Ring_RPD'],
                    'r_PBF': r['Mol_PBF'], 'p_PBF': p['Mol_PBF'],
                    'r_Fsp3': r['Fsp3'], 'p_Fsp3': p['Fsp3'],
                    'r_Q': r['Ring_Cremer_Pople_Q'], 'p_Q': p['Ring_Cremer_Pople_Q'],
                    'r_NPR1': r['NPR1'], 'p_NPR1': p['NPR1'],
                    'r_NPR2': r['NPR2'], 'p_NPR2': p['NPR2'],
                    'r_Asp': r['Asphericity'], 'p_Asp': p['Asphericity'],
                }
                # Δ 值: 正值 = 芳香性损失 / 结构增加
                pair['dHOMA'] = r['HOMA'] - p['HOMA']
                pair['dMBCO'] = r['MBCO'] - p['MBCO']
                pair['dNICS'] = p['NICS_ZZ'] - r['NICS_ZZ']  # NICS 越负越芳香, 反向
                pair['dRPD']  = p['Ring_RPD'] - r['Ring_RPD']
                pair['dPBF']  = p['Mol_PBF'] - r['Mol_PBF']
                pair['dFsp3'] = p['Fsp3'] - r['Fsp3']
                pair['dQ']    = p['Ring_Cremer_Pople_Q'] - r['Ring_Cremer_Pople_Q']
                pairs.append(pair)

    paired = pd.DataFrame(pairs)
    print(f"配对完成: {len(paired)} 个反应对")
    print(f"  type3 分布: {paired['type3'].value_counts().to_dict()}")
    print(f"  type1 分布: {paired['type1'].value_counts().to_dict()}")
    return paired


# ════════════════════════════════════════════════════════════════
#  辅助绘图函数
# ════════════════════════════════════════════════════════════════
def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, linestyle='--')

def paired_boxplot(ax, r_vals, p_vals, ylabel='', title=''):
    """配对箱线图 + 散点 + 连接线"""
    data = [r_vals.dropna(), p_vals.dropna()]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color='#555', linewidth=1),
                    capprops=dict(color='#555', linewidth=1))
    for patch, color in zip(bp['boxes'], [COLORS['reactant'], COLORS['product']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')

    # 配对连接线
    for r, p in zip(r_vals, p_vals):
        if not (np.isnan(r) or np.isnan(p)):
            ax.plot([0, 1], [r, p], color='#888', alpha=0.15, linewidth=0.5, zorder=1)

    # 散点
    for i, vals in enumerate([r_vals, p_vals]):
        jitter = np.random.normal(0, 0.04, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, alpha=0.4,
                   color=[COLORS['reactant'], COLORS['product']][i],
                   edgecolors='none', zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Reactant', 'Product'])
    ax.set_ylabel(ylabel)

    # Wilcoxon 配对检验
    valid_mask = ~(r_vals.isna() | p_vals.isna())
    if valid_mask.sum() >= 5:
        try:
            stat, pval = stats.wilcoxon(r_vals[valid_mask], p_vals[valid_mask])
            ax.text(0.5, 0.97, f'p = {pval:.2e}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9))
        except Exception:
            pass

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    style_ax(ax)


def grouped_boxplot(ax, data_dict, ylabel='', title='', colors=None):
    """分组箱线图 + 散点"""
    labels = list(data_dict.keys())
    data = [data_dict[l].dropna() for l in labels]
    if colors is None:
        colors = [COLORS.get(l, '#888') for l in labels]

    bp = ax.boxplot(data, positions=range(len(labels)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color='#555', linewidth=1),
                    capprops=dict(color='#555', linewidth=1))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')

    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, alpha=0.4,
                   color=colors[i], edgecolors='none', zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)

    # Kruskal-Wallis 检验
    valid = [d for d in data if len(d) >= 3]
    if len(valid) >= 2:
        try:
            stat, pval = stats.kruskal(*valid)
            ax.text(0.98, 0.97, f'KW p = {pval:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9))
        except Exception:
            pass

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    style_ax(ax)


def scatter_with_fit(ax, x, y, xlabel='', ylabel='', title='',
                     color_by=None, color_map=None):
    """散点 + 线性拟合 + Pearson r"""
    valid_mask = ~(x.isna() | y.isna())
    x_v, y_v = x[valid_mask], y[valid_mask]

    if color_by is not None and color_map is not None:
        for cat in x_v.index.map(lambda i: paired_df.loc[i, color_by]).unique():
            mask = paired_df.loc[x_v.index, color_by] == cat
            ax.scatter(x_v[mask], y_v[mask], s=18, alpha=0.5,
                       color=color_map.get(cat, '#888'),
                       edgecolors='none', label=cat, zorder=3)
        ax.legend(fontsize=7.5, frameon=True, framealpha=0.9,
                  edgecolor='#ccc', loc='best')
    else:
        ax.scatter(x_v, y_v, s=18, alpha=0.5, color='#3C5488',
                   edgecolors='none', zorder=3)

    # 线性拟合
    if len(x_v) >= 5:
        slope, intercept, r, pval, se = stats.linregress(x_v, y_v)
        x_fit = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='#E64B35',
                linewidth=1.5, alpha=0.8, zorder=4)
        ax.text(0.05, 0.97, f'r = {r:.3f}\np = {pval:.2e}',
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9))

    # 零线
    ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axvline(x=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    style_ax(ax)


# ════════════════════════════════════════════════════════════════
#  图 1: 反应物 vs 产物配对比较 (HOMA, MBCO, NICS, RPD, PBF, Fsp3)
# ════════════════════════════════════════════════════════════════
def fig01_reactant_vs_product(paired_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', '(A) HOMA'),
        ('r_MBCO', 'p_MBCO', 'MBCO', '(B) MBCO'),
        ('r_NICS', 'p_NICS', 'NICS-ZZ (ppm)', '(C) NICS-ZZ'),
        ('r_RPD',  'p_RPD',  'Ring RPD (Å)', '(D) Ring RPD'),
        ('r_PBF',  'p_PBF',  'Mol PBF (Å)',  '(E) Mol PBF'),
        ('r_Fsp3', 'p_Fsp3', 'Fsp³',         '(F) Fsp³'),
    ]

    for ax, (rc, pc, yl, title) in zip(axes.flat, metrics):
        paired_boxplot(ax, paired_df[rc], paired_df[pc], ylabel=yl, title=title)

    fig.suptitle('Reactant vs Product: Aromaticity & 3D Structure',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'Fig01_reactant_vs_product.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig01: Reactant vs Product ✓')


# ════════════════════════════════════════════════════════════════
#  图 2: Δ 按反应类型分组 (HOMA, MBCO, NICS, RPD, PBF, Fsp3)
# ════════════════════════════════════════════════════════════════
def fig02_delta_by_reaction_type(paired_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ('dHOMA', 'ΔHOMA', '(A) ΔHOMA by Reaction Type'),
        ('dMBCO', 'ΔMBCO', '(B) ΔMBCO by Reaction Type'),
        ('dNICS', 'ΔNICS (ppm)', '(C) ΔNICS by Reaction Type'),
        ('dRPD',  'ΔRPD (Å)', '(D) ΔRPD by Reaction Type'),
        ('dPBF',  'ΔPBF (Å)', '(E) ΔPBF by Reaction Type'),
        ('dFsp3', 'ΔFsp³',   '(F) ΔFsp³ by Reaction Type'),
    ]

    for ax, (col, yl, title) in zip(axes.flat, metrics):
        data_dict = {}
        for t3 in TYPE3_ORDER:
            vals = paired_df[paired_df['type3'] == t3][col]
            if len(vals) > 0:
                data_dict[t3] = vals
        grouped_boxplot(ax, data_dict, ylabel=yl, title=title)
        ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    fig.suptitle('Aromaticity Loss & Structural Change by Reaction Type',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'Fig02_delta_by_reaction_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig02: Δ by Reaction Type ✓')


# ════════════════════════════════════════════════════════════════
#  图 3: Δ 按环系分组 (HOMA, MBCO, NICS)
# ════════════════════════════════════════════════════════════════
def fig03_delta_by_ring_type(paired_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('dHOMA', 'ΔHOMA', '(A) ΔHOMA by Ring Family'),
        ('dMBCO', 'ΔMBCO', '(B) ΔMBCO by Ring Family'),
        ('dNICS', 'ΔNICS (ppm)', '(C) ΔNICS by Ring Family'),
    ]

    for ax, (col, yl, title) in zip(axes.flat, metrics):
        data_dict = {}
        for t1 in TYPE1_ORDER:
            vals = paired_df[paired_df['type1'] == t1][col]
            if len(vals) > 0:
                data_dict[t1] = vals
        colors = [RING_PALETTE.get(t, '#888') for t in data_dict.keys()]
        grouped_boxplot(ax, data_dict, ylabel=yl, title=title, colors=colors)
        ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    fig.suptitle('Aromaticity Loss by Ring Family',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig03_delta_by_ring_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig03: Δ by Ring Type ✓')


# ════════════════════════════════════════════════════════════════
#  图 4: Δ 芳香性指标间相关性
# ════════════════════════════════════════════════════════════════
def fig04_aromaticity_correlations(paired_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    scatter_with_fit(axes[0], paired_df['dHOMA'], paired_df['dMBCO'],
                     xlabel='ΔHOMA', ylabel='ΔMBCO',
                     title='(A) ΔHOMA vs ΔMBCO')

    scatter_with_fit(axes[1], paired_df['dHOMA'], paired_df['dNICS'],
                     xlabel='ΔHOMA', ylabel='ΔNICS (ppm)',
                     title='(B) ΔHOMA vs ΔNICS')

    fig.suptitle('Concordance of Aromaticity Descriptors',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig04_aromaticity_correlations.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig04: Aromaticity Correlations ✓')


# ════════════════════════════════════════════════════════════════
#  图 5: ΔHOMA vs Δ 结构指标
# ════════════════════════════════════════════════════════════════
def fig05_aromaticity_vs_structure(paired_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    scatter_with_fit(axes[0], paired_df['dHOMA'], paired_df['dRPD'],
                     xlabel='ΔHOMA', ylabel='ΔRPD (Å)',
                     title='(A) ΔHOMA vs ΔRPD')

    scatter_with_fit(axes[1], paired_df['dHOMA'], paired_df['dPBF'],
                     xlabel='ΔHOMA', ylabel='ΔPBF (Å)',
                     title='(B) ΔHOMA vs ΔPBF')

    scatter_with_fit(axes[2], paired_df['dHOMA'], paired_df['dFsp3'],
                     xlabel='ΔHOMA', ylabel='ΔFsp³',
                     title='(C) ΔHOMA vs ΔFsp³')

    fig.suptitle('Aromaticity Loss vs Structural Reorganization',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig05_aromaticity_vs_structure.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig05: Aromaticity vs Structure ✓')


# ════════════════════════════════════════════════════════════════
#  图 6: ΔHOMA vs ΔRPD 按反应类型着色 (主文核心图)
# ════════════════════════════════════════════════════════════════
def fig06_delta_homa_vs_rpd_by_type(paired_df):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for t3 in TYPE3_ORDER:
        sub = paired_df[paired_df['type3'] == t3]
        ax.scatter(sub['dHOMA'], sub['dRPD'], s=22, alpha=0.6,
                   color=COLORS.get(t3, '#888'), edgecolors='white',
                   linewidth=0.3, label=f'{t3} (n={len(sub)})', zorder=3)

    # 总体拟合
    valid = paired_df.dropna(subset=['dHOMA', 'dRPD'])
    if len(valid) >= 5:
        slope, intercept, r, pval, se = stats.linregress(valid['dHOMA'], valid['dRPD'])
        x_fit = np.linspace(valid['dHOMA'].min(), valid['dHOMA'].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                linewidth=1.5, alpha=0.7, zorder=4, label=f'Fit: r={r:.3f}, p={pval:.1e}')

    ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axvline(x=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    # 象限标注
    ax.text(0.97, 0.97, 'Aromaticity loss\n+ Ring puckering',
            transform=ax.transAxes, fontsize=8, color='#E64B35', alpha=0.6,
            ha='right', va='top', style='italic')
    ax.text(0.03, 0.03, 'Low aromaticity loss\n+ Flat ring',
            transform=ax.transAxes, fontsize=8, color='#3C5488', alpha=0.6,
            ha='left', va='bottom', style='italic')

    ax.set_xlabel('ΔHOMA (aromaticity loss →)', fontsize=12)
    ax.set_ylabel('ΔRPD (ring non-planarity →)', fontsize=12)
    ax.set_title('Aromaticity Loss vs Ring Puckering\nby Reaction Type',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9,
              edgecolor='#ccc', loc='lower right')
    style_ax(ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig06_deltaHOMA_vs_deltaRPD_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig06: ΔHOMA vs ΔRPD by Type ✓')


# ════════════════════════════════════════════════════════════════
#  图 7: NPR1 vs NPR2 形状空间 (反应物 vs 产物)
# ════════════════════════════════════════════════════════════════
def fig07_shape_space(paired_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (rc, pc, title) in zip(axes,
            [('r_NPR1', 'r_NPR2', 'Reactants'),
             ('p_NPR1', 'p_NPR2', 'Products')]):

        # 参考三角形
        ax.fill([0, 1, 0], [0, 1, 1], alpha=0.04, color='blue')
        ax.plot([0, 1, 0, 0], [0, 1, 1, 0], 'k-', linewidth=1)
        ax.text(0.05, 0.35, 'Sphere', fontsize=9, ha='center', style='italic', color='blue')
        ax.text(0.7, 0.85, 'Disc', fontsize=9, ha='center', style='italic', color='green')
        ax.text(0.7, 0.15, 'Rod', fontsize=9, ha='center', style='italic', color='red')

        # 散点
        for t3 in TYPE3_ORDER:
            sub = paired_df[paired_df['type3'] == t3]
            ax.scatter(sub[rc], sub[pc], s=18, alpha=0.5,
                       color=COLORS.get(t3, '#888'), edgecolors='none',
                       label=t3)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('NPR1 (I₁/I₃)', fontsize=11)
        ax.set_ylabel('NPR2 (I₂/I₃)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.grid(alpha=0.15, linestyle='--')

    fig.suptitle('Molecular Shape Space (NPR1 vs NPR2)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig07_shape_space.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig07: Shape Space ✓')


# ════════════════════════════════════════════════════════════════
#  图 8: 数据组成矩阵 (ring type × reaction mode)
# ════════════════════════════════════════════════════════════════
def fig08_composition_matrix(paired_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    matrix = np.zeros((len(TYPE1_ORDER), len(TYPE3_ORDER)))
    for i, t1 in enumerate(TYPE1_ORDER):
        for j, t3 in enumerate(TYPE3_ORDER):
            matrix[i, j] = len(paired_df[(paired_df['type1'] == t1) &
                                          (paired_df['type3'] == t3)])

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(TYPE3_ORDER)))
    ax.set_xticklabels(TYPE3_ORDER, fontsize=10)
    ax.set_yticks(range(len(TYPE1_ORDER)))
    ax.set_yticklabels(TYPE1_ORDER, fontsize=10)

    for i in range(len(TYPE1_ORDER)):
        for j in range(len(TYPE3_ORDER)):
            val = int(matrix[i, j])
            if val > 0:
                color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='Number of Pairs', shrink=0.8)
    ax.set_title('Dataset Composition: Ring Family × Reaction Type',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Reaction Type', fontsize=11)
    ax.set_ylabel('Ring Family', fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig08_composition_matrix.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig08: Composition Matrix ✓')


# ════════════════════════════════════════════════════════════════
#  图 9-11: 吲哚固定底物 — 不同 type3 对比
# ════════════════════════════════════════════════════════════════
def fig09_indole_aromaticity_by_type(paired_df):
    """吲哚: 芳香性变化 (ΔHOMA, ΔMBCO, ΔNICS) 按 type3 分组"""
    indole = paired_df[paired_df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    metrics = [
        ('dHOMA', 'ΔHOMA', '(A) ΔHOMA'),
        ('dMBCO', 'ΔMBCO', '(B) ΔMBCO'),
        ('dNICS', 'ΔNICS (ppm)', '(C) ΔNICS'),
    ]

    for ax, (col, yl, title) in zip(axes.flat, metrics):
        data_dict = {}
        for t3 in TYPE3_ORDER:
            vals = indole[indole['type3'] == t3][col]
            if len(vals) > 0:
                data_dict[t3] = vals
        grouped_boxplot(ax, data_dict, ylabel=yl, title=title)
        ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    fig.suptitle('Indole Dearomatization: Aromaticity Changes by Reaction Type',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig09_indole_aromaticity_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig09: Indole Aromaticity by Type ✓')


def fig10_indole_stereo_by_type(paired_df):
    """吲哚: 立体程度变化 (ΔRPD, ΔPBF, ΔFsp3) 按 type3 分组"""
    indole = paired_df[paired_df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    metrics = [
        ('dRPD',  'ΔRPD (Å)', '(A) ΔRPD'),
        ('dPBF',  'ΔPBF (Å)', '(B) ΔPBF'),
        ('dFsp3', 'ΔFsp³',   '(C) ΔFsp³'),
    ]

    for ax, (col, yl, title) in zip(axes.flat, metrics):
        data_dict = {}
        for t3 in TYPE3_ORDER:
            vals = indole[indole['type3'] == t3][col]
            if len(vals) > 0:
                data_dict[t3] = vals
        grouped_boxplot(ax, data_dict, ylabel=yl, title=title)
        ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)

    fig.suptitle('Indole Dearomatization: Structural Changes by Reaction Type',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig10_indole_stereo_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig10: Indole Stereo by Type ✓')


def fig11_indole_aromaticity_vs_stereo(paired_df):
    """吲哚: 芳香性变化 vs 立体程度变化, 按 type3 着色"""
    indole = paired_df[paired_df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        ('dHOMA', 'dRPD',  'ΔHOMA', 'ΔRPD (Å)',  '(A) ΔHOMA vs ΔRPD'),
        ('dHOMA', 'dPBF',  'ΔHOMA', 'ΔPBF (Å)',  '(B) ΔHOMA vs ΔPBF'),
        ('dHOMA', 'dFsp3', 'ΔHOMA', 'ΔFsp³',    '(C) ΔHOMA vs ΔFsp³'),
    ]

    for ax, (xcol, ycol, xl, yl, title) in zip(axes.flat, panels):
        for t3 in TYPE3_ORDER:
            sub = indole[indole['type3'] == t3]
            ax.scatter(sub[xcol], sub[ycol], s=24, alpha=0.6,
                       color=COLORS.get(t3, '#888'), edgecolors='white',
                       linewidth=0.3, label=f'{t3} (n={len(sub)})', zorder=3)

        # 总体拟合
        valid = indole.dropna(subset=[xcol, ycol])
        if len(valid) >= 5:
            slope, intercept, r, pval, se = stats.linregress(valid[xcol], valid[ycol])
            x_fit = np.linspace(valid[xcol].min(), valid[xcol].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, '--', color='#333',
                    linewidth=1.5, alpha=0.7, zorder=4)
            ax.text(0.05, 0.97, f'r = {r:.3f}\np = {pval:.2e}',
                    transform=ax.transAxes, fontsize=8.5, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9))

        ax.axhline(y=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.axvline(x=0, color='#888', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.legend(fontsize=7.5, frameon=True, framealpha=0.9,
                  edgecolor='#ccc', loc='lower right')
        style_ax(ax)

    fig.suptitle('Indole: Aromaticity Loss vs Structural Reorganization',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig11_indole_aroma_vs_stereo.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig11: Indole Aroma vs Stereo ✓')


# ════════════════════════════════════════════════════════════════
#  图 12: 吲哚反应物 vs 产物配对比较 (按 type3 分列)
# ════════════════════════════════════════════════════════════════
def fig12_indole_paired_by_type(paired_df):
    """吲哚: HOMA 和 RPD 的反应物→产物配对变化, 按 type3 分列"""
    indole = paired_df[paired_df['type1'] == '吲哚'].copy()
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    metrics = [
        ('r_HOMA', 'p_HOMA', 'HOMA', 'HOMA'),
        ('r_RPD',  'p_RPD',  'Ring RPD (Å)', 'Ring RPD'),
        ('r_PBF',  'p_PBF',  'Mol PBF (Å)', 'Mol PBF'),
    ]

    for row, (rc, pc, yl, name) in enumerate(metrics):
        for col, t3 in enumerate(TYPE3_ORDER):
            ax = axes[row][col]
            sub = indole[indole['type3'] == t3]
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            paired_boxplot(ax, sub[rc], sub[pc], ylabel=yl,
                          title=f'{"ABC"[row]}{col+1}) {name} — {t3}')
            if col > 0:
                ax.set_ylabel('')

    fig.suptitle('Indole: Reactant → Product Paired Changes by Reaction Type',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'Fig12_indole_paired_by_type.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig12: Indole Paired by Type ✓')


# ════════════════════════════════════════════════════════════════
#  图 13: Cremer-Pople Q 和 Asphericity 补充分析
# ════════════════════════════════════════════════════════════════
def fig13_supplementary_3d(paired_df):
    """SI 补充: Cremer-Pople Q, Asphericity 的反应物 vs 产物"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Q: reactant vs product
    paired_boxplot(axes[0], paired_df['r_Q'], paired_df['p_Q'],
                   ylabel='Cremer-Pople Q (Å)',
                   title='(A) Ring Puckering Amplitude')

    # Asphericity
    paired_boxplot(axes[1], paired_df['r_Asp'], paired_df['p_Asp'],
                   ylabel='Asphericity',
                   title='(B) Molecular Asphericity')

    # ΔHOMA vs ΔQ
    scatter_with_fit(axes[2], paired_df['dHOMA'], paired_df['dQ'],
                     xlabel='ΔHOMA', ylabel='ΔCremer-Pople Q (Å)',
                     title='(C) ΔHOMA vs ΔQ')

    fig.suptitle('Supplementary 3D Descriptors',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Fig13_supplementary_3d.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Fig13: Supplementary 3D ✓')


# ════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════
paired_df = None  # 全局, 供 scatter_with_fit 的 color_by 使用

def main():
    global paired_df
    print('=' * 60)
    print('CADA 芳香性 & 立体程度分析 — 图表生成')
    print('=' * 60)

    paired_df = load_and_pair()

    # 保存配对数据
    paired_df.to_excel(os.path.join(OUTPUT_DIR, 'paired_data.xlsx'), index=False)
    print(f"  配对数据已保存: {OUTPUT_DIR}/paired_data.xlsx")

    # 打印 Δ 统计
    print(f"\n  mean(ΔHOMA) = {paired_df['dHOMA'].mean():.4f}")
    print(f"  mean(ΔRPD)  = {paired_df['dRPD'].mean():.4f}")
    print(f"  mean(ΔPBF)  = {paired_df['dPBF'].mean():.4f}")
    print(f"  mean(ΔFsp3) = {paired_df['dFsp3'].mean():.4f}")

    print('\n生成图表:')
    # 全局分析
    fig01_reactant_vs_product(paired_df)
    fig02_delta_by_reaction_type(paired_df)
    fig03_delta_by_ring_type(paired_df)
    fig04_aromaticity_correlations(paired_df)
    fig05_aromaticity_vs_structure(paired_df)
    fig06_delta_homa_vs_rpd_by_type(paired_df)
    fig07_shape_space(paired_df)
    fig08_composition_matrix(paired_df)

    # 吲哚专题
    fig09_indole_aromaticity_by_type(paired_df)
    fig10_indole_stereo_by_type(paired_df)
    fig11_indole_aromaticity_vs_stereo(paired_df)
    fig12_indole_paired_by_type(paired_df)

    # SI 补充
    fig13_supplementary_3d(paired_df)

    print(f'\n完成! 共生成 13 张图表 → {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
