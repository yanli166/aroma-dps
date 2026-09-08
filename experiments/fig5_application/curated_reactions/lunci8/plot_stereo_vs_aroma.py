"""
lunci8 反应前后芳香性预测值 vs 分子立体程度可视化

数据源: lunci8-predicted-draw.csv
- TYPE1: 反应类型 (1/2/3/4), 用相同颜色表示
- TYPE2: 1=反应物, 2=产物, 用深浅区分

立体程度指标 (RDKit 计算):
  1. Fsp3 (sp3 碳占比): 越高越立体
  2. n_chiral (手性中心数): 越多越立体
  3. n_stereo_bonds (E/Z 立体键数)
  4. n_rotatable (可旋转键数)
  5. rgyr (3D 回转半径): 越大越伸展
  6. PMI_ratio (n1/n3 主惯性矩比): 1=平面, 0=线性, ~0.5=球形

可视化版本:
  V1: 散点+箭头图 (立体指标 vs 预测值, 反应物→产物)
  V2: 反应前后配对柱状图
  V3: Δ 变化散点图 (Δ立体 vs Δ预测)
  V4: 斜率/折线图 (反应物→产物轨迹)
  V5: 综合仪表盘 (多指标 + 三任务)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import seaborn as sns

# 中文字体配置 (Noto Sans CJK JP - ttc 文件实际注册名)
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
_CJK_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK_PATH):
    fm.fontManager.addfont(_CJK_PATH)
    _CJK_FONT = fm.FontProperties(fname=_CJK_PATH).get_name()
    plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, FindMolChiralCenters
from rdkit.Chem.rdMolDescriptors import CalcPMI1, CalcPMI2, CalcPMI3

# ============== 配置 ==============
CSV_PATH = '/home/ubuntu/aroma-dps-code/lunci8/lunci8-predicted-draw.csv'
OUT_DIR = '/home/ubuntu/aroma-dps-code/lunci8/plots'
os.makedirs(OUT_DIR, exist_ok=True)

# 反应类型颜色 (4 种反应类型, 同色)
TYPE1_COLORS = {
    1: '#1f77b4',  # 蓝
    2: '#ff7f0e',  # 橙
    3: '#2ca02c',  # 绿
    4: '#d62728',  # 红
}
TYPE1_NAMES = {
    1: '反应类型 1 (吲哚 Diels-Alder)',
    2: '反应类型 2 (吡咯 Diels-Alder)',
    3: '反应类型 3 (苯酚/吡啶 Diels-Alder)',
    4: '反应类型 4 (萘酚 Diels-Alder)',
}
# TYPE2 深浅: 1=反应物 (浅), 2=产物 (深)
TYPE2_ALPHA = {1: 0.45, 2: 1.0}
TYPE2_NAMES = {1: '反应物', 2: '产物'}

TASKS = [
    ('HOMA_pred', 'HOMA', '预测 HOMA 值'),
    ('NICS_1zz_pred', 'NICS(1)zz', '预测 NICS(1)zz 值 (ppm)'),
    ('MBCO_pred', 'MBCO', '预测 MBCO 值'),
]

STEREO_METRICS = [
    ('fsp3', 'Fsp3 (sp3 碳占比)', 'Fsp3'),
    ('n_chiral', '手性中心数', 'n_chiral'),
    ('n_stereo_bonds', '立体键数 (E/Z)', 'n_stereo_bonds'),
    ('n_rotatable', '可旋转键数', 'n_rotatable'),
    ('rgyr', '回转半径 (Å)', 'rgyr'),
    ('pmi_ratio', 'PMI 比 (n1/n3)', 'pmi_ratio'),
]


# ============== 立体程度指标计算 ==============
def compute_stereo_metrics(smiles):
    """计算分子的多个立体程度指标

    Returns:
        dict with keys: fsp3, n_chiral, n_stereo_bonds, n_rotatable, rgyr, pmi_ratio
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k, *_ in STEREO_METRICS}

    # 1. Fsp3 = sp3 碳数 / 总碳数
    carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    n_c = len(carbons)
    n_sp3 = sum(1 for a in carbons if a.GetHybridization() == Chem.HybridizationType.SP3)
    fsp3 = n_sp3 / n_c if n_c > 0 else 0.0

    # 2. 手性中心数
    try:
        chiral = FindMolChiralCenters(mol, includeUnassigned=True)
        n_chiral = len(chiral)
    except Exception:
        n_chiral = 0

    # 3. 立体键数 (E/Z) - 从 SMILES 中数 / 和 \ (排除 \n 等)
    # 用 RDKit 的 GetStereoBonds 更准确
    n_stereo_bonds = 0
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo != Chem.BondStereo.STEREONONE:
            n_stereo_bonds += 1

    # 4. 可旋转键数
    n_rotatable = Descriptors.NumRotatableBonds(mol)

    # 5. & 6. 3D 描述符 (回转半径 + PMI 比)
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        # 回转半径
        conf = mol_h.GetConformer()
        from rdkit.Chem import rdFreeSASA
        # 用原子坐标直接算回转半径
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())])
        center = pos.mean(axis=0)
        rgyr = float(np.sqrt(((pos - center) ** 2).sum(axis=1).mean()))
        # PMI
        pmi1 = CalcPMI1(mol_h)
        pmi3 = CalcPMI3(mol_h)
        if pmi3 > 0:
            pmi_ratio = pmi1 / pmi3
        else:
            pmi_ratio = np.nan
    except Exception:
        rgyr = np.nan
        pmi_ratio = np.nan

    return {
        'fsp3': fsp3,
        'n_chiral': n_chiral,
        'n_stereo_bonds': n_stereo_bonds,
        'n_rotatable': n_rotatable,
        'rgyr': rgyr,
        'pmi_ratio': pmi_ratio,
    }


def load_data(csv_path):
    """加载 CSV 并附加立体指标 + 配对信息"""
    df = pd.read_csv(csv_path)
    # 计算立体指标
    metrics_list = [compute_stereo_metrics(s) for s in df['SMILES']]
    metrics_df = pd.DataFrame(metrics_list)
    for col in metrics_df.columns:
        df[col] = metrics_df[col]

    # 配对 ID: 同 TYPE1 内, 反应物按顺序配产物
    df = df.sort_values(['TYPE1', 'TYPE2', 'no']).reset_index(drop=True)
    pair_id = []
    for t1 in sorted(df['TYPE1'].unique()):
        sub = df[df['TYPE1'] == t1]
        reactants = sub[sub['TYPE2'] == 1]['no'].tolist()
        products = sub[sub['TYPE2'] == 2]['no'].tolist()
        rmap = {r: i for i, r in enumerate(reactants)}
        pmap = {p: i for i, p in enumerate(products)}
        for _, row in sub.iterrows():
            if row['TYPE2'] == 1:
                pair_id.append(f"T{t1}-P{rmap[row['no']]+1}")
            else:
                pair_id.append(f"T{t1}-P{pmap[row['no']]+1}")
    df['pair_id'] = pair_id
    return df


# ============== V1: 散点 + 箭头图 ==============
def plot_v1_scatter_arrows(df, out_dir):
    """V1: 立体指标 (X) vs 预测值 (Y), 反应物→产物箭头连接

    每个立体指标一行, 三个任务一列, 共 6×3 = 18 子图。
    每对反应物→产物用箭头连接, 颜色按 TYPE1。
    """
    n_metrics = len(STEREO_METRICS)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_metrics, n_tasks, figsize=(15, 4 * n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)

    for i, (mkey, mname, _) in enumerate(STEREO_METRICS):
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[i, j]
            # 绘制每对反应物→产物箭头
            for pid in df['pair_id'].unique():
                sub = df[df['pair_id'] == pid].sort_values('TYPE2')
                if len(sub) != 2:
                    continue
                r = sub[sub['TYPE2'] == 1].iloc[0]
                p = sub[sub['TYPE2'] == 2].iloc[0]
                t1 = int(r['TYPE1'])
                color = TYPE1_COLORS[t1]
                x0, y0 = r[mkey], r[tkey]
                x1, y1 = p[mkey], p[tkey]
                # 反应物 (浅)
                ax.scatter(x0, y0, c=color, alpha=TYPE2_ALPHA[1], s=90,
                           edgecolors='black', linewidths=0.6, zorder=3)
                # 产物 (深)
                ax.scatter(x1, y1, c=color, alpha=TYPE2_ALPHA[2], s=110,
                           edgecolors='black', linewidths=0.8, zorder=4,
                           marker='s')
                # 箭头
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='->', color=color,
                                        alpha=0.6, mutation_scale=12,
                                        linewidth=1.2, zorder=2)
                ax.add_patch(arrow)

            ax.set_xlabel(mname, fontsize=10)
            ax.set_ylabel(tdesc, fontsize=10)
            if i == 0:
                ax.set_title(tname, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=9)

    # 图例
    legend_elements = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=TYPE1_COLORS[t1], markersize=10,
                                      label=TYPE1_NAMES[t1]))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                  alpha=0.45, markersize=10, label='反应物 (浅)'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                  alpha=1.0, markersize=10, label='产物 (深)'))
    legend_elements.append(Line2D([0], [0], color='gray', alpha=0.6,
                                  linewidth=1.5, label='反应物 → 产物'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('V1: 分子立体程度 vs 预测芳香性指标 (反应物→产物)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = os.path.join(out_dir, 'V1_scatter_arrows.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V1 → {out_path}")


# ============== V2: 配对柱状图 ==============
def plot_v2_paired_bars(df, out_dir):
    """V2: 每对反应物/产物并列柱状图

    3 个子图 (HOMA, NICS, MBCO), X 轴为分子对 (按 TYPE1 分组)。
    反应物浅色, 产物深色, 同 TYPE1 同色调。
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    pair_order = df.drop_duplicates('pair_id').sort_values(['TYPE1', 'pair_id'])['pair_id'].tolist()
    x = np.arange(len(pair_order))
    width = 0.38

    for ax, (tkey, tname, tdesc) in zip(axes, TASKS):
        r_vals, p_vals, colors = [], [], []
        for pid in pair_order:
            sub = df[df['pair_id'] == pid]
            r = sub[sub['TYPE2'] == 1].iloc[0]
            p = sub[sub['TYPE2'] == 2].iloc[0]
            r_vals.append(r[tkey])
            p_vals.append(p[tkey])
            colors.append(TYPE1_COLORS[int(r['TYPE1'])])

        # 反应物 (浅色)
        bars_r = ax.bar(x - width/2, r_vals, width, color=colors, alpha=TYPE2_ALPHA[1],
                        edgecolor='black', linewidth=0.6, label='反应物')
        # 产物 (深色)
        bars_p = ax.bar(x + width/2, p_vals, width, color=colors, alpha=TYPE2_ALPHA[2],
                        edgecolor='black', linewidth=0.6, label='产物',
                        hatch='///')

        ax.set_xticks(x)
        ax.set_xticklabels(pair_order, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(tdesc, fontsize=11)
        ax.set_title(f'{tname}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5)

        # 在 TYPE1 边界画分隔线
        prev_t1 = None
        for i, pid in enumerate(pair_order):
            t1 = int(pid.split('-')[0][1:])
            if prev_t1 is not None and t1 != prev_t1:
                ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            prev_t1 = t1

    # 图例
    legend_elements = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        legend_elements.append(Line2D([0], [0], color=TYPE1_COLORS[t1], linewidth=10,
                                      label=TYPE1_NAMES[t1]))
    legend_elements.append(Line2D([0], [0], color='gray', alpha=0.45, linewidth=10,
                                  label='反应物 (浅, 实心)'))
    legend_elements.append(Line2D([0], [0], color='gray', alpha=1.0, linewidth=10,
                                  label='产物 (深, 斜线)'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('V2: 反应前后芳香性指标配对柱状图 (按反应类型分组)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_path = os.path.join(out_dir, 'V2_paired_bars.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V2 → {out_path}")


# ============== V3: Δ 变化散点图 ==============
def plot_v3_delta_scatter(df, out_dir):
    """V3: Δ立体指标 (X) vs Δ预测值 (Y)

    每个立体指标一行, 三个任务一列。每个点是一个反应对 (产物-反应物)。
    颜色按 TYPE1, 点大小按反应物预测值。
    """
    # 计算每个 pair 的 Δ
    delta_rows = []
    for pid in df['pair_id'].unique():
        sub = df[df['pair_id'] == pid].sort_values('TYPE2')
        if len(sub) != 2:
            continue
        r, p = sub.iloc[0], sub.iloc[1]
        row = {'pair_id': pid, 'TYPE1': int(r['TYPE1'])}
        for mkey, _, _ in STEREO_METRICS:
            row[f'd_{mkey}'] = p[mkey] - r[mkey]
        for tkey, _, _ in TASKS:
            row[f'd_{tkey}'] = p[tkey] - r[tkey]
            row[f'r_{tkey}'] = r[tkey]
        delta_rows.append(row)
    ddf = pd.DataFrame(delta_rows)

    n_metrics = len(STEREO_METRICS)
    fig, axes = plt.subplots(n_metrics, 3, figsize=(16, 4 * n_metrics))

    for i, (mkey, mname, _) in enumerate(STEREO_METRICS):
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[i, j]
            for t1 in sorted(TYPE1_COLORS.keys()):
                sub = ddf[ddf['TYPE1'] == t1]
                ax.scatter(sub[f'd_{mkey}'], sub[f'd_{tkey}'],
                           c=TYPE1_COLORS[t1], s=130, alpha=0.85,
                           edgecolors='black', linewidths=0.8,
                           label=TYPE1_NAMES[t1], zorder=3)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel(f'Δ {mname} (产物-反应物)', fontsize=10)
            ax.set_ylabel(f'Δ {tname} (产物-反应物)', fontsize=10)
            if i == 0:
                ax.set_title(tname, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=9)

            # 象限标注
            ax.text(0.97, 0.97, '立体↑ 指标↑', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='green', alpha=0.6)
            ax.text(0.03, 0.03, '立体↓ 指标↓', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=8, color='red', alpha=0.6)

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=TYPE1_COLORS[t1], markersize=12,
                              label=TYPE1_NAMES[t1])
                       for t1 in sorted(TYPE1_COLORS.keys())]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('V3: Δ立体程度 vs Δ预测值 变化散点图 (产物 - 反应物)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = os.path.join(out_dir, 'V3_delta_scatter.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V3 → {out_path}")


# ============== V4: 斜率/折线图 ==============
def plot_v4_slope_lines(df, out_dir):
    """V4: 反应物→产物折线图 (每条线一个反应对)

    上半: 立体指标 (6 个子图), 下半: 预测指标 (3 个子图)。
    X 轴: TYPE2 (1=反应物, 2=产物), Y 轴: 指标值。
    每条线颜色 = TYPE1, 末端标记 pair_id。
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

    # 上半: 6 个立体指标 (2 行 × 3 列)
    for i, (mkey, mname, _) in enumerate(STEREO_METRICS):
        r, c = i // 3, i % 3
        ax = fig.add_subplot(gs[r, c])
        for pid in df['pair_id'].unique():
            sub = df[df['pair_id'] == pid].sort_values('TYPE2')
            if len(sub) != 2:
                continue
            t1 = int(sub.iloc[0]['TYPE1'])
            color = TYPE1_COLORS[t1]
            vals = sub[mkey].values
            ax.plot([1, 2], vals, '-o', color=color, alpha=0.7,
                    linewidth=1.5, markersize=7, markeredgecolor='black',
                    markeredgewidth=0.4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['反应物', '产物'], fontsize=10)
        ax.set_ylabel(mname, fontsize=10)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', axis='y')
        ax.set_xlim(0.85, 2.25)

    # 下半: 3 个预测指标 (1 行 × 3 列)
    for j, (tkey, tname, tdesc) in enumerate(TASKS):
        ax = fig.add_subplot(gs[2, j])
        for pid in df['pair_id'].unique():
            sub = df[df['pair_id'] == pid].sort_values('TYPE2')
            if len(sub) != 2:
                continue
            t1 = int(sub.iloc[0]['TYPE1'])
            color = TYPE1_COLORS[t1]
            vals = sub[tkey].values
            ax.plot([1, 2], vals, '-o', color=color, alpha=0.85,
                    linewidth=2.0, markersize=8, markeredgecolor='black',
                    markeredgewidth=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['反应物', '产物'], fontsize=11)
        ax.set_ylabel(tdesc, fontsize=10)
        ax.set_title(f'{tname} (预测)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', axis='y')
        ax.set_xlim(0.85, 2.25)

    legend_elements = [Line2D([0], [0], color=TYPE1_COLORS[t1], linewidth=3,
                              marker='o', markersize=8,
                              label=TYPE1_NAMES[t1])
                       for t1 in sorted(TYPE1_COLORS.keys())]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle('V4: 反应物→产物 指标变化斜率图 (上: 立体程度, 下: 预测芳香性)',
                 fontsize=14, fontweight='bold', y=0.995)
    out_path = os.path.join(out_dir, 'V4_slope_lines.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V4 → {out_path}")


# ============== V5: 综合仪表盘 ==============
def plot_v5_dashboard(df, out_dir):
    """V5: 综合仪表盘

    布局:
      左上: Fsp3 vs HOMA (散点+箭头, 主图)
      右上: 平均变化柱状图 (3 任务 × 4 TYPE1)
      左下: 雷达图 (4 TYPE1 的 6 立体指标均值, 反应物 vs 产物)
      右下: ΔFsp3 vs ΔHOMA (变化散点)
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    # (1) 左上: Fsp3 vs HOMA 散点+箭头
    ax1 = fig.add_subplot(gs[0, 0])
    for pid in df['pair_id'].unique():
        sub = df[df['pair_id'] == pid].sort_values('TYPE2')
        if len(sub) != 2:
            continue
        r, p = sub.iloc[0], sub.iloc[1]
        t1 = int(r['TYPE1'])
        color = TYPE1_COLORS[t1]
        ax1.scatter(r['fsp3'], r['HOMA_pred'], c=color, alpha=0.45, s=130,
                    edgecolors='black', linewidths=0.6, zorder=3)
        ax1.scatter(p['fsp3'], p['HOMA_pred'], c=color, alpha=1.0, s=150,
                    edgecolors='black', linewidths=0.8, marker='s', zorder=4)
        arrow = FancyArrowPatch((r['fsp3'], r['HOMA_pred']),
                                (p['fsp3'], p['HOMA_pred']),
                                arrowstyle='->', color=color, alpha=0.6,
                                mutation_scale=14, linewidth=1.4, zorder=2)
        ax1.add_patch(arrow)
    ax1.set_xlabel('Fsp3 (sp3 碳占比)', fontsize=12)
    ax1.set_ylabel('预测 HOMA', fontsize=12)
    ax1.set_title('Fsp3 vs HOMA (反应物圆→产物方)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')

    # (2) 右上: 平均变化柱状图 (3 任务 × 4 TYPE1)
    ax2 = fig.add_subplot(gs[0, 1])
    delta_summary = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        sub = df[df['TYPE1'] == t1]
        for pid in sub['pair_id'].unique():
            psub = sub[sub['pair_id'] == pid].sort_values('TYPE2')
            if len(psub) != 2:
                continue
            r, p = psub.iloc[0], psub.iloc[1]
            for tkey, tname, _ in TASKS:
                delta_summary.append({
                    'TYPE1': t1, 'task': tname,
                    'delta': p[tkey] - r[tkey]
                })
    ddf = pd.DataFrame(delta_summary)
    pivot = ddf.groupby(['TYPE1', 'task'])['delta'].mean().unstack()
    # 保持任务顺序
    task_order = [t[1] for t in TASKS]
    pivot = pivot[task_order]
    pivot.plot(kind='bar', ax=ax2, edgecolor='black', linewidth=0.6,
               width=0.8)
    ax2.set_xticklabels([TYPE1_NAMES[i].split(' (')[0] for i in pivot.index],
                        rotation=0, fontsize=10)
    ax2.set_ylabel('平均变化 (产物 - 反应物)', fontsize=12)
    ax2.set_title('各反应类型平均指标变化', fontsize=13, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(title='指标', fontsize=10, title_fontsize=11)

    # (3) 左下: 雷达图 (4 TYPE1 的立体指标, 反应物 vs 产物)
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    radar_metrics = ['fsp3', 'n_chiral', 'n_stereo_bonds', 'n_rotatable', 'rgyr', 'pmi_ratio']
    # 归一化到 [0,1]
    norm_df = df.copy()
    for m in radar_metrics:
        vmax = df[m].max()
        if vmax > 0:
            norm_df[m] = df[m] / vmax
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for t1 in sorted(TYPE1_COLORS.keys()):
        for t2 in [1, 2]:
            sub = norm_df[(norm_df['TYPE1'] == t1) & (norm_df['TYPE2'] == t2)]
            if len(sub) == 0:
                continue
            vals = [sub[m].mean() for m in radar_metrics]
            vals += vals[:1]
            color = TYPE1_COLORS[t1]
            ax3.plot(angles, vals, '-', color=color, alpha=TYPE2_ALPHA[t2],
                     linewidth=2 if t2 == 2 else 1.2,
                     label=f'{TYPE1_NAMES[t1].split(" (")[0]} - {TYPE2_NAMES[t2]}')
            ax3.fill(angles, vals, color=color, alpha=0.12 if t2 == 1 else 0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(['Fsp3', '手性中心', '立体键', '可旋转键', '回转半径', 'PMI比'],
                        fontsize=10)
    ax3.set_title('立体程度雷达图 (归一化, 浅=反应物, 深=产物)',
                  fontsize=13, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=9)

    # (4) 右下: ΔFsp3 vs ΔHOMA 变化散点
    ax4 = fig.add_subplot(gs[1, 1])
    for t1 in sorted(TYPE1_COLORS.keys()):
        sub_pairs = []
        for pid in df[df['TYPE1'] == t1]['pair_id'].unique():
            psub = df[df['pair_id'] == pid].sort_values('TYPE2')
            if len(psub) != 2:
                continue
            r, p = psub.iloc[0], psub.iloc[1]
            sub_pairs.append({
                'd_fsp3': p['fsp3'] - r['fsp3'],
                'd_HOMA': p['HOMA_pred'] - r['HOMA_pred'],
                'd_NICS': p['NICS_1zz_pred'] - r['NICS_1zz_pred'],
                'd_MBCO': p['MBCO_pred'] - r['MBCO_pred'],
            })
        spp = pd.DataFrame(sub_pairs)
        ax4.scatter(spp['d_fsp3'], spp['d_HOMA'], c=TYPE1_COLORS[t1], s=140,
                    alpha=0.85, edgecolors='black', linewidths=0.8,
                    label=TYPE1_NAMES[t1], zorder=3)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Δ Fsp3 (产物 - 反应物)', fontsize=12)
    ax4.set_ylabel('Δ HOMA (产物 - 反应物)', fontsize=12)
    ax4.set_title('ΔFsp3 vs ΔHOMA 变化关系', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10)
    # 象限标注
    ax4.text(0.97, 0.97, '立体↑ HOMA↑', transform=ax4.transAxes,
             ha='right', va='top', fontsize=9, color='green', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax4.text(0.03, 0.03, '立体↓ HOMA↓', transform=ax4.transAxes,
             ha='left', va='bottom', fontsize=9, color='red', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.suptitle('V5: 综合仪表盘 — 立体程度 vs 芳香性预测',
                 fontsize=15, fontweight='bold', y=0.995)
    out_path = os.path.join(out_dir, 'V5_dashboard.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V5 → {out_path}")


# ============== V6: 热力图 (反应前后变化矩阵) ==============
def plot_v6_heatmap(df, out_dir):
    """V6: 热力图 — 每个反应对的 9 个指标变化 (6 立体 + 3 预测)

    行: 17 个反应对 (按 TYPE1 分组)
    列: 9 个指标 (Δ 值, 归一化)
    """
    delta_rows = []
    row_labels = []
    row_types = []
    for pid in df['pair_id'].unique():
        sub = df[df['pair_id'] == pid].sort_values('TYPE2')
        if len(sub) != 2:
            continue
        r, p = sub.iloc[0], sub.iloc[1]
        t1 = int(r['TYPE1'])
        row = {}
        for mkey, mname, _ in STEREO_METRICS:
            row[f'Δ{mname}'] = p[mkey] - r[mkey]
        for tkey, tname, _ in TASKS:
            row[f'Δ{tname}'] = p[tkey] - r[tkey]
        delta_rows.append(row)
        row_labels.append(pid)
        row_types.append(t1)
    ddf = pd.DataFrame(delta_rows, index=row_labels)

    # 按 TYPE1 排序
    ddf['_t1'] = row_types
    ddf = ddf.sort_values(['_t1']).drop(columns=['_t1'])

    fig, ax = plt.subplots(figsize=(13, 10))
    # 用 diverging colormap (变化有正有负)
    sns.heatmap(ddf, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.6, linecolor='gray',
                cbar_kws={'label': 'Δ 值 (产物 - 反应物)', 'shrink': 0.8},
                annot_kws={'fontsize': 8})
    ax.set_title('V6: 反应前后指标变化热力图\n(红=增加, 蓝=减少; 浅=反应物→产物变化)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('反应对 (按反应类型分组)', fontsize=11)
    ax.set_xlabel('指标 (Δ = 产物 - 反应物)', fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    # 在 TYPE1 边界画粗线
    prev_t1 = None
    for i, t1 in enumerate(sorted(row_types)):
        if prev_t1 is not None and t1 != prev_t1:
            ax.axhline(i, color='black', linewidth=2)
        prev_t1 = t1

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'V6_heatmap.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V6 → {out_path}")


# ============== 主入口 ==============
def main():
    print(f"加载数据: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"  {len(df)} 行, {df['pair_id'].nunique()} 反应对")
    print(f"  立体指标计算完成: {[m[0] for m in STEREO_METRICS]}")
    print(f"\n开始绘图 (输出目录: {OUT_DIR})...")

    plot_v1_scatter_arrows(df, OUT_DIR)
    plot_v2_paired_bars(df, OUT_DIR)
    plot_v3_delta_scatter(df, OUT_DIR)
    plot_v4_slope_lines(df, OUT_DIR)
    plot_v5_dashboard(df, OUT_DIR)
    plot_v6_heatmap(df, OUT_DIR)

    # 保存带立体指标的数据
    out_csv = os.path.join(OUT_DIR, 'lunci8_with_stereo_metrics.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n带立体指标的数据已保存: {out_csv}")
    print(f"\n所有图已保存到: {OUT_DIR}/")


if __name__ == '__main__':
    main()
