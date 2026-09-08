"""
lunci8 反应前后芳香性 vs 分子立体程度 (环扭曲 / 折叠) 可视化

立体程度指标 (聚焦芳香平面扭曲/折叠, 与芳香性物理意义匹配):

母核骨架类 (推荐, 化学意义最强):
  1. ring_Q         - Cremer-Pople 环褶皱幅度 Q (Å), 0=完美平面
  2. core_rmsd      - 母核重原子到最佳拟合平面的 RMSD (Å)
  3. butterfly_angle- 稠环折叠角 (°), 180°=平面, 减小=折叠 (仅稠环体系)
  4. ring_dihed_sum - 目标环相邻原子二面角绝对值和 (°), 0=平面
  5. ring_max_dev   - 目标环原子到拟合平面最大偏离 (Å)

分子整体类:
  6. mol_rg         - 分子回转半径 (Å)
  7. mol_ca_ratio   - 椭球扁率 c/a (0=扁平, 1=球形)
  8. mol_sasa       - 溶剂可及表面积 (Å²)

输出图:
  V1: 全指标散点矩阵 (8 立体指标 × 3 任务)
  V2: 关键指标大图 (单图单指标, 高质量)
  V3: 反应物→产物箭头图 (核心指标)
  V4: Δ变化散点图
  V5: 综合仪表盘
  V6: 热力图
"""
import os
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import seaborn as sns

# 中文字体配置
import matplotlib.font_manager as fm
_CJK_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK_PATH):
    fm.fontManager.addfont(_CJK_PATH)
    _CJK_FONT = fm.FontProperties(fname=_CJK_PATH).get_name()
    plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcPMI1, CalcPMI2, CalcPMI3
from rdkit.Chem import rdFreeSASA

# ============== 配置 ==============
CSV_PATH = '/home/ubuntu/aroma-dps-code/lunci8/lunci8-predicted-draw.csv'
OUT_DIR = '/home/ubuntu/aroma-dps-code/lunci8/plots_v3'
os.makedirs(OUT_DIR, exist_ok=True)

# 反应类型颜色 (TYPE1, 4 种, 用颜色区分)
TYPE1_COLORS = {
    1: '#1f77b4',  # 蓝 - 吲哚 DA
    2: '#ff7f0e',  # 橙 - 吡咯 DA
    3: '#2ca02c',  # 绿 - 苯酚/吡啶 DA
    4: '#d62728',  # 红 - 萘酚 DA
}
TYPE1_NAMES = {
    1: 'T1: 吲哚 DA',
    2: 'T2: 吡咯 DA',
    3: 'T3: 苯酚 DA',
    4: 'T4: 萘酚 DA',
}
TYPE2_ALPHA = {1: 0.45, 2: 1.0}   # 反应物浅, 产物深
TYPE2_MARKER = {1: 'o', 2: 's'}   # 反应物圆, 产物方
TYPE2_NAMES = {1: '反应物', 2: '产物'}

TASKS = [
    ('HOMA_pred', 'HOMA', '预测 HOMA'),
    ('NICS_1zz_pred', 'NICS(1)zz', '预测 NICS(1)zz (ppm)'),
    ('MBCO_pred', 'MBCO', '预测 MBCO'),
]

# 立体程度指标 (key, 显示名, 单位, 类别)
STEREO_METRICS = [
    ('ring_Q', 'Cremer-Pople Q', 'Å', '母核'),
    ('core_rmsd', '母核平面 RMSD', 'Å', '母核'),
    ('butterfly_angle', '稠环折叠角', '°', '母核'),
    ('ring_dihed_sum', '环二面角绝对值和', '°', '母核'),
    ('ring_max_dev', '环最大平面偏离', 'Å', '母核'),
    ('mol_rg', '分子回转半径', 'Å', '整体'),
    ('mol_ca_ratio', '椭球扁率 c/a', '', '整体'),
    ('mol_sasa', '溶剂可及表面积', 'Å²', '整体'),
]


# ============== 3D 构象生成 ==============
def get_3d_mol(smiles, seed=42):
    """生成 3D 优化构象 (MMFF94)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, randomSeed=seed, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=1000)
    except Exception:
        try:
            AllChem.Compute2DCoords(mol_h)
        except Exception:
            return None
    return mol_h


def get_coords(mol_h, atom_indices):
    """获取指定原子的 3D 坐标 (NumPy 数组)"""
    conf = mol_h.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in atom_indices])


def best_fit_plane(coords):
    """最佳拟合平面 (SVD), 返回 (平面法向量, 质心)"""
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # SVD: 最小奇异向量 = 平面法向量
    u, s, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def plane_distances(coords, normal, centroid):
    """各原子到平面的有符号距离"""
    return (coords - centroid) @ normal


# ============== 立体指标计算 ==============
def compute_ring_Q(coords):
    """Cremer-Pople 环褶皱幅度 Q = sqrt(sum(z_i^2))

    Q=0 完美平面, 数值越大环越褶皱。
    使用最佳拟合平面 (SVD) 近似。
    """
    if len(coords) < 3:
        return np.nan
    normal, centroid = best_fit_plane(coords)
    z = plane_distances(coords, normal, centroid)
    return float(np.sqrt(np.sum(z ** 2)))


def compute_rmsd_to_plane(coords):
    """原子到最佳拟合平面的 RMSD"""
    if len(coords) < 3:
        return np.nan
    normal, centroid = best_fit_plane(coords)
    z = plane_distances(coords, normal, centroid)
    return float(np.sqrt(np.mean(z ** 2)))


def compute_max_deviation(coords):
    """原子到最佳拟合平面的最大偏离"""
    if len(coords) < 3:
        return np.nan
    normal, centroid = best_fit_plane(coords)
    z = plane_distances(coords, normal, centroid)
    return float(np.max(np.abs(z)))


def compute_butterfly_angle(mol_h, target_ring_0idx):
    """稠环折叠角 (蝴蝶角): 180°=共平面, <180°=折叠

    找与目标环共享 2+ 原子的相邻环, 计算两环平面法向量夹角,
    返回 180° - angle(度)。
    若无稠合伙伴 (单环), 返回 NaN。
    """
    ring_info = mol_h.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    target_set = set(target_ring_0idx)

    # 找稠合伙伴
    partner = None
    for ring in atom_rings:
        if set(ring) == target_set:
            continue
        shared = set(ring) & target_set
        if len(shared) >= 2:
            partner = list(ring)
            break

    if partner is None:
        return np.nan

    # 两环平面法向量
    try:
        c1 = get_coords(mol_h, list(target_ring_0idx))
        c2 = get_coords(mol_h, partner)
        n1, _ = best_fit_plane(c1)
        n2, _ = best_fit_plane(c2)
        # 法向量夹角
        cos_angle = abs(np.dot(n1, n2))
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle = np.degrees(np.arccos(cos_angle))
        # 蝴蝶角: 180° - angle (180=平面, 90=垂直折叠)
        return float(180.0 - angle)
    except Exception:
        return np.nan


def compute_ring_dihedral_sum(mol_h, ring_0idx):
    """环内相邻 4 原子二面角绝对值和 (度)

    对环 [a0,a1,...,an-1], 计算 dihedral(a_{i}, a_{i+1}, a_{i+2}, a_{i+3})
    循环索引。0° = 共面, 越大越扭曲。
    """
    n = len(ring_0idx)
    if n < 4:
        return np.nan
    total = 0.0
    for i in range(n):
        a0 = ring_0idx[i]
        a1 = ring_0idx[(i + 1) % n]
        a2 = ring_0idx[(i + 2) % n]
        a3 = ring_0idx[(i + 3) % n]
        try:
            # RDKit GetTorsionDegree 返回度数 (-180, 180)
            # 用 CalcTorsionAngle 更准确
            from rdkit.Chem import rdMolTransforms
            conf = mol_h.GetConformer()
            angle = rdMolTransforms.GetDihedralDeg(conf, a0, a1, a2, a3)
            total += abs(angle)
        except Exception:
            continue
    return float(total)


def compute_core_atoms(mol_h, target_ring_0idx):
    """获取母核所有原子 (目标环 + 稠合伙伴环原子)"""
    ring_info = mol_h.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    target_set = set(target_ring_0idx)
    core = set(target_ring_0idx)
    for ring in atom_rings:
        shared = set(ring) & target_set
        if len(shared) >= 2:
            core.update(ring)
    return sorted(core)


def compute_mol_rg(mol_h):
    """分子回转半径 (Å)"""
    try:
        conf = mol_h.GetConformer()
        n = mol_h.GetNumAtoms()
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
        center = pos.mean(axis=0)
        return float(np.sqrt(((pos - center) ** 2).sum(axis=1).mean()))
    except Exception:
        return np.nan


def compute_ca_ratio(mol_h):
    """椭球扁率 c/a: 0=扁平, 1=球形

    通过坐标协方差矩阵特征值计算主轴长度比。
    """
    try:
        conf = mol_h.GetConformer()
        n = mol_h.GetNumAtoms()
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
        centered = pos - pos.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # a² >= b² >= c²
        a_sq, b_sq, c_sq = eigvals
        if a_sq > 0:
            return float(np.sqrt(c_sq / a_sq))
        return np.nan
    except Exception:
        return np.nan


def compute_sasa(mol_h):
    """溶剂可及表面积 (Å²)"""
    try:
        radii = rdFreeSASA.classifyAtoms(mol_h)
        return float(rdFreeSASA.CalcSASA(mol_h, radii))
    except Exception:
        return np.nan


def compute_all_metrics(smiles, ring_atoms_1idx):
    """计算全部分子立体程度指标

    Args:
        smiles: SMILES 字符串
        ring_atoms_1idx: 目标环原子 (1-indexed 列表)

    Returns:
        dict of all stereo metrics
    """
    mol_h = get_3d_mol(smiles)
    if mol_h is None:
        return {k: np.nan for k, _, _, _ in STEREO_METRICS}

    # 转 0-indexed
    ring_0idx = [i - 1 for i in ring_atoms_1idx if i > 0]

    # 验证原子索引有效
    n_atoms = mol_h.GetNumAtoms()
    ring_0idx = [i for i in ring_0idx if i < n_atoms]
    if len(ring_0idx) < 3:
        return {k: np.nan for k, _, _, _ in STEREO_METRICS}

    # 目标环坐标
    ring_coords = get_coords(mol_h, ring_0idx)

    # 母核原子
    core_atoms = compute_core_atoms(mol_h, ring_0idx)
    core_coords = get_coords(mol_h, core_atoms)

    metrics = {
        'ring_Q': compute_ring_Q(ring_coords),
        'core_rmsd': compute_rmsd_to_plane(core_coords),
        'butterfly_angle': compute_butterfly_angle(mol_h, ring_0idx),
        'ring_dihed_sum': compute_ring_dihedral_sum(mol_h, ring_0idx),
        'ring_max_dev': compute_max_deviation(ring_coords),
        'mol_rg': compute_mol_rg(mol_h),
        'mol_ca_ratio': compute_ca_ratio(mol_h),
        'mol_sasa': compute_sasa(mol_h),
    }
    return metrics


# ============== 数据加载 ==============
def load_data(csv_path):
    """加载 CSV, 计算立体指标, 建立反应物-产物配对"""
    df = pd.read_csv(csv_path)

    # 解析 ring_atoms 字符串 "[12, 13, 14, 15, 20]" → list
    def parse_ring(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else s
        except Exception:
            return []

    df['ring_atoms_list'] = df['ring_atoms'].apply(parse_ring)

    # 计算立体指标
    print("计算立体程度指标...")
    all_metrics = []
    for _, row in df.iterrows():
        m = compute_all_metrics(row['SMILES'], row['ring_atoms_list'])
        all_metrics.append(m)
    metrics_df = pd.DataFrame(all_metrics)
    for col in metrics_df.columns:
        df[col] = metrics_df[col]

    # 建立 pair_id (同 TYPE1 内, 反应物按序配产物)
    df = df.sort_values(['TYPE1', 'TYPE2', 'no']).reset_index(drop=True)
    pair_ids = []
    for t1 in sorted(df['TYPE1'].unique()):
        sub = df[df['TYPE1'] == t1]
        reactants = sub[sub['TYPE2'] == 1]['no'].tolist()
        products = sub[sub['TYPE2'] == 2]['no'].tolist()
        rmap = {r: i for i, r in enumerate(reactants)}
        pmap = {p: i for i, p in enumerate(products)}
        for _, row in sub.iterrows():
            if row['TYPE2'] == 1:
                pair_ids.append(f"T{t1}-P{rmap[row['no']]+1}")
            else:
                pair_ids.append(f"T{t1}-P{pmap[row['no']]+1}")
    df['pair_id'] = pair_ids

    # 打印指标摘要
    print("\n立体指标摘要:")
    for k, name, unit, cat in STEREO_METRICS:
        vals = df[k].dropna()
        print(f"  {name:25s} ({cat}): n={len(vals)}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}, "
              f"mean={vals.mean():.3f}")
    return df


# ============== V1: 全指标散点矩阵 ==============
def plot_v1_full_matrix(df, out_dir):
    """V1: 8 立体指标 × 3 任务 = 24 子图散点矩阵

    X=立体指标, Y=预测值; 反应物圆/产物方, 同对箭头连接; 颜色=TYPE1。
    """
    n_m = len(STEREO_METRICS)
    n_t = len(TASKS)
    fig, axes = plt.subplots(n_m, n_t, figsize=(15, 3.4 * n_m))

    for i, (mkey, mname, munit, mcat) in enumerate(STEREO_METRICS):
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[i, j]
            # 绘制每对反应物→产物
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

                # 跳过 NaN
                if np.isnan(x0) or np.isnan(x1):
                    continue

                # 反应物 (浅圆)
                ax.scatter(x0, y0, c=color, alpha=TYPE2_ALPHA[1], s=85,
                           edgecolors='black', linewidths=0.5, zorder=3,
                           marker=TYPE2_MARKER[1])
                # 产物 (深方)
                ax.scatter(x1, y1, c=color, alpha=TYPE2_ALPHA[2], s=100,
                           edgecolors='black', linewidths=0.7, zorder=4,
                           marker=TYPE2_MARKER[2])
                # 箭头
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='->', color=color,
                                        alpha=0.55, mutation_scale=11,
                                        linewidth=1.1, zorder=2)
                ax.add_patch(arrow)

            xlabel = f"{mname} ({munit})" if munit else mname
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(tdesc, fontsize=9)
            if i == 0:
                ax.set_title(tname, fontsize=12, fontweight='bold')
            if j == 0:
                # 在左侧标注类别
                ax.text(-0.30, 0.5, f"[{mcat}]\n{mname}", transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        rotation=90)
            ax.grid(alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=8)

    # 图例
    legend = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        legend.append(Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=TYPE1_COLORS[t1], markersize=10,
                             markeredgecolor='black', markeredgewidth=0.5,
                             label=TYPE1_NAMES[t1]))
    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                         alpha=0.45, markersize=10, markeredgecolor='black',
                         markeredgewidth=0.5, label='反应物 (浅圆)'))
    legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                         alpha=1.0, markersize=10, markeredgecolor='black',
                         markeredgewidth=0.5, label='产物 (深方)'))
    legend.append(Line2D([0], [0], color='gray', alpha=0.55, linewidth=1.3,
                         label='反应物 → 产物'))
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle('V1: 分子立体程度 (环扭曲/折叠) vs 预测芳香性 — 全指标散点矩阵',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    out_path = os.path.join(out_dir, 'V1_full_matrix.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V1 → {out_path}")


# ============== V2: 关键指标大图 (每图一指标) ==============
def plot_v2_key_large(df, out_dir):
    """V2: 关键指标 × 3 任务大图 (每张图一个立体指标, 3 子图)

    生成多张图: 每个母核类指标一张大图 (3 列 = HOMA/NICS/MBCO)。
    """
    key_metrics = [
        ('ring_Q', 'Cremer-Pople Q', 'Å', '母核环褶皱幅度'),
        ('core_rmsd', '母核平面 RMSD', 'Å', '母核重原子平面偏离'),
        ('butterfly_angle', '稠环折叠角', '°', '稠环折叠角 (180°=平面)'),
        ('ring_dihed_sum', '环二面角绝对值和', '°', '环内二面角扭曲总量'),
        ('ring_max_dev', '环最大平面偏离', 'Å', '环原子最大平面偏离'),
    ]

    for mkey, mname, munit, mdesc in key_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[j]
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
                if np.isnan(x0) or np.isnan(x1):
                    continue
                ax.scatter(x0, y0, c=color, alpha=TYPE2_ALPHA[1], s=140,
                           edgecolors='black', linewidths=0.6, zorder=3,
                           marker=TYPE2_MARKER[1])
                ax.scatter(x1, y1, c=color, alpha=TYPE2_ALPHA[2], s=160,
                           edgecolors='black', linewidths=0.8, zorder=4,
                           marker=TYPE2_MARKER[2])
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='->', color=color,
                                        alpha=0.6, mutation_scale=14,
                                        linewidth=1.3, zorder=2)
                ax.add_patch(arrow)

            xlabel = f"{mname} ({munit})" if munit else mname
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(tdesc, fontsize=12)
            ax.set_title(f'{tname}', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=10)

        legend = []
        for t1 in sorted(TYPE1_COLORS.keys()):
            legend.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=TYPE1_COLORS[t1], markersize=12,
                                 markeredgecolor='black', markeredgewidth=0.5,
                                 label=TYPE1_NAMES[t1]))
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                             alpha=0.45, markersize=12, markeredgecolor='black',
                             markeredgewidth=0.5, label='反应物 (浅圆)'))
        legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                             alpha=1.0, markersize=12, markeredgecolor='black',
                             markeredgewidth=0.5, label='产物 (深方)'))
        legend.append(Line2D([0], [0], color='gray', alpha=0.6, linewidth=1.5,
                             label='反应物 → 产物'))
        fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=11,
                   frameon=True, bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(f'V2: {mname} — {mdesc}\n(X: {mname}, Y: 预测芳香性)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0.06, 1, 0.98])
        out_path = os.path.join(out_dir, f'V2_{mkey}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  V2_{mkey} → {out_path}")


# ============== V3: 分子整体类指标大图 ==============
def plot_v3_whole_mol(df, out_dir):
    """V3: 分子整体类指标 (Rg, c/a, SASA) × 3 任务"""
    whole_metrics = [
        ('mol_rg', '分子回转半径', 'Å'),
        ('mol_ca_ratio', '椭球扁率 c/a', ''),
        ('mol_sasa', '溶剂可及表面积', 'Å²'),
    ]

    fig, axes = plt.subplots(len(whole_metrics), 3, figsize=(18, 5 * len(whole_metrics)))
    for i, (mkey, mname, munit) in enumerate(whole_metrics):
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[i, j]
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
                if np.isnan(x0) or np.isnan(x1):
                    continue
                ax.scatter(x0, y0, c=color, alpha=TYPE2_ALPHA[1], s=120,
                           edgecolors='black', linewidths=0.6, zorder=3,
                           marker=TYPE2_MARKER[1])
                ax.scatter(x1, y1, c=color, alpha=TYPE2_ALPHA[2], s=140,
                           edgecolors='black', linewidths=0.8, zorder=4,
                           marker=TYPE2_MARKER[2])
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='->', color=color,
                                        alpha=0.6, mutation_scale=13,
                                        linewidth=1.3, zorder=2)
                ax.add_patch(arrow)
            xlabel = f"{mname} ({munit})" if munit else mname
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(tdesc, fontsize=11)
            if i == 0:
                ax.set_title(tname, fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=9)

    legend = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        legend.append(Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=TYPE1_COLORS[t1], markersize=11,
                             markeredgecolor='black', markeredgewidth=0.5,
                             label=TYPE1_NAMES[t1]))
    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                         alpha=0.45, markersize=11, markeredgecolor='black',
                         markeredgewidth=0.5, label='反应物 (浅圆)'))
    legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                         alpha=1.0, markersize=11, markeredgecolor='black',
                         markeredgewidth=0.5, label='产物 (深方)'))
    legend.append(Line2D([0], [0], color='gray', alpha=0.6, linewidth=1.5,
                         label='反应物 → 产物'))
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle('V3: 分子整体立体指标 (Rg / c/a / SASA) vs 预测芳香性',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = os.path.join(out_dir, 'V3_whole_molecule.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V3 → {out_path}")


# ============== V4: Δ变化散点图 ==============
def plot_v4_delta(df, out_dir):
    """V4: Δ立体指标 (X) vs Δ预测值 (Y), 每点=一个反应对"""
    delta_rows = []
    for pid in df['pair_id'].unique():
        sub = df[df['pair_id'] == pid].sort_values('TYPE2')
        if len(sub) != 2:
            continue
        r, p = sub.iloc[0], sub.iloc[1]
        row = {'pair_id': pid, 'TYPE1': int(r['TYPE1'])}
        for mkey, _, _, _ in STEREO_METRICS:
            row[f'd_{mkey}'] = p[mkey] - r[mkey]
        for tkey, tname, _ in TASKS:
            row[f'd_{tkey}'] = p[tkey] - r[tkey]
        delta_rows.append(row)
    ddf = pd.DataFrame(delta_rows)

    n_m = len(STEREO_METRICS)
    fig, axes = plt.subplots(n_m, 3, figsize=(16, 3.4 * n_m))

    for i, (mkey, mname, munit, mcat) in enumerate(STEREO_METRICS):
        for j, (tkey, tname, tdesc) in enumerate(TASKS):
            ax = axes[i, j]
            for t1 in sorted(TYPE1_COLORS.keys()):
                sub = ddf[ddf['TYPE1'] == t1]
                ax.scatter(sub[f'd_{mkey}'], sub[f'd_{tkey}'],
                           c=TYPE1_COLORS[t1], s=140, alpha=0.85,
                           edgecolors='black', linewidths=0.8,
                           label=TYPE1_NAMES[t1], zorder=3)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            xlabel = f'Δ {mname}' + (f' ({munit})' if munit else '')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(f'Δ {tname}', fontsize=9)
            if i == 0:
                ax.set_title(tname, fontsize=12, fontweight='bold')
            if j == 0:
                ax.text(-0.30, 0.5, f"[{mcat}]\n{mname}", transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        rotation=90)
            ax.grid(alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=8)

    legend = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=TYPE1_COLORS[t1], markersize=11,
                     markeredgecolor='black', markeredgewidth=0.5,
                     label=TYPE1_NAMES[t1])
              for t1 in sorted(TYPE1_COLORS.keys())]
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle('V4: Δ立体程度 vs Δ预测值 (产物 - 反应物)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    out_path = os.path.join(out_dir, 'V4_delta_scatter.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V4 → {out_path}")


# ============== V5: 综合仪表盘 ==============
def plot_v5_dashboard(df, out_dir):
    """V5: 综合仪表盘
    (1) ring_Q vs HOMA   (2) butterfly vs HOMA   (3) 平均变化柱状图
    (4) 雷达图            (5) Δring_Q vs ΔHOMA    (6) core_rmsd vs NICS
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.30)

    def draw_scatter(ax, mkey, tkey, mname, munit, tdesc):
        for pid in df['pair_id'].unique():
            sub = df[df['pair_id'] == pid].sort_values('TYPE2')
            if len(sub) != 2:
                continue
            r, p = sub.iloc[0], sub.iloc[1]
            t1 = int(r['TYPE1'])
            color = TYPE1_COLORS[t1]
            if np.isnan(r[mkey]) or np.isnan(p[mkey]):
                continue
            ax.scatter(r[mkey], r[tkey], c=color, alpha=0.45, s=120,
                       edgecolors='black', linewidths=0.5, zorder=3, marker='o')
            ax.scatter(p[mkey], p[tkey], c=color, alpha=1.0, s=140,
                       edgecolors='black', linewidths=0.7, zorder=4, marker='s')
            arrow = FancyArrowPatch((r[mkey], r[tkey]), (p[mkey], p[tkey]),
                                    arrowstyle='->', color=color, alpha=0.55,
                                    mutation_scale=12, linewidth=1.2, zorder=2)
            ax.add_patch(arrow)
        ax.set_xlabel(f"{mname} ({munit})" if munit else mname, fontsize=11)
        ax.set_ylabel(tdesc, fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')

    # (1) ring_Q vs HOMA
    ax1 = fig.add_subplot(gs[0, 0])
    draw_scatter(ax1, 'ring_Q', 'HOMA_pred', 'Cremer-Pople Q', 'Å', 'HOMA')
    ax1.set_title('Cremer-Pople Q vs HOMA', fontsize=12, fontweight='bold')

    # (2) butterfly_angle vs HOMA
    ax2 = fig.add_subplot(gs[0, 1])
    draw_scatter(ax2, 'butterfly_angle', 'HOMA_pred', '稠环折叠角', '°', 'HOMA')
    ax2.set_title('稠环折叠角 vs HOMA (180°=平面)', fontsize=12, fontweight='bold')

    # (3) core_rmsd vs NICS
    ax3 = fig.add_subplot(gs[0, 2])
    draw_scatter(ax3, 'core_rmsd', 'NICS_1zz_pred', '母核 RMSD', 'Å', 'NICS(1)zz')
    ax3.set_title('母核平面 RMSD vs NICS(1)zz', fontsize=12, fontweight='bold')

    # (4) 平均变化柱状图 (3 任务 × 4 TYPE1)
    ax4 = fig.add_subplot(gs[1, 0])
    delta_rows = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        sub = df[df['TYPE1'] == t1]
        for pid in sub['pair_id'].unique():
            psub = sub[sub['pair_id'] == pid].sort_values('TYPE2')
            if len(psub) != 2:
                continue
            r, p = psub.iloc[0], psub.iloc[1]
            for tkey, tname, _ in TASKS:
                delta_rows.append({'TYPE1': t1, 'task': tname, 'delta': p[tkey] - r[tkey]})
    ddf = pd.DataFrame(delta_rows)
    pivot = ddf.groupby(['TYPE1', 'task'])['delta'].mean().unstack()
    pivot = pivot[[t[1] for t in TASKS]]
    pivot.plot(kind='bar', ax=ax4, edgecolor='black', linewidth=0.6, width=0.8)
    ax4.set_xticklabels([TYPE1_NAMES[i] for i in pivot.index], rotation=0, fontsize=9)
    ax4.set_ylabel('平均 Δ (产物 - 反应物)', fontsize=11)
    ax4.set_title('各反应类型平均指标变化', fontsize=12, fontweight='bold')
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend(title='指标', fontsize=9, title_fontsize=10)

    # (5) Δring_Q vs ΔHOMA
    ax5 = fig.add_subplot(gs[1, 1])
    for t1 in sorted(TYPE1_COLORS.keys()):
        subs = []
        for pid in df[df['TYPE1'] == t1]['pair_id'].unique():
            psub = df[df['pair_id'] == pid].sort_values('TYPE2')
            if len(psub) != 2:
                continue
            r, p = psub.iloc[0], psub.iloc[1]
            subs.append({'d_Q': p['ring_Q'] - r['ring_Q'],
                         'd_HOMA': p['HOMA_pred'] - r['HOMA_pred']})
        spp = pd.DataFrame(subs)
        ax5.scatter(spp['d_Q'], spp['d_HOMA'], c=TYPE1_COLORS[t1], s=150,
                    alpha=0.85, edgecolors='black', linewidths=0.8,
                    label=TYPE1_NAMES[t1], zorder=3)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Δ Cremer-Pople Q (Å)', fontsize=11)
    ax5.set_ylabel('Δ HOMA', fontsize=11)
    ax5.set_title('ΔQ vs ΔHOMA (产物-反应物)', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, linestyle='--')
    ax5.legend(fontsize=9)

    # (6) ring_dihed_sum vs MBCO
    ax6 = fig.add_subplot(gs[1, 2])
    draw_scatter(ax6, 'ring_dihed_sum', 'MBCO_pred', '环二面角和', '°', 'MBCO')
    ax6.set_title('环二面角绝对值和 vs MBCO', fontsize=12, fontweight='bold')

    # (7) 雷达图 (下半, 跨 3 列)
    ax7 = fig.add_subplot(gs[2, :], polar=True)
    radar_metrics = ['ring_Q', 'core_rmsd', 'ring_dihed_sum', 'ring_max_dev',
                     'mol_rg', 'mol_ca_ratio', 'mol_sasa']
    # 归一化
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
            ax7.plot(angles, vals, '-', color=color, alpha=TYPE2_ALPHA[t2],
                     linewidth=2 if t2 == 2 else 1.2,
                     label=f'{TYPE1_NAMES[t1]} - {TYPE2_NAMES[t2]}')
            ax7.fill(angles, vals, color=color, alpha=0.10 if t2 == 1 else 0.22)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(['Q', '母核RMSD', '环二面角和', '环最大偏离',
                         'Rg', 'c/a', 'SASA'], fontsize=10)
    ax7.set_title('立体指标雷达图 (归一化; 浅=反应物, 深=产物)',
                  fontsize=13, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9, ncol=1)

    legend = []
    for t1 in sorted(TYPE1_COLORS.keys()):
        legend.append(Line2D([0], [0], color=TYPE1_COLORS[t1], linewidth=3,
                              marker='o', markersize=8, label=TYPE1_NAMES[t1]))
    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                         alpha=0.45, markersize=10, label='反应物 (浅圆)'))
    legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                         alpha=1.0, markersize=10, label='产物 (深方)'))
    legend.append(Line2D([0], [0], color='gray', alpha=0.6, linewidth=1.5,
                         label='反应物 → 产物'))
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle('V5: 综合仪表盘 — 环扭曲/折叠 vs 芳香性预测',
                 fontsize=15, fontweight='bold', y=0.995)
    out_path = os.path.join(out_dir, 'V5_dashboard.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  V5 → {out_path}")


# ============== V6: 热力图 ==============
def plot_v6_heatmap(df, out_dir):
    """V6: 17 反应对 × 11 指标 Δ值热力图 (8 立体 + 3 预测)"""
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
        for mkey, mname, _, _ in STEREO_METRICS:
            row[f'Δ{mname}'] = p[mkey] - r[mkey]
        for tkey, tname, _ in TASKS:
            row[f'Δ{tname}'] = p[tkey] - r[tkey]
        delta_rows.append(row)
        row_labels.append(pid)
        row_types.append(t1)
    ddf = pd.DataFrame(delta_rows, index=row_labels)
    ddf['_t1'] = row_types
    ddf = ddf.sort_values(['_t1']).drop(columns=['_t1'])

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(ddf, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.6, linecolor='gray',
                cbar_kws={'label': 'Δ 值 (产物 - 反应物)', 'shrink': 0.8},
                annot_kws={'fontsize': 8})
    ax.set_title('V6: 反应前后指标变化热力图\n(红=增加, 蓝=减少; 浅=立体, 右=预测)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('反应对 (按反应类型分组)', fontsize=11)
    ax.set_xlabel('指标 (Δ = 产物 - 反应物)', fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

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


# ============== V7: 单指标单任务大图 (最受欢迎的几个组合) ==============
def plot_v7_highlight_pairs(df, out_dir):
    """V7: 最具化学意义的指标-任务组合, 每张图一个组合, 大图高清"""
    highlights = [
        ('ring_Q', 'HOMA_pred', 'Cremer-Pople Q (Å)', '预测 HOMA',
         'Cremer-Pople 环褶皱 Q vs HOMA\n(芳环去芳构化: Q增大, HOMA减小)'),
        ('butterfly_angle', 'HOMA_pred', '稠环折叠角 (°)', '预测 HOMA',
         '稠环折叠角 vs HOMA\n(180°=共平面; 去芳构化折叠, HOMA降低)'),
        ('ring_Q', 'NICS_1zz_pred', 'Cremer-Pople Q (Å)', '预测 NICS(1)zz (ppm)',
         'Cremer-Pople Q vs NICS(1)zz\n(芳环: Q≈0, NICS≈-25; 去芳构: Q↑, NICS→0)'),
        ('core_rmsd', 'MBCO_pred', '母核平面 RMSD (Å)', '预测 MBCO',
         '母核平面 RMSD vs MBCO\n(母核扭曲越大, MBCO越低)'),
        ('ring_dihed_sum', 'HOMA_pred', '环二面角绝对值和 (°)', '预测 HOMA',
         '环二面角扭曲总量 vs HOMA\n(0°=完美平面; 扭曲↑, HOMA↓)'),
        ('butterfly_angle', 'NICS_1zz_pred', '稠环折叠角 (°)', '预测 NICS(1)zz (ppm)',
         '稠环折叠角 vs NICS(1)zz\n(折叠角减小=立体折叠, NICS向0靠拢)'),
    ]

    for i, (mkey, tkey, mlabel, tlabel, title) in enumerate(highlights):
        fig, ax = plt.subplots(figsize=(10, 7))
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
            if np.isnan(x0) or np.isnan(x1):
                continue
            ax.scatter(x0, y0, c=color, alpha=TYPE2_ALPHA[1], s=200,
                       edgecolors='black', linewidths=0.7, zorder=3,
                       marker=TYPE2_MARKER[1])
            ax.scatter(x1, y1, c=color, alpha=TYPE2_ALPHA[2], s=230,
                       edgecolors='black', linewidths=0.9, zorder=4,
                       marker=TYPE2_MARKER[2])
            arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                    arrowstyle='->', color=color,
                                    alpha=0.6, mutation_scale=18,
                                    linewidth=1.5, zorder=2)
            ax.add_patch(arrow)

        ax.set_xlabel(mlabel, fontsize=13)
        ax.set_ylabel(tlabel, fontsize=13)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)

        legend = []
        for t1 in sorted(TYPE1_COLORS.keys()):
            legend.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=TYPE1_COLORS[t1], markersize=13,
                                 markeredgecolor='black', markeredgewidth=0.6,
                                 label=TYPE1_NAMES[t1]))
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                             alpha=0.45, markersize=13, markeredgecolor='black',
                             markeredgewidth=0.6, label='反应物 (浅圆)'))
        legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                             alpha=1.0, markersize=13, markeredgecolor='black',
                             markeredgewidth=0.6, label='产物 (深方)'))
        legend.append(Line2D([0], [0], color='gray', alpha=0.6, linewidth=1.8,
                             label='反应物 → 产物'))
        ax.legend(handles=legend, loc='best', fontsize=10, frameon=True)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f'V7_highlight_{i+1}_{mkey}_vs_{tkey}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  V7_{i+1} → {out_path}")


# ============== 主入口 ==============
def main():
    print(f"加载数据: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"\n原始数据: {len(df)} 行, {df['pair_id'].nunique()} 反应对")

    # 过滤异常数据: ring_Q > 0.4 Å (g32 及其配对反应物 g31)
    # 或直接过滤掉 pair_id = T4-P4
    outlier_pairs = df[df['ring_Q'] > 0.4]['pair_id'].unique()
    if len(outlier_pairs) > 0:
        print(f"\n过滤异常反应对: {outlier_pairs.tolist()} (ring_Q > 0.4 Å)")
        df = df[~df['pair_id'].isin(outlier_pairs)]
        print(f"过滤后: {len(df)} 行, {df['pair_id'].nunique()} 反应对")

    print(f"\n开始绘图 (输出: {OUT_DIR})...")

    plot_v1_full_matrix(df, OUT_DIR)
    plot_v2_key_large(df, OUT_DIR)
    plot_v3_whole_mol(df, OUT_DIR)
    plot_v4_delta(df, OUT_DIR)
    plot_v5_dashboard(df, OUT_DIR)
    plot_v6_heatmap(df, OUT_DIR)
    plot_v7_highlight_pairs(df, OUT_DIR)

    # 保存数据
    out_csv = os.path.join(OUT_DIR, 'lunci8_with_stereo_v2.csv')
    df.drop(columns=['ring_atoms_list']).to_csv(out_csv, index=False)
    print(f"\n带立体指标数据: {out_csv}")
    print(f"所有图保存到: {OUT_DIR}/")


if __name__ == '__main__':
    main()
