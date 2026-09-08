"""
高级化学空间分析 (Part 2)
包含: PMI修复、Bertz复杂度、官能团统计、环系细分、骨架层级、
      最近邻Tanimoto、覆盖度三要素、TMAP树状图
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from collections import Counter
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
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PROJ_ROOT = _PROJ_ROOT
DATASET_PATH = os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv')
OUTPUT_DIR = os.path.join(PROJ_ROOT, '0427_unified_results', 'chemical_space')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 数据加载与分子属性计算 ==========

def load_and_compute():
    """加载数据并计算所有分子属性"""
    df = pd.read_csv(DATASET_PATH)
    unique_smiles = df['smiles'].unique()
    print(f"数据集: {len(df)} 行, {len(unique_smiles)} 个唯一分子")

    props = []
    fps = []
    print("计算分子描述符...")

    for idx, smiles in enumerate(unique_smiles):
        if idx % 500 == 0:
            print(f"  进度: {idx}/{len(unique_smiles)}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 基本属性
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        # Bertz 复杂度
        bertz = Descriptors.BertzCT(mol)

        # PMI
        pmi1, pmi2, pmi3 = 0, 0, 0
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == 0:
                AllChem.MMFFOptimizeMolecule(mol_h)
                pmi1 = rdMolDescriptors.CalcPMI1(mol_h)
                pmi2 = rdMolDescriptors.CalcPMI2(mol_h)
                pmi3 = rdMolDescriptors.CalcPMI3(mol_h)
        except Exception:
            pass

        # Murcko 骨架
        scaffold_smiles = ''
        generic_scaffold = ''
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            generic_scaffold = Chem.MolToSmiles(generic)
        except Exception:
            pass

        # 环系细分
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        n_sized_rings = len(atom_rings)

        # 稠环/螺环/桥环检测
        n_fused = 0
        n_spiro = 0
        n_bridged = 0
        ring_sets = [set(r) for r in atom_rings]
        for i in range(len(ring_sets)):
            for j in range(i+1, len(ring_sets)):
                shared = ring_sets[i] & ring_sets[j]
                if len(shared) == 0:
                    continue
                elif len(shared) == 1:
                    n_spiro += 1
                elif len(shared) == 2:
                    # 检查是否桥环 (共享边但还有其他共享原子路径)
                    n_fused += 1
                else:
                    n_bridged += 1

        # 杂环原子统计
        ring_atom_set = set()
        for r in atom_rings:
            ring_atom_set.update(r)
        ring_n_atoms = [mol.GetAtomWithIdx(i) for i in ring_atom_set]
        ring_n_N = sum(1 for a in ring_n_atoms if a.GetAtomicNum() == 7)
        ring_n_O = sum(1 for a in ring_n_atoms if a.GetAtomicNum() == 8)
        ring_n_S = sum(1 for a in ring_n_atoms if a.GetAtomicNum() == 16)
        ring_n_C = sum(1 for a in ring_n_atoms if a.GetAtomicNum() == 6)

        # 芳香/脂肪环分类
        n_aromatic_rings_fg = 0
        n_aliphatic_rings_fg = 0
        n_hetero_aromatic = 0
        n_carbo_aromatic = 0
        for r in atom_rings:
            is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            has_hetero = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r)
            if is_aromatic:
                n_aromatic_rings_fg += 1
                if has_hetero:
                    n_hetero_aromatic += 1
                else:
                    n_carbo_aromatic += 1
            else:
                n_aliphatic_rings_fg += 1

        # 官能团检测
        fg_counts = detect_functional_groups(mol)

        # 分子指纹
        fp = None
        try:
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = fp_gen.GetFingerprint(mol)
        except Exception:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            except Exception:
                pass
        fps.append(fp)

        props.append({
            'smiles': smiles,
            'mw': mw, 'logp': logp, 'n_heavy': n_heavy,
            'n_rings': n_rings, 'n_aromatic_rings': n_aromatic,
            'n_aliphatic_rings': n_aliphatic, 'fsp3': fsp3,
            'bertz': bertz,
            'pmi1': pmi1, 'pmi2': pmi2, 'pmi3': pmi3,
            'scaffold': scaffold_smiles,
            'generic_scaffold': generic_scaffold,
            'n_ring_systems': n_sized_rings,
            'n_fused': n_fused, 'n_spiro': n_spiro, 'n_bridged': n_bridged,
            'ring_n_N': ring_n_N, 'ring_n_O': ring_n_O, 'ring_n_S': ring_n_S,
            'ring_n_C': ring_n_C,
            'n_aromatic_fg': n_aromatic_rings_fg,
            'n_aliphatic_fg': n_aliphatic_rings_fg,
            'n_hetero_aromatic': n_hetero_aromatic,
            'n_carbo_aromatic': n_carbo_aromatic,
            **fg_counts,
        })

    props_df = pd.DataFrame(props)
    print(f"成功计算 {len(props_df)} 个分子")
    return props_df, fps, df


def detect_functional_groups(mol):
    """检测官能团"""
    fg = {}
    smarts_patterns = {
        'hydroxyl (-OH)': '[OX2H]',
        'primary amine (-NH2)': '[NX3H2]',
        'secondary amine (-NH-)': '[NX3H1]',
        'tertiary amine (-N<)': '[NX3H0]',
        'aromatic amine (Ar-N)': '[a][NX3]',
        'amide (C(=O)N)': '[CX3](=[OX1])[NX3H2,H1,H0]',
        'carbonyl (C=O) ketone': '[CX3]=[OX1]',
        'aldehyde (-CHO)': '[CX3H1]=[OX1]',
        'carboxylic acid (-COOH)': '[CX3](=[OX1])[OX2H1]',
        'ester (-COO-)': '[CX3](=[OX1])[OX2H0]',
        'ether (-O-)': '[OD2]([#6])[#6]',
        'nitro (-NO2)': '[$([NX3](=O)=O)]',
        'nitrile (-C≡N)': '[CX1]#[NX1]',
        'sulfide (-S-)': '[SX2]',
        'sulfonyl (-SO2-)': '[$([SX4](=[OX1])(=[OX1])([#6])([#6]))]',
        'halide (F)': '[F]',
        'halide (Cl)': '[Cl]',
        'halide (Br)': '[Br]',
        'halide (I)': '[I]',
        'alkene (C=C)': '[CX2]=[CX2]',
        'alkyne (C≡C)': '[CX1]#[CX1]',
        'aromatic ring (benzene)': 'c1ccccc1',
        'pyridine': 'c1ccncc1',
        'furan': 'c1ccoc1',
        'thiophene': 'c1ccsc1',
        'pyrrole': 'c1cc[nH]c1',
        'imidazole': 'c1cnc[nH]1',
        'pyrimidine': 'c1cncnc1',
        'indole': 'c1ccc2c(c1)[nH]cc2',
        'pyrazine': 'c1cnccn1',
    }

    for name, smarts in smarts_patterns.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                matches = mol.GetSubstructMatches(patt)
                fg[name] = len(matches)
            else:
                fg[name] = 0
        except Exception:
            fg[name] = 0

    return fg


# ========== 图表生成函数 ==========

def plot_pmi_fixed(props_df, save_dir):
    """图2修复: PMI 三角形图"""
    valid = props_df[(props_df['pmi1'] > 0) & (props_df['pmi2'] > 0) & (props_df['pmi3'] > 0)].copy()
    if len(valid) == 0:
        print("  PMI 数据不足, 跳过")
        return

    valid['n1'] = valid['pmi1'] / valid['pmi3']
    valid['n2'] = valid['pmi2'] / valid['pmi3']

    fig, ax = plt.subplots(figsize=(9, 9))

    # 三角形: 顶点为 Rod(0,0), Disc(0,1), Sphere(1,1)
    # 由于 n1 <= n2, 所有点在 n2 >= n1 的区域 (上三角)
    triangle_x = [0, 0, 1, 0]
    triangle_y = [0, 1, 1, 0]
    ax.fill(triangle_x, triangle_y, alpha=0.08, color='blue')
    ax.plot([0, 0, 1, 0], [0, 1, 1, 0], 'k-', linewidth=2)

    # 对角线 (n1 = n2, 即 disc-to-sphere 边界)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)

    # 顶点标注 (修正位置)
    ax.annotate('Rod\n(1D)', xy=(0, 0), xytext=(0.08, 0.08),
                fontsize=14, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.annotate('Disc\n(2D)', xy=(0, 1), xytext=(0.08, 0.88),
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.annotate('Sphere\n(3D)', xy=(1, 1), xytext=(0.82, 0.88),
                fontsize=14, fontweight='bold', color='blue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # 数据点
    sc = ax.scatter(valid['n1'], valid['n2'], c=valid['mw'], cmap='viridis',
                    alpha=0.5, s=10, edgecolors='none', zorder=5)
    plt.colorbar(sc, ax=ax, label='Molecular Weight (Da)', shrink=0.7)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('n₁ = I₁/I₃ (normalized)', fontsize=13)
    ax.set_ylabel('n₂ = I₂/I₃ (normalized)', fontsize=13)
    ax.set_title('Principal Moments of Inertia (PMI)\nMolecular Shape Analysis', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    # 统计
    n_rod = sum((valid['n1'] < 0.2) & (valid['n2'] < 0.3))
    n_disc = sum((valid['n1'] < 0.3) & (valid['n2'] > 0.6))
    n_sphere = sum((valid['n1'] > 0.6) & (valid['n2'] > 0.8))
    ax.text(0.95, 0.05, f'Rod-like: {n_rod}\nDisc-like: {n_disc}\nSphere-like: {n_sphere}',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_pmi_shape.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图2: PMI 形状图 (修复) ✓")


def plot_bertz(props_df, save_dir):
    """图11: Bertz 复杂度指数分布"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 直方图
    ax = axes[0]
    bertz_vals = props_df['bertz'].values
    ax.hist(bertz_vals, bins=60, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(bertz_vals), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(bertz_vals):.0f}')
    ax.axvline(np.median(bertz_vals), color='green', linestyle='--', linewidth=2,
               label=f'Median={np.median(bertz_vals):.0f}')
    ax.set_xlabel('Bertz Complexity Index (CT)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Bertz Complexity Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 箱线图 (按环数分组)
    ax = axes[1]
    ring_groups = sorted(props_df['n_rings'].unique())
    bertz_by_rings = [props_df[props_df['n_rings']==r]['bertz'].values for r in ring_groups]
    bp = ax.boxplot(bertz_by_rings, labels=[str(r) for r in ring_groups],
                    patch_artist=True, widths=0.6)
    colors = cm.Set2(np.linspace(0, 1, len(ring_groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Number of Rings', fontsize=12)
    ax.set_ylabel('Bertz Complexity (CT)', fontsize=12)
    ax.set_title('Bertz Complexity by Ring Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Bertz Complexity Index (Molecular Topological Complexity)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '11_bertz_complexity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图11: Bertz 复杂度 ✓")


def plot_functional_groups(props_df, save_dir):
    """图12: 官能团丰度统计"""
    fg_cols = [c for c in props_df.columns if c not in 
               ['smiles','mw','logp','n_heavy','n_rings','n_aromatic_rings',
                'n_aliphatic_rings','fsp3','bertz','pmi1','pmi2','pmi3',
                'scaffold','generic_scaffold','n_ring_systems','n_fused','n_spiro',
                'n_bridged','ring_n_N','ring_n_O','ring_n_S','ring_n_C',
                'n_aromatic_fg','n_aliphatic_fg','n_hetero_aromatic','n_carbo_aromatic']]

    # 统计: 出现该官能团的分子数 (count > 0)
    fg_molecule_counts = {}
    fg_total_counts = {}
    for col in fg_cols:
        fg_molecule_counts[col] = (props_df[col] > 0).sum()
        fg_total_counts[col] = props_df[col].sum()

    # 排序
    sorted_fg = sorted(fg_molecule_counts.items(), key=lambda x: x[1], reverse=True)
    top_fg = sorted_fg[:25]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 左图: 含该官能团的分子数
    ax = axes[0]
    names = [x[0] for x in top_fg]
    counts = [x[1] for x in top_fg]
    colors_bar = cm.tab20(np.linspace(0, 1, len(names)))
    ax.barh(range(len(names)), counts, color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Molecules Containing FG', fontsize=12)
    ax.set_title('Top 25 Functional Groups - Molecule Coverage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(counts):
        pct = v / len(props_df) * 100
        ax.text(v + 5, i, f'{v} ({pct:.1f}%)', va='center', fontsize=8)

    # 右图: 官能团总出现次数
    ax = axes[1]
    sorted_total = sorted(fg_total_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    names2 = [x[0] for x in sorted_total]
    counts2 = [x[1] for x in sorted_total]
    ax.barh(range(len(names2)), counts2, color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(names2)))
    ax.set_yticklabels(names2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Total Occurrences', fontsize=12)
    ax.set_title('Top 25 Functional Groups - Total Frequency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(counts2):
        ax.text(v + 5, i, str(v), va='center', fontsize=8)

    plt.suptitle('Functional Group (FG) Abundance & Diversity', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '12_functional_groups.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图12: 官能团统计 ✓")


def plot_ring_system_types(props_df, save_dir):
    """图13: 环系类型细分分布"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 13a: 芳香环 vs 脂肪环 vs 无环
    ax = axes[0][0]
    categories = []
    for _, row in props_df.iterrows():
        if row['n_aromatic_fg'] > 0 and row['n_aliphatic_fg'] > 0:
            categories.append('Aromatic + Aliphatic')
        elif row['n_aromatic_fg'] > 0:
            categories.append('Aromatic only')
        elif row['n_aliphatic_fg'] > 0:
            categories.append('Aliphatic only')
        else:
            categories.append('No rings')
    cat_counts = Counter(categories)
    labels = list(cat_counts.keys())
    sizes = list(cat_counts.values())
    colors_pie = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie[:len(labels)],
           startangle=90, textprops={'fontsize': 11})
    ax.set_title('Ring Type: Aromatic vs Aliphatic', fontsize=13, fontweight='bold')

    # 13b: 杂环 vs 碳环 (芳香环)
    ax = axes[0][1]
    n_carbo = props_df['n_carbo_aromatic'].sum()
    n_hetero = props_df['n_hetero_aromatic'].sum()
    ax.pie([n_carbo, n_hetero], labels=['Carbo-aromatic', 'Hetero-aromatic'],
           autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'], startangle=90,
           textprops={'fontsize': 12})
    ax.set_title('Aromatic Rings: Carbo vs Hetero', fontsize=13, fontweight='bold')

    # 13c: 环连接方式 (单环/稠环/螺环/桥环)
    ax = axes[1][0]
    connection_types = []
    for _, row in props_df.iterrows():
        if row['n_fused'] > 0:
            connection_types.append('Fused')
        elif row['n_spiro'] > 0:
            connection_types.append('Spiro')
        elif row['n_bridged'] > 0:
            connection_types.append('Bridged')
        elif row['n_ring_systems'] > 1:
            connection_types.append('Isolated multi-ring')
        elif row['n_ring_systems'] == 1:
            connection_types.append('Single ring')
        else:
            connection_types.append('No ring')
    conn_counts = Counter(connection_types)
    labels3 = list(conn_counts.keys())
    sizes3 = list(conn_counts.values())
    colors3 = cm.Set2(np.linspace(0, 1, len(labels3)))
    ax.bar(range(len(labels3)), sizes3, color=colors3, edgecolor='white')
    ax.set_xticks(range(len(labels3)))
    ax.set_xticklabels(labels3, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Number of Molecules', fontsize=12)
    ax.set_title('Ring Connection Types', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sizes3):
        ax.text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')

    # 13d: 杂环原子类型 (N/O/S)
    ax = axes[1][1]
    total_N = props_df['ring_n_N'].sum()
    total_O = props_df['ring_n_O'].sum()
    total_S = props_df['ring_n_S'].sum()
    total_C = props_df['ring_n_C'].sum()
    atom_labels = ['Carbon (C)', 'Nitrogen (N)', 'Oxygen (O)', 'Sulfur (S)']
    atom_counts = [total_C, total_N, total_O, total_S]
    colors4 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    ax.bar(atom_labels, atom_counts, color=colors4, edgecolor='white')
    ax.set_ylabel('Total Ring Atom Count', fontsize=12)
    ax.set_title('Ring Heteroatom Composition', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(atom_counts):
        ax.text(i, v + 50, str(v), ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Ring System Type Breakdown', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '13_ring_system_types.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图13: 环系类型细分 ✓")


def plot_scaffold_hierarchy(props_df, save_dir):
    """图14: Murcko 骨架层级多样性"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 层级统计
    n_molecules = len(props_df)
    n_scaffolds = props_df['scaffold'].nunique()
    n_generic = props_df['generic_scaffold'].nunique()

    # 左图: 层级数量对比
    ax = axes[0]
    levels = ['Molecules', 'Unique\nScaffolds', 'Generic\nScaffolds (all-C)']
    counts = [n_molecules, n_scaffolds, n_generic]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bars = ax.bar(levels, counts, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}\n({val/n_molecules*100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Scaffold Hierarchy Diversity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 右图: Top 10 骨架占比
    ax = axes[1]
    scaffold_counts = props_df['scaffold'].value_counts().head(10)
    generic_counts = props_df['generic_scaffold'].value_counts().head(10)

    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, scaffold_counts.values, width, label='Specific Scaffold', color='#4C72B0')
    ax.bar(x + width/2, [generic_counts.get(s, 0) for s in scaffold_counts.index], 
           width, label='Generic Scaffold', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(10)], fontsize=10)
    ax.set_ylabel('Molecule Count', fontsize=12)
    ax.set_title('Top 10 Scaffolds: Specific vs Generic', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 表格
    table_data = []
    for i, (scaffold, count) in enumerate(scaffold_counts.items()):
        gen_count = (props_df['generic_scaffold'] == 
                     props_df[props_df['scaffold']==scaffold]['generic_scaffold'].iloc[0]).sum()
        table_data.append([f'S{i+1}', scaffold[:30], count, gen_count])

    plt.suptitle('Murcko Scaffold Hierarchy Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '14_scaffold_hierarchy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存层级表
    hierarchy_df = pd.DataFrame({
        'Level': ['Molecules', 'Unique Scaffolds', 'Generic Scaffolds'],
        'Count': [n_molecules, n_scaffolds, n_generic],
        'Diversity (%)': [100, n_scaffolds/n_molecules*100, n_generic/n_molecules*100],
    })
    hierarchy_df.to_csv(os.path.join(save_dir, 'scaffold_hierarchy.csv'), index=False)
    print("  图14: 骨架层级多样性 ✓")


def plot_nn_tanimoto(fps, save_dir):
    """图15: 最近邻 Tanimoto 相似性分布"""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图15: 指纹数据不足, 跳过")
        return

    print("  计算最近邻 Tanimoto 相似性...")
    nn_sims = []
    for i in valid_idx:
        max_sim = 0
        for j in valid_idx:
            if i == j:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim > max_sim:
                max_sim = sim
            if max_sim == 1.0:
                break
        nn_sims.append(max_sim)
        if len(nn_sims) % 500 == 0:
            print(f"    进度: {len(nn_sims)}/{len(valid_idx)}")

    nn_sims = np.array(nn_sims)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 直方图
    ax = axes[0]
    ax.hist(nn_sims, bins=80, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(nn_sims), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(nn_sims):.4f}')
    ax.axvline(np.median(nn_sims), color='green', linestyle='--', linewidth=2,
               label=f'Median={np.median(nn_sims):.4f}')
    ax.axvline(0.85, color='orange', linestyle=':', linewidth=2, label='T=0.85 (very similar)')
    ax.set_xlabel('Nearest-Neighbor Tanimoto Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Nearest-Neighbor Tanimoto Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 累积分布曲线
    ax = axes[1]
    sorted_sims = np.sort(nn_sims)
    cdf = np.arange(1, len(sorted_sims)+1) / len(sorted_sims)
    ax.plot(sorted_sims, cdf, 'b-', linewidth=2)
    ax.fill_between(sorted_sims, 0, cdf, alpha=0.2, color='blue')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(np.median(nn_sims), color='green', linestyle='--', linewidth=2,
               label=f'Median={np.median(nn_sims):.4f}')
    ax.set_xlabel('Nearest-Neighbor Tanimoto Similarity', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative Distribution of NN Similarity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Nearest-Neighbor Tanimoto Similarity Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '15_nn_tanimoto.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图15: 最近邻 Tanimoto ✓")


def plot_coverage_metrics(fps, save_dir):
    """图16: 化学空间覆盖度三要素"""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图16: 指纹数据不足, 跳过")
        return

    # 提取指纹数组
    fp_arrays = []
    for i in valid_idx:
        arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fps[i], arr)
        fp_arrays.append(arr)
    X = np.array(fp_arrays)

    # PCA 降维到 2D
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 1. 丰富度 (Richness): 网格化后有分子的网格数
    n_bins = 20
    x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
    x_bins = np.linspace(x_min, x_max + 0.01, n_bins + 1)
    y_bins = np.linspace(y_min, y_max + 0.01, n_bins + 1)

    grid = np.zeros((n_bins, n_bins))
    for i in range(len(X_pca)):
        xi = min(int((X_pca[i, 0] - x_min) / (x_max - x_min) * n_bins), n_bins - 1)
        yi = min(int((X_pca[i, 1] - y_min) / (y_max - y_min) * n_bins), n_bins - 1)
        grid[yi, xi] += 1

    richness = np.sum(grid > 0)
    total_cells = n_bins * n_bins
    richness_pct = richness / total_cells * 100

    # 2. 均匀度 (Uniformity): Shannon entropy / max entropy
    p = grid[grid > 0] / grid[grid > 0].sum()
    shannon = -np.sum(p * np.log(p))
    max_shannon = np.log(richness) if richness > 0 else 1
    uniformity = shannon / max_shannon if max_shannon > 0 else 0

    # 3. 广度 (Extent): 最远两点的欧氏距离
    from scipy.spatial.distance import pdist
    if len(X_pca) > 1000:
        # 采样以加速
        sample_idx = np.random.choice(len(X_pca), 1000, replace=False)
        distances = pdist(X_pca[sample_idx])
    else:
        distances = pdist(X_pca)
    extent = np.max(distances)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左图: 网格热力图
    ax = axes[0]
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Molecule Count per Cell', shrink=0.7)
    ax.set_xlabel('PC1 Bin', fontsize=12)
    ax.set_ylabel('PC2 Bin', fontsize=12)
    ax.set_title(f'Chemical Space Grid Heatmap\n(Richness: {richness}/{total_cells} cells = {richness_pct:.1f}%)',
                 fontsize=13, fontweight='bold')

    # 右图: 指标汇总
    ax = axes[1]
    ax.axis('off')
    metrics_text = (
        f"Chemical Space Coverage Metrics\n"
        f"{'='*50}\n\n"
        f"  Richness (丰富度):\n"
        f"    Occupied cells: {richness} / {total_cells}\n"
        f"    Coverage: {richness_pct:.1f}%\n\n"
        f"  Uniformity (均匀度):\n"
        f"    Shannon Entropy: {shannon:.4f}\n"
        f"    Max Entropy: {max_shannon:.4f}\n"
        f"    Uniformity Index: {uniformity:.4f}\n"
        f"    (1.0 = perfectly uniform)\n\n"
        f"  Extent (广度):\n"
        f"    Max pairwise distance: {extent:.4f}\n"
        f"    Mean pairwise distance: {np.mean(distances):.4f}\n"
        f"    (PCA space, {len(X_pca)} molecules)\n\n"
        f"  Total molecules: {len(X_pca)}\n"
        f"  PCA explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Chemical Space Coverage: Richness, Uniformity, Extent', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '16_coverage_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图16: 覆盖度三要素 ✓")


def plot_tmap(fps, props_df, save_dir):
    """图17: TMAP 树状化学空间图"""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图17: 指纹数据不足, 跳过")
        return

    try:
        import tmap as tmap_lib
        from faerun import Faerun
    except ImportError:
        print("  tmap/faerun 未安装, 使用替代可视化")
        plot_tmap_alternative(fps, props_df, save_dir)
        return

    print("  生成 TMAP 树状图...")

    # 提取指纹
    fp_arrays = []
    for i in valid_idx:
        arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fps[i], arr)
        fp_arrays.append(arr)

    # 使用 MinHash
    from tmap.vector import Minhash
    from tmap.layout import Layout, MSPD

    # 生成 MinHash
    enc = Minhash(perm=128, seed=42)
    lf = Layout(enc)

    # 对指纹进行 MinHash 编码
    data = np.array(fp_arrays, dtype=np.uint32)
    # 将二值指纹转换为稀疏表示
    sparse_data = []
    for arr in fp_arrays:
        sparse_data.append(np.where(arr > 0)[0].tolist())

    x, y, s, t, _ = lf.layout_from_sparse_binary_list(sparse_data)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 12))
    valid_props = props_df.iloc[valid_idx].reset_index(drop=True)

    sc = ax.scatter(x, y, c=valid_props['n_aromatic_rings'], cmap='Set1',
                    alpha=0.5, s=8, edgecolors='none')

    # 绘制树边
    for i in range(len(s)):
        ax.plot([x[s[i]], x[t[i]]], [y[s[i]], y[t[i]]], 
                'gray', alpha=0.05, linewidth=0.3)

    plt.colorbar(sc, ax=ax, label='# Aromatic Rings', shrink=0.6)
    ax.set_xlabel('TMAP X', fontsize=12)
    ax.set_ylabel('TMAP Y', fontsize=12)
    ax.set_title('TMAP Tree-based Chemical Space Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '17_tmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图17: TMAP 树状图 ✓")


def plot_tmap_alternative(fps, props_df, save_dir):
    """TMAP 替代方案: 层次聚类树状图"""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform

    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    
    # 采样以加速 (最多 500 个分子)
    if len(valid_idx) > 500:
        np.random.seed(42)
        valid_idx = np.random.choice(valid_idx, 500, replace=False).tolist()

    # 计算 Tanimoto 距离矩阵
    print("  计算距离矩阵 (采样 {} 分子)...".format(len(valid_idx)))
    n = len(valid_idx)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[valid_idx[i]], fps[valid_idx[j]])
            dist_matrix[i, j] = 1 - sim
            dist_matrix[j, i] = 1 - sim

    # 层次聚类
    print("  层次聚类...")
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')

    # 绘制树状图
    fig, ax = plt.subplots(figsize=(20, 8))
    valid_props = props_df.iloc[valid_idx].reset_index(drop=True)
    dendrogram(Z, no_plot=True, color_threshold=0.7*max(Z[:,2]))
    
    # 使用 scipy 的 dendrogram
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(Z, t=0.7*max(Z[:,2]), criterion='distance')
    
    ddata = dendrogram(Z, color_threshold=0.7*max(Z[:,2]), 
                       above_threshold_color='gray', ax=ax)
    
    ax.set_xlabel('Molecule Index', fontsize=12)
    ax.set_ylabel('Tanimoto Distance (1 - T)', fontsize=12)
    ax.set_title(f'Hierarchical Clustering Dendrogram (TMAP Alternative)\n'
                 f'{n} molecules sampled, {len(set(clusters))} clusters',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '17_tmap_dendrogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图17: 层次聚类树状图 (TMAP替代) ✓")


def main():
    print("="*60)
    print("高级化学空间分析 (Part 2)")
    print("="*60)

    props_df, fps, df = load_and_compute()

    print("\n生成图表:")
    plot_pmi_fixed(props_df, OUTPUT_DIR)
    plot_bertz(props_df, OUTPUT_DIR)
    plot_functional_groups(props_df, OUTPUT_DIR)
    plot_ring_system_types(props_df, OUTPUT_DIR)
    plot_scaffold_hierarchy(props_df, OUTPUT_DIR)
    plot_nn_tanimoto(fps, OUTPUT_DIR)
    plot_coverage_metrics(fps, OUTPUT_DIR)
    plot_tmap(fps, props_df, OUTPUT_DIR)

    print(f"\n所有图表保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
