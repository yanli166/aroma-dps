"""
数据集化学空间覆盖范围分析
生成期刊中常见的多种数据集描述图表

包含:
1. 分子量/原子数/LogP等基本性质分布
2. 主惯性矩 (PMI) 图 - 分子形状分析
3. 化学空间 PCA/t-SNE 可视化
4. Tanimoto 相似性分布 - 分子多样性
5. 环大小/数量分布
6. HOMA 值分布
7. 骨架多样性分析
8. Fsp3 / QED / Lipinski 性质
9. 芳香性 vs HOMA 关系
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PROJ_ROOT = '_PROJ_ROOT'
DATASET_PATH = os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv')
OUTPUT_DIR = os.path.join(PROJ_ROOT, '0427_unified_results', 'chemical_space')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_molecular_properties(df):
    """计算所有分子的描述符"""
    print("计算分子描述符 (这可能需要几分钟)...")
    
    props = []
    fps = []
    unique_smiles = df['smiles'].unique()
    print(f"唯一分子数: {len(unique_smiles)} (总行数: {len(df)})")
    
    for idx, smiles in enumerate(unique_smiles):
        if idx % 500 == 0:
            print(f"  进度: {idx}/{len(unique_smiles)}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # 基本描述符
        mw = Descriptors.MolWt(mol)
        n_atoms = mol.GetNumAtoms()
        n_heavy = mol.GetNumHeavyAtoms()
        n_bonds = mol.GetNumBonds()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
        n_hetero = Descriptors.NumHeteroatoms(mol)
        n_rotatable = Descriptors.NumRotatableBonds(mol)
        n_hbd = Descriptors.NumHDonors(mol)
        n_hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        mr = Descriptors.MolMR(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        qed = Descriptors.qed(mol)
        
        # 主惯性矩 (需要3D构型)
        pmis = [0, 0, 0]
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == 0:
                AllChem.MMFFOptimizeMolecule(mol_h)
                pmis = [
                    rdMolDescriptors.CalcPMI1(mol_h),
                    rdMolDescriptors.CalcPMI2(mol_h),
                    rdMolDescriptors.CalcPMI3(mol_h),
                ]
        except Exception:
            pass
        
        # Murcko 骨架
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except Exception:
            scaffold_smiles = ''
        
        # 分子指纹 (Morgan fingerprint)
        try:
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = fp_gen.GetFingerprint(mol)
            fps.append(fp)
        except Exception:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
            except Exception:
                fps.append(None)
        
        # 环大小统计 (该分子所有环)
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        
        props.append({
            'smiles': smiles,
            'mw': mw,
            'n_atoms': n_atoms,
            'n_heavy': n_heavy,
            'n_bonds': n_bonds,
            'n_rings': n_rings,
            'n_aromatic_rings': n_aromatic,
            'n_aliphatic_rings': n_aliphatic,
            'n_heteroatoms': n_hetero,
            'n_rotatable': n_rotatable,
            'n_hbd': n_hbd,
            'n_hba': n_hba,
            'tpsa': tpsa,
            'logp': logp,
            'mr': mr,
            'fsp3': fsp3,
            'qed': qed,
            'pmi1': pmis[0],
            'pmi2': pmis[1],
            'pmi3': pmis[2],
            'n_sized_rings': len(ring_sizes),
            'min_ring_size': min(ring_sizes) if ring_sizes else 0,
            'max_ring_size': max(ring_sizes) if ring_sizes else 0,
            'scaffold': scaffold_smiles,
        })
    
    props_df = pd.DataFrame(props)
    print(f"成功计算 {len(props_df)} 个分子的描述符")
    return props_df, fps


def plot_basic_distributions(props_df, df, save_dir):
    """图1: 基本性质分布 (4子图)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # MW
    ax = axes[0][0]
    ax.hist(props_df['mw'], bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['mw'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["mw"].mean():.1f}')
    ax.axvline(500, color='green', linestyle=':', linewidth=2, label='Lipinski: 500')
    ax.set_xlabel('Molecular Weight (Da)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Molecular Weight Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # LogP
    ax = axes[0][1]
    ax.hist(props_df['logp'], bins=50, color='#DD8452', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['logp'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["logp"].mean():.2f}')
    ax.axvline(5, color='green', linestyle=':', linewidth=2, label='Lipinski: 5')
    ax.set_xlabel('cLogP', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Lipophilicity (cLogP) Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Heavy atoms
    ax = axes[0][2]
    ax.hist(props_df['n_heavy'], bins=50, color='#55A868', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['n_heavy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["n_heavy"].mean():.1f}')
    ax.set_xlabel('Number of Heavy Atoms', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Heavy Atom Count Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # HBA
    ax = axes[1][0]
    ax.hist(props_df['n_hba'], bins=30, color='#C44E52', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['n_hba'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["n_hba"].mean():.1f}')
    ax.axvline(10, color='green', linestyle=':', linewidth=2, label='Lipinski: 10')
    ax.set_xlabel('H-Bond Acceptors', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('H-Bond Acceptors Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # HBD
    ax = axes[1][1]
    ax.hist(props_df['n_hbd'], bins=20, color='#8172B3', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['n_hbd'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["n_hbd"].mean():.1f}')
    ax.axvline(5, color='green', linestyle=':', linewidth=2, label='Lipinski: 5')
    ax.set_xlabel('H-Bond Donors', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('H-Bond Donors Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # TPSA
    ax = axes[1][2]
    ax.hist(props_df['tpsa'], bins=50, color='#937860', edgecolor='white', alpha=0.8)
    ax.axvline(props_df['tpsa'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={props_df["tpsa"].mean():.1f}')
    ax.axvline(140, color='green', linestyle=':', linewidth=2, label='Veber: 140')
    ax.set_xlabel('TPSA (Å²)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Topological Polar Surface Area', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dataset Basic Molecular Properties', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_basic_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图1: 基本性质分布 ✓")


def plot_pmi(props_df, save_dir):
    """图2: 主惯性矩 (PMI) 图 - 分子形状分析"""
    valid = props_df[(props_df['pmi1'] > 0) & (props_df['pmi2'] > 0) & (props_df['pmi3'] > 0)].copy()
    
    if len(valid) == 0:
        print("  图2: PMI 数据不足, 跳过")
        return
    
    # 归一化: n1=pmi1/pmi3, n2=pmi2/pmi3
    valid['n1'] = valid['pmi1'] / valid['pmi3']
    valid['n2'] = valid['pmi2'] / valid['pmi3']
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # 绘制参考三角形
    # 三个顶点: (0,0)=球, (1,1)=盘, (1,0)=棒
    triangle_x = [0, 1, 0]
    triangle_y = [0, 1, 1]
    ax.fill(triangle_x, triangle_y, alpha=0.05, color='blue')
    ax.plot([0, 1, 0, 0], [0, 1, 1, 0], 'k-', linewidth=1.5)
    
    # 标注区域
    ax.text(0.05, 0.35, 'Sphere\n(3D)', fontsize=13, ha='center', style='italic', color='blue')
    ax.text(0.65, 0.85, 'Disc\n(2D)', fontsize=13, ha='center', style='italic', color='green')
    ax.text(0.65, 0.15, 'Rod\n(1D)', fontsize=13, ha='center', style='italic', color='red')
    
    # 按 HOMA 值着色
    sc = ax.scatter(valid['n1'], valid['n2'], c=valid['mw'], cmap='viridis', 
                    alpha=0.5, s=10, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Molecular Weight (Da)', shrink=0.7)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('n1 (I₁/I₃)', fontsize=13)
    ax.set_ylabel('n2 (I₂/I₃)', fontsize=13)
    ax.set_title('Principal Moments of Inertia (PMI)\nMolecular Shape Analysis', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_pmi_shape.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图2: 主惯性矩 (PMI) 形状图 ✓")


def plot_chemical_space_pca(props_df, fps, save_dir):
    """图3: 化学空间 PCA 可视化 (Morgan指纹)"""
    # 提取有效指纹
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图3: 指纹数据不足, 跳过")
        return
    
    # 转换指纹为numpy数组
    fp_arrays = []
    for i in valid_idx:
        fp = fps[i]
        arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_arrays.append(arr)
    
    X = np.array(fp_arrays)
    valid_props = props_df.iloc[valid_idx].reset_index(drop=True)
    
    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 按分子量着色
    ax = axes[0]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=valid_props['mw'], cmap='viridis',
                    alpha=0.5, s=8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Molecular Weight', shrink=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Chemical Space - PCA (colored by MW)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # 按 LogP 着色
    ax = axes[1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=valid_props['logp'], cmap='coolwarm',
                    alpha=0.5, s=8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='cLogP', shrink=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Chemical Space - PCA (colored by LogP)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.suptitle('Chemical Space Coverage (Morgan Fingerprint PCA)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_chemical_space_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图3: 化学空间 PCA ✓")


def plot_chemical_space_tsne(props_df, fps, save_dir):
    """图4: 化学空间 t-SNE 可视化"""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图4: 指纹数据不足, 跳过")
        return
    
    fp_arrays = []
    for i in valid_idx:
        fp = fps[i]
        arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_arrays.append(arr)
    
    X = np.array(fp_arrays)
    valid_props = props_df.iloc[valid_idx].reset_index(drop=True)
    
    # 先 PCA 降维到 50 维, 再 t-SNE 到 2 维
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]-1))
    X_pca = pca.fit_transform(X)
    
    print("  运行 t-SNE (可能需要几分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 按芳香环数着色
    ax = axes[0]
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=valid_props['n_aromatic_rings'],
                    cmap='Set1', alpha=0.5, s=8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='# Aromatic Rings', shrink=0.7)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Chemical Space - t-SNE (colored by Aromatic Rings)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # 按 Fsp3 着色
    ax = axes[1]
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=valid_props['fsp3'],
                    cmap='RdYlBu', alpha=0.5, s=8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Fsp3', shrink=0.7)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Chemical Space - t-SNE (colored by Fsp3)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.suptitle('Chemical Space Coverage (Morgan Fingerprint t-SNE)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_chemical_space_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图4: 化学空间 t-SNE ✓")


def plot_ring_analysis(df, props_df, save_dir):
    """图5: 环信息分析"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 环大小分布 (数据集的 target ring) - 柱状图版本保留
    ax = axes[0][0]
    ring_size_counts = df['Ring_Size'].value_counts().sort_index()
    colors = cm.Set2(np.linspace(0, 1, len(ring_size_counts)))
    ax.bar(ring_size_counts.index, ring_size_counts.values, color=colors, edgecolor='white')
    for i, (size, count) in enumerate(ring_size_counts.items()):
        ax.text(size, count + 20, str(count), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Target Ring Size', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Ring Size Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 每分子的环数分布 - 柱状图版本保留
    ax = axes[0][1]
    ax.hist(props_df['n_rings'], bins=range(0, int(props_df['n_rings'].max())+2),
            color='#4C72B0', edgecolor='white', alpha=0.8, rwidth=0.85)
    ax.set_xlabel('Number of Rings per Molecule', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Total Ring Count Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 芳香环 vs 脂肪族环
    ax = axes[1][0]
    x = np.arange(len(props_df))
    ax.hist(props_df['n_aromatic_rings'], bins=range(0, 10), alpha=0.6, 
            label='Aromatic Rings', color='#DD8452', edgecolor='white')
    ax.hist(props_df['n_aliphatic_rings'], bins=range(0, 10), alpha=0.6,
            label='Aliphatic Rings', color='#55A868', edgecolor='white')
    ax.set_xlabel('Number of Rings', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Aromatic vs Aliphatic Rings', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # HOMA 按环大小分布
    ax = axes[1][1]
    ring_sizes = sorted(df['Ring_Size'].unique())
    homa_by_size = [df[df['Ring_Size']==s]['homa_value'].values for s in ring_sizes]
    bp = ax.boxplot(homa_by_size, labels=[str(s) for s in ring_sizes], 
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], cm.Set3(np.linspace(0, 1, len(ring_sizes)))):
        patch.set_facecolor(color)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Ring Size', fontsize=12)
    ax.set_ylabel('HOMA Value', fontsize=12)
    ax.set_title('HOMA Distribution by Ring Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Ring Information Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_ring_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图5: 环信息分析 ✓")
    
    # 额外生成饼状图版本（合并图）
    plot_ring_distribution_pie(df, props_df, save_dir)


def plot_ring_distribution_pie(df, props_df, save_dir):
    """图5b: 环分布饼状图（合并版本）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: Target Ring Size Distribution (饼状图)
    ax = axes[0]
    ring_size_counts = df['Ring_Size'].value_counts().sort_index()
    colors1 = cm.Set3(np.linspace(0, 1, len(ring_size_counts)))
    
    # 准备标签：显示环大小和数量
    labels1 = [f'{size}-ring\n({count})' for size, count in ring_size_counts.items()]
    
    wedges, texts, autotexts = ax.pie(ring_size_counts.values, 
                                       labels=labels1,
                                       autopct='%1.1f%%',
                                       colors=colors1,
                                       startangle=90,
                                       textprops={'fontsize': 11})
    
    # 设置百分比字体
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Target Ring Size Distribution\n(Pie Chart)', fontsize=14, fontweight='bold')
    
    # 右图: Total Ring Count Distribution (饼状图)
    ax = axes[1]
    ring_count_distribution = props_df['n_rings'].value_counts().sort_index()
    colors2 = cm.Paired(np.linspace(0, 1, len(ring_count_distribution)))
    
    # 准备标签：显示环数和分子数
    labels2 = [f'{count} rings\n({num} mols)' for count, num in ring_count_distribution.items()]
    
    wedges2, texts2, autotexts2 = ax.pie(ring_count_distribution.values,
                                          labels=labels2,
                                          autopct='%1.1f%%',
                                          colors=colors2,
                                          startangle=90,
                                          textprops={'fontsize': 11})
    
    for autotext in autotexts2:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Total Ring Count Distribution\n(Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Ring Distribution Overview (Pie Charts)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05b_ring_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图5b: 环分布饼状图 ✓")


def plot_homa_distribution(df, save_dir):
    """图6: HOMA 值分布"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # HOMA 直方图
    ax = axes[0]
    ax.hist(df['homa_value'], bins=80, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(df['homa_value'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={df["homa_value"].mean():.2f}')
    ax.axvline(df['homa_value'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median={df["homa_value"].median():.2f}')
    ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='HOMA=0.5 (borderline)')
    ax.set_xlabel('HOMA Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('HOMA Value Distribution (Target Variable)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # HOMA vs NICS
    ax = axes[1]
    sc = ax.scatter(df['Ring_NICS_ZZ_1'], df['homa_value'], 
                    c=df['Ring_Size'], cmap='Set1', alpha=0.4, s=8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Ring Size', shrink=0.7)
    ax.set_xlabel('Ring NICS-ZZ (ppm)', fontsize=12)
    ax.set_ylabel('HOMA Value', fontsize=12)
    ax.set_title('HOMA vs NICS-ZZ Correlation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 计算相关系数
    corr = df['Ring_NICS_ZZ_1'].corr(df['homa_value'])
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Target Variable (HOMA) Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '06_homa_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图6: HOMA 分布 ✓")


def plot_lipinski_qed(props_df, save_dir):
    """图7: Lipinski 五规则和 QED 药物相似性"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Lipinski 违反数
    ax = axes[0]
    violations = []
    for _, row in props_df.iterrows():
        v = 0
        if row['mw'] > 500: v += 1
        if row['logp'] > 5: v += 1
        if row['n_hbd'] > 5: v += 1
        if row['n_hba'] > 10: v += 1
        violations.append(v)
    v_counts = pd.Series(violations).value_counts().sort_index()
    colors = ['#55A868', '#DD8452', '#C44E52', '#8C564B']
    ax.bar(v_counts.index, v_counts.values, color=colors[:len(v_counts)], edgecolor='white')
    for i, (v, c) in enumerate(v_counts.items()):
        pct = c / len(props_df) * 100
        ax.text(v, c + 10, f'{c}\n({pct:.1f}%)', ha='center', fontsize=10)
    ax.set_xlabel('Number of Violations', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Lipinski Rule of Five Violations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # QED 分布
    ax = axes[1]
    ax.hist(props_df['qed'], bins=50, color='#8172B3', edgecolor='white', alpha=0.8)
    ax.axvline(0.7, color='green', linestyle='--', linewidth=2, label='QED>0.7 (good)')
    ax.axvline(props_df['qed'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={props_df["qed"].mean():.3f}')
    ax.set_xlabel('QED Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('QED Drug-likeness Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Fsp3 分布
    ax = axes[2]
    ax.hist(props_df['fsp3'], bins=50, color='#937860', edgecolor='white', alpha=0.8)
    ax.axvline(0.47, color='green', linestyle='--', linewidth=2, label='Fsp3>0.47 (complex)')
    ax.axvline(props_df['fsp3'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={props_df["fsp3"].mean():.3f}')
    ax.set_xlabel('Fsp3 (Fraction sp³)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Carbon Saturation (Fsp3)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Drug-likeness and Molecular Complexity', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '07_druglikeness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图7: 药物相似性 ✓")


def plot_scaffold_diversity(props_df, save_dir):
    """图8: Murcko 骨架多样性分析"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 骨架数量统计
    n_unique_scaffolds = props_df['scaffold'].nunique()
    n_molecules = len(props_df)
    
    ax = axes[0]
    scaffold_counts = props_df['scaffold'].value_counts()
    top_scaffolds = scaffold_counts.head(20)
    ax.barh(range(len(top_scaffolds)), top_scaffolds.values, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(top_scaffolds)))
    ax.set_yticklabels([s[:30]+'...' if len(s)>30 else s for s in top_scaffolds.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'Top 20 Murcko Scaffolds\n({n_unique_scaffolds} unique scaffolds / {n_molecules} molecules)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 骨架频率分布
    ax = axes[1]
    scaffold_freq = scaffold_counts.values
    ax.hist(scaffold_freq, bins=50, color='#55A868', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Molecules per Scaffold', fontsize=12)
    ax.set_ylabel('Number of Scaffolds', fontsize=12)
    ax.set_title('Scaffold Frequency Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 计算骨架覆盖率
    coverage = n_unique_scaffolds / n_molecules * 100
    ax.text(0.95, 0.95, f'Scaffold Diversity: {coverage:.1f}%\n({n_unique_scaffolds}/{n_molecules})',
            transform=ax.transAxes, fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Murcko Scaffold Diversity Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '08_scaffold_diversity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图8: 骨架多样性 ✓")


def plot_tanimoto_similarity(fps, save_dir):
    """图9: Tanimoto 相似性分布"""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        print("  图9: 指纹数据不足, 跳过")
        return
    
    # 随机采样 2000 对计算相似性
    np.random.seed(42)
    n_pairs = min(50000, len(valid_idx) * 10)
    similarities = []
    
    for _ in range(n_pairs):
        i, j = np.random.choice(len(valid_idx), 2, replace=False)
        fp1 = fps[valid_idx[i]]
        fp2 = fps[valid_idx[j]]
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(sim)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=100, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(similarities):.3f}')
    ax.axvline(np.median(similarities), color='green', linestyle='--', linewidth=2,
               label=f'Median={np.median(similarities):.3f}')
    ax.axvline(0.85, color='orange', linestyle=':', linewidth=2, label='T=0.85 (very similar)')
    ax.set_xlabel('Tanimoto Similarity', fontsize=13)
    ax.set_ylabel('Pair Count', fontsize=13)
    ax.set_title(f'Tanimoto Similarity Distribution\n({n_pairs} random pairs)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '09_tanimoto_similarity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图9: Tanimoto 相似性 ✓")


def plot_mw_logp_scatter(props_df, save_dir):
    """图10: MW vs LogP 散点图 (药物化学空间)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc = ax.scatter(props_df['logp'], props_df['mw'], 
                    c=props_df['n_aromatic_rings'], cmap='YlOrRd',
                    alpha=0.5, s=10, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='# Aromatic Rings', shrink=0.7)
    
    # Lipinski 边界
    ax.axvline(5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='LogP=5')
    ax.axhline(500, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='MW=500')
    
    # 标注区域
    ax.text(2, 600, 'Beyond Ro5', fontsize=12, color='red', alpha=0.5, ha='center')
    ax.text(-2, 200, 'Drug-like', fontsize=12, color='green', alpha=0.5, ha='center')
    
    ax.set_xlabel('cLogP', fontsize=13)
    ax.set_ylabel('Molecular Weight (Da)', fontsize=13)
    ax.set_title('Chemical Space: MW vs LogP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '10_mw_vs_logp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图10: MW vs LogP 散点图 ✓")


def plot_summary_stats(props_df, df, save_dir):
    """生成统计摘要表"""
    stats = {
        '总分子数': len(props_df),
        '总数据行数 (环)': len(df),
        '每分子平均环数': len(df) / len(props_df),
        '唯一 Murcko 骨架数': props_df['scaffold'].nunique(),
        '骨架多样性 (%)': props_df['scaffold'].nunique() / len(props_df) * 100,
        '平均分子量': props_df['mw'].mean(),
        '平均 LogP': props_df['logp'].mean(),
        '平均原子数': props_df['n_atoms'].mean(),
        '平均重原子数': props_df['n_heavy'].mean(),
        '平均环数': props_df['n_rings'].mean(),
        '平均芳香环数': props_df['n_aromatic_rings'].mean(),
        '平均 TPSA': props_df['tpsa'].mean(),
        '平均 Fsp3': props_df['fsp3'].mean(),
        '平均 QED': props_df['qed'].mean(),
        'Lipinski 合规率 (%)': sum((props_df['mw']<=500) & (props_df['logp']<=5) & 
                                    (props_df['n_hbd']<=5) & (props_df['n_hba']<=10)) / len(props_df) * 100,
        'HOMA 均值': df['homa_value'].mean(),
        'HOMA 标准差': df['homa_value'].std(),
        'HOMA 最小值': df['homa_value'].min(),
        'HOMA 最大值': df['homa_value'].max(),
        '环大小范围': f"{df['Ring_Size'].min()}-{df['Ring_Size'].max()}",
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['指标', '值'])
    stats_df.to_csv(os.path.join(save_dir, 'dataset_statistics.csv'), index=False)
    
    print("\n" + "="*50)
    print("数据集统计摘要")
    print("="*50)
    for _, row in stats_df.iterrows():
        if isinstance(row['值'], float):
            print(f"  {row['指标']}: {row['值']:.4f}")
        else:
            print(f"  {row['指标']}: {row['值']}")
    
    return stats_df


def main():
    print("="*60)
    print("数据集化学空间覆盖范围分析")
    print("="*60)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"数据集: {len(df)} 行, {df['smiles'].nunique()} 个唯一分子")
    
    # 计算分子属性
    props_df, fps = compute_molecular_properties(df)
    
    # 生成所有图表
    print("\n生成图表:")
    plot_basic_distributions(props_df, df, OUTPUT_DIR)
    plot_pmi(props_df, OUTPUT_DIR)
    plot_chemical_space_pca(props_df, fps, OUTPUT_DIR)
    plot_chemical_space_tsne(props_df, fps, OUTPUT_DIR)
    plot_ring_analysis(df, props_df, OUTPUT_DIR)
    plot_homa_distribution(df, OUTPUT_DIR)
    plot_lipinski_qed(props_df, OUTPUT_DIR)
    plot_scaffold_diversity(props_df, OUTPUT_DIR)
    plot_tanimoto_similarity(fps, OUTPUT_DIR)
    plot_mw_logp_scatter(props_df, OUTPUT_DIR)
    
    # 统计摘要
    plot_summary_stats(props_df, df, OUTPUT_DIR)
    
    print(f"\n所有图表保存到: {OUTPUT_DIR}")
    print(f"共生成 10 张分析图 + 1 个统计表")


if __name__ == '__main__':
    main()
