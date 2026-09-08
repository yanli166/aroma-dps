"""
分析数据集中芳环-取代基覆盖情况
判断是否需要补充更多不同位点/不同取代基的常见芳环
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from collections import Counter, defaultdict
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

PROJ_ROOT = _PROJ_ROOT
DATASET_PATH = os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv')
OUTPUT_DIR = os.path.join(PROJ_ROOT, '0427_unified_results', 'chemical_space')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_ring_substituent_coverage():
    """分析环-取代基覆盖"""
    df = pd.read_csv(DATASET_PATH)
    print(f"数据集: {len(df)} 行, {df['smiles'].nunique()} 个唯一分子")

    # 收集所有目标环的信息
    ring_info_list = []
    print("分析每个目标环的取代基模式...")

    for idx, row in df.iterrows():
        smiles = row['smiles']
        atom_on_ring = eval(row['atom_on_ring']) if isinstance(row['atom_on_ring'], str) else row['atom_on_ring']
        ring_size = row['Ring_Size']
        homa = row['homa_value']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # atom_on_ring is 0-based (verified by P0-2 audit across 20,605 rows)
        ring_atom_indices = [int(i) for i in atom_on_ring if isinstance(i, (int, float)) and 0 <= i < mol.GetNumAtoms()]
        if len(ring_atom_indices) == 0:
            continue

        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring_atom_indices]
        n_ring_atoms = len(ring_atoms)

        # 环上原子类型
        ring_atom_types = tuple(sorted([a.GetSymbol() for a in ring_atoms]))
        ring_composition = ''.join(ring_atom_types)

        # 环上芳香性
        is_aromatic = all(a.GetIsAromatic() for a in ring_atoms)

        # 取代基分析: 每个环上原子的非环取代基
        substituents = []
        substituent_positions = []  # 取代基在环上的位置 (0-indexed)
        for pos, atom in enumerate(ring_atoms):
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_atom_indices:
                    # 取代基类型
                    sub_type = classify_substituent(mol, atom.GetIdx(), neighbor.GetIdx())
                    substituents.append(sub_type)
                    substituent_positions.append(pos)

        # 取代基数量
        n_substituents = len(substituents)

        # 取代基模式 (sorted tuple of substituent types)
        sub_pattern = tuple(sorted(substituents)) if substituents else ('H',)

        # 取代基位置模式 (哪些位置有取代基)
        pos_pattern = tuple(sorted(set(substituent_positions)))

        ring_info_list.append({
            'smiles': smiles,
            'ring_size': ring_size,
            'ring_composition': ring_composition,
            'is_aromatic': is_aromatic,
            'n_substituents': n_substituents,
            'substituents': substituents,
            'sub_pattern': sub_pattern,
            'pos_pattern': pos_pattern,
            'homa': homa,
            'ring_atom_types': ring_atom_types,
        })

    return pd.DataFrame(ring_info_list)


def classify_substituent(mol, ring_atom_idx, sub_atom_idx):
    """分类取代基类型"""
    sub_atom = mol.GetAtomWithIdx(sub_atom_idx)
    ring_atom = mol.GetAtomWithIdx(ring_atom_idx)

    symbol = sub_atom.GetSymbol()
    bond = mol.GetBondBetweenAtoms(ring_atom_idx, sub_atom_idx)
    bond_type = bond.GetBondType()

    # 检查是否是简单氢 (不应该出现, AddHs 未调用)
    if symbol == 'H':
        return 'H'

    # 检查是否在另一个环中 (稠环/并环)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if sub_atom_idx in ring and ring_atom_idx not in ring:
            return 'fused_ring'

    # 简单卤素
    if symbol in ['F', 'Cl', 'Br', 'I']:
        return f'X({symbol})'

    # 羟基 -OH
    if symbol == 'O':
        if sub_atom.GetTotalNumHs() == 1 and len(sub_atom.GetNeighbors()) == 1:
            return 'OH'
        # 醚 -O-R
        return 'OR'

    # 硫醇/硫醚
    if symbol == 'S':
        if sub_atom.GetTotalNumHs() == 1 and len(sub_atom.GetNeighbors()) == 1:
            return 'SH'
        return 'SR'

    # 氨基
    if symbol == 'N':
        n_h = sub_atom.GetTotalNumHs()
        n_neighbors = len(sub_atom.GetNeighbors())
        if n_h == 2:
            return 'NH2'
        elif n_h == 1:
            return 'NHR'
        else:
            return 'NR2'

    # 羰基相关 (C=O 连接到环)
    if symbol == 'C' and bond_type == Chem.BondType.DOUBLE:
        # 检查 C 是否连有 O
        for n in sub_atom.GetNeighbors():
            if n.GetSymbol() == 'O' and n.GetIdx() != ring_atom_idx:
                return 'C=O(carbonyl)'

    # 检查是否含羧基/酯/酰胺 (更复杂的取代基)
    if symbol == 'C':
        # 递归检查取代基组成
        sub_smiles = get_substituent_smiles(mol, ring_atom_idx, sub_atom_idx)
        if sub_smiles:
            if 'C(=O)O' in sub_smiles and 'C(=O)[O;H]' in sub_smiles:
                return 'COOH'
            if 'C(=O)O' in sub_smiles:
                return 'COOR'
            if 'C(=O)N' in sub_smiles:
                return 'CONH'
            if 'C=O' in sub_smiles or 'C(=O)' in sub_smiles:
                return 'C=O'
            if 'C#' in sub_smiles and 'N' in sub_smiles:
                return 'C≡N'
            if 'NO2' in sub_smiles:
                return 'NO2'

        # 简单烷基
        if sub_atom.GetTotalNumHs() > 0:
            return 'alkyl'
        return 'aryl/R'

    return f'other({symbol})'


def get_substituent_smiles(mol, ring_atom_idx, sub_atom_idx):
    """获取取代基的 SMILES (简化版)"""
    try:
        # 使用 RDKit 的子结构提取
        amap = {}
        sub_atoms = [sub_atom_idx]

        # BFS 获取所有连接的原子 (不经过环原子)
        visited = {sub_atom_idx}
        queue = [sub_atom_idx]
        ring_atom_set = set()
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if ring_atom_idx in ring:
                ring_atom_set.update(ring)

        while queue:
            current = queue.pop(0)
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nid = neighbor.GetIdx()
                if nid not in visited and nid not in ring_atom_set:
                    visited.add(nid)
                    sub_atoms.append(nid)
                    queue.append(nid)

        # 创建子分子的 SMILES
        sub_mol = Chem.PathToSubmol(mol, sub_atoms)
        return Chem.MolToSmiles(sub_mol)
    except Exception:
        return ''


def plot_coverage_analysis(ring_df, save_dir):
    """生成覆盖分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ===== 1. 环骨架类型分布 =====
    ax = axes[0][0]
    # 提取环骨架 (去掉取代基, 只看环上原子组成)
    ring_skeletons = ring_df['ring_composition'].value_counts().head(15)
    colors = cm.Set3(np.linspace(0, 1, len(ring_skeletons)))
    ax.barh(range(len(ring_skeletons)), ring_skeletons.values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(ring_skeletons)))
    ax.set_yticklabels(ring_skeletons.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Top 15 Ring Skeletons\n(by atom composition)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(ring_skeletons.values):
        ax.text(v + 10, i, str(v), va='center', fontsize=8)

    # ===== 2. 取代基类型频次 =====
    ax = axes[0][1]
    all_subs = []
    for subs in ring_df['substituents']:
        all_subs.extend(subs)
    sub_counts = Counter(all_subs)
    top_subs = sub_counts.most_common(20)
    names = [x[0] for x in top_subs]
    counts = [x[1] for x in top_subs]
    colors2 = cm.tab20(np.linspace(0, 1, len(names)))
    ax.barh(range(len(names)), counts, color=colors2, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_title('Top 20 Substituent Types', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(counts):
        ax.text(v + 5, i, str(v), va='center', fontsize=8)

    # ===== 3. 取代基数量分布 =====
    ax = axes[0][2]
    sub_counts_by_ring = ring_df['n_substituents'].value_counts().sort_index()
    ax.bar(sub_counts_by_ring.index, sub_counts_by_ring.values, color='#4C72B0', edgecolor='white')
    ax.set_xlabel('Number of Substituents on Ring', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Substituent Count Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in zip(sub_counts_by_ring.index, sub_counts_by_ring.values):
        ax.text(i, v + 20, str(v), ha='center', fontsize=10, fontweight='bold')

    # ===== 4. 取代基位置模式热力图 (6元环) =====
    ax = axes[1][0]
    six_ring = ring_df[ring_df['ring_size'] == 6].copy()
    if len(six_ring) > 0:
        # 位置模式: 哪些位置有取代基
        pos_patterns = six_ring['pos_pattern'].value_counts().head(15)
        # 转换为可读的位置标记
        pos_labels = []
        for p in pos_patterns.index:
            if len(p) == 0:
                pos_labels.append('unsubstituted')
            else:
                pos_labels.append(','.join([str(x) for x in p]))

        colors3 = cm.viridis(np.linspace(0.2, 0.8, len(pos_patterns)))
        ax.barh(range(len(pos_patterns)), pos_patterns.values, color=colors3, edgecolor='white')
        ax.set_yticks(range(len(pos_patterns)))
        ax.set_yticklabels(pos_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title('Substitution Position Patterns (6-ring)\n(0-indexed positions)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No 6-rings', ha='center', va='center', fontsize=14)
        ax.set_title('Substitution Patterns (6-ring)', fontsize=12)

    # ===== 5. 环骨架 × 取代基模式 矩阵 =====
    ax = axes[1][1]
    # Top 10 环骨架 × Top 8 取代基
    top_skeletons = ring_df['ring_composition'].value_counts().head(10).index
    top_sub_types = [x[0] for x in Counter(all_subs).most_common(8)]

    matrix = np.zeros((len(top_skeletons), len(top_sub_types)))
    for i, skel in enumerate(top_skeletons):
        skel_rows = ring_df[ring_df['ring_composition'] == skel]
        for j, sub_type in enumerate(top_sub_types):
            count = sum(1 for subs in skel_rows['substituents'] if sub_type in subs)
            matrix[i, j] = count

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(len(top_skeletons)))
    ax.set_yticklabels(top_skeletons, fontsize=8)
    ax.set_xticks(range(len(top_sub_types)))
    ax.set_xticklabels(top_sub_types, rotation=45, ha='right', fontsize=8)
    ax.set_title('Ring Skeleton × Substituent Matrix\n(count of molecules)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
    # 标注数值
    for i in range(len(top_skeletons)):
        for j in range(len(top_sub_types)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if matrix[i, j] > matrix.max()*0.6 else 'black')

    # ===== 6. HOMA 按取代基数量分布 =====
    ax = axes[1][2]
    sub_groups = sorted(ring_df['n_substituents'].unique())
    homa_by_sub = [ring_df[ring_df['n_substituents']==n]['homa'].values for n in sub_groups]
    bp = ax.boxplot(homa_by_sub, labels=[str(n) for n in sub_groups], patch_artist=True, widths=0.6)
    colors4 = cm.RdYlGn(np.linspace(0.2, 0.8, len(sub_groups)))
    for patch, color in zip(bp['boxes'], colors4):
        patch.set_facecolor(color)
    ax.set_xlabel('Number of Substituents', fontsize=11)
    ax.set_ylabel('HOMA Value', fontsize=11)
    ax.set_title('HOMA Distribution by Substituent Count', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ring-Substituent Coverage Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '18_ring_substituent_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  图18: 环-取代基覆盖分析 ✓")


def print_gap_analysis(ring_df):
    """打印覆盖缺口分析"""
    print("\n" + "="*70)
    print("覆盖缺口分析")
    print("="*70)

    # 1. 环骨架类型覆盖
    print("\n1. 环骨架类型覆盖:")
    print(f"   总环骨架类型数: {ring_df['ring_composition'].nunique()}")
    top_skeletons = ring_df['ring_composition'].value_counts().head(10)
    print(f"   Top 10 骨架覆盖率: {top_skeletons.sum() / len(ring_df) * 100:.1f}%")
    print(f"   前5种骨架:")
    for skel, count in top_skeletons.head(5).items():
        print(f"     {skel}: {count} ({count/len(ring_df)*100:.1f}%)")

    # 2. 取代基类型覆盖
    print("\n2. 取代基类型覆盖:")
    all_subs = []
    for subs in ring_df['substituents']:
        all_subs.extend(subs)
    sub_counts = Counter(all_subs)
    print(f"   取代基种类数: {len(sub_counts)}")
    print(f"   取代基总频次: {sum(sub_counts.values())}")
    print(f"   前10种取代基:")
    for sub, count in sub_counts.most_common(10):
        print(f"     {sub}: {count} ({count/sum(sub_counts.values())*100:.1f}%)")

    # 3. 缺失的常见取代基
    print("\n3. 常见取代基覆盖检查:")
    common_subs = ['OH', 'OR', 'NH2', 'NHR', 'NR2', 'X(F)', 'X(Cl)', 'X(Br)', 'X(I)',
                   'COOH', 'COOR', 'CONH', 'C=O', 'NO2', 'C≡N', 'SH', 'SR',
                   'alkyl', 'aryl/R', 'fused_ring']
    missing = []
    rare = []
    for sub in common_subs:
        if sub not in sub_counts:
            missing.append(sub)
        elif sub_counts[sub] < 20:
            rare.append(f"{sub}({sub_counts[sub]})")
    if missing:
        print(f"   缺失的取代基: {', '.join(missing)}")
    else:
        print(f"   无缺失的常见取代基")
    if rare:
        print(f"   稀少的取代基 (<20次): {', '.join(rare)}")

    # 4. 取代基位置覆盖 (6元环)
    print("\n4. 6元环取代基位置覆盖:")
    six_ring = ring_df[ring_df['ring_size'] == 6]
    if len(six_ring) > 0:
        # 检查各种位置组合
        position_combos = {
            '无取代': 0, '单取代 (1位)': 0, '1,2-双取代': 0, '1,3-双取代': 0,
            '1,4-双取代': 0, '1,2,3-三取代': 0, '1,3,5-三取代': 0,
            '三取代(其他)': 0, '四取代+': 0
        }
        for pos_pattern in six_ring['pos_pattern']:
            n = len(pos_pattern)
            if n == 0:
                position_combos['无取代'] += 1
            elif n == 1:
                position_combos['单取代 (1位)'] += 1
            elif n == 2:
                diff = pos_pattern[1] - pos_pattern[0]
                if diff == 1:
                    position_combos['1,2-双取代'] += 1
                elif diff == 2:
                    position_combos['1,3-双取代'] += 1
                elif diff == 3:
                    position_combos['1,4-双取代'] += 1
                else:
                    position_combos['1,4-双取代'] += 1
            elif n == 3:
                diffs = [pos_pattern[i+1]-pos_pattern[i] for i in range(2)]
                if diffs == [1, 1]:
                    position_combos['1,2,3-三取代'] += 1
                elif diffs == [2, 2]:
                    position_combos['1,3,5-三取代'] += 1
                else:
                    position_combos['三取代(其他)'] += 1
            else:
                position_combos['四取代+'] += 1

        for combo, count in position_combos.items():
            pct = count / len(six_ring) * 100
            status = "✓" if count > 50 else "⚠️ 偏少" if count > 0 else "✗ 缺失"
            print(f"   {combo}: {count} ({pct:.1f}%) {status}")

    # 5. 覆盖建议
    print("\n" + "="*70)
    print("覆盖评估建议")
    print("="*70)

    # 计算覆盖率分数
    coverage_score = 0
    issues = []

    # 环骨架多样性
    if ring_df['ring_composition'].nunique() > 20:
        coverage_score += 25
    else:
        issues.append("环骨架类型偏少")
        coverage_score += 10

    # 取代基多样性
    if len(sub_counts) > 15:
        coverage_score += 25
    else:
        issues.append("取代基种类不足")
        coverage_score += 10

    # 位置模式多样性
    six_ring = ring_df[ring_df['ring_size'] == 6]
    if len(six_ring) > 0:
        n_pos_patterns = six_ring['pos_pattern'].nunique()
        if n_pos_patterns > 15:
            coverage_score += 25
        else:
            issues.append(f"6元环位置模式仅 {n_pos_patterns} 种, 偏少")
            coverage_score += 10

    # 取代基数量梯度
    n_sub_groups = ring_df['n_substituents'].nunique()
    if n_sub_groups >= 5:
        coverage_score += 25
    else:
        issues.append("取代基数量梯度不足")
        coverage_score += 10

    print(f"\n覆盖度总分: {coverage_score}/100")
    if issues:
        print(f"存在的问题:")
        for issue in issues:
            print(f"  - {issue}")

    if coverage_score >= 80:
        print("\n结论: ✅ 数据集覆盖度良好, 可暂不补充")
    elif coverage_score >= 60:
        print("\n结论: ⚠️ 数据集覆盖度尚可, 建议针对性补充少量缺口")
    else:
        print("\n结论: ❌ 数据集覆盖度不足, 强烈建议补充更多样本")

    return coverage_score, issues


def main():
    print("="*60)
    print("环-取代基覆盖范围分析")
    print("="*60)

    ring_df = analyze_ring_substituent_coverage()
    print(f"\n分析了 {len(ring_df)} 个目标环")

    print("\n生成图表...")
    plot_coverage_analysis(ring_df, OUTPUT_DIR)

    coverage_score, issues = print_gap_analysis(ring_df)

    # 保存详细数据
    ring_df.to_csv(os.path.join(OUTPUT_DIR, 'ring_substituent_analysis.csv'), index=False)
    print(f"\n详细数据: {os.path.join(OUTPUT_DIR, 'ring_substituent_analysis.csv')}")


if __name__ == '__main__':
    main()
