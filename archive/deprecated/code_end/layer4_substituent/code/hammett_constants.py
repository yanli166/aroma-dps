"""
Hammett 取代基常数表 (σ_meta, σ_para)

数据来源: Hansch, C.; Leo, A.; Taft, R. W. Chem. Rev. 1991, 91, 165-195.
用途: Method 1 (Hammett σ 嵌入) 和 Method 2 (单调性约束) 的查表基础

对于不在表中的取代基, 默认 σ=0 (无电子效应信息)
"""
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Hammett σ 常数表 (常见取代基)
# 格式: SMARTS模式 -> (sigma_meta, sigma_para, 取代基名称)
HAMMETT_TABLE = {
    # 吸电子基团 (σ > 0)
    '[NX3+](=O)[O-]':   (0.71, 0.78, 'NO2'),       # 硝基
    '[NX3](=O)=O':      (0.71, 0.78, 'NO2_alt'),
    'C#N':              (0.56, 0.66, 'CN'),          # 氰基
    'C(=O)O':           (0.37, 0.45, 'COOH'),        # 羧基
    'C(=O)[OH]':        (0.37, 0.45, 'COOH_alt'),
    'C(=O)OC':          (0.37, 0.45, 'COOMe'),       # 甲酯基
    'C(=O)N':           (0.38, 0.36, 'CONH2'),       # 酰胺基
    'C=O':              (0.35, 0.42, 'CHO'),         # 醛基
    'C(=O)C':           (0.38, 0.50, 'COCH3'),       # 乙酰基
    'S(=O)(=O)O':       (0.39, 0.57, 'SO3H'),        # 磺酸基
    'S(=O)(=O)C':       (0.51, 0.72, 'SO2CH3'),      # 甲基磺酰基
    'S(=O)(=O)N':       (0.46, 0.57, 'SO2NH2'),      # 磺酰胺基
    'F':                (0.34, 0.06, 'F'),            # 氟
    'Cl':               (0.37, 0.23, 'Cl'),           # 氯
    'Br':               (0.39, 0.23, 'Br'),           # 溴
    'I':                (0.35, 0.18, 'I'),            # 碘
    'CF3':              (0.43, 0.54, 'CF3'),          # 三氟甲基
    'CCl3':             (0.40, 0.46, 'CCl3'),         # 三氯甲基
    # 给电子基团 (σ < 0)
    'O':                (0.12, -0.37, 'OH'),          # 羟基
    'OC':               (0.12, -0.27, 'OCH3'),        # 甲氧基
    'OC(C)C':           (0.10, -0.25, 'OiPr'),        # 异丙氧基
    'N':                (-0.16, -0.66, 'NH2'),        # 氨基
    'NC':               (-0.30, -0.84, 'NMe2'),       # 二甲氨基
    'NC(C)C':           (-0.22, -0.72, 'NiPr'),       # 二异丙氨基
    'C':                (-0.07, -0.17, 'CH3'),        # 甲基
    'CC':               (-0.07, -0.15, 'C2H5'),       # 乙基
    'C(C)C':            (-0.12, -0.20, 'iPr'),        # 异丙基
    'C(C)(C)C':         (-0.10, -0.20, 'tBu'),        # 叔丁基
    # 弱电子效应
    'c':                (0.06, -0.01, 'Ph'),          # 苯基
    'P':                (0.05, 0.02, 'PH2'),          # 膦基
    'S':                (0.25, 0.00, 'SH'),           # 巯基
    'SC':               (0.15, 0.00, 'SCH3'),         # 甲硫基
    '[Si](C)(C)C':      (-0.04, 0.00, 'TMS'),         # 三甲基硅基
}

# 预编译 SMARTS 模式 (按复杂度排序, 优先匹配复杂基团)
_COMPILED_PATTERNS = []
for smarts, (sigma_m, sigma_p, name) in sorted(
    HAMMETT_TABLE.items(), key=lambda x: -len(x[0])
):
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        _COMPILED_PATTERNS.append((patt, sigma_m, sigma_p, name, smarts))


def identify_substituents(mol, ring_atom_indices=None):
    """识别分子中芳香环上的取代基及其位置

    Args:
        mol: RDKit Mol 对象
        ring_atom_indices: 芳香环原子索引集合 (若为None, 自动检测)

    Returns:
        list of dict: [{atom_idx, smarts, sigma_m, sigma_p, name, position}]
        position: 'ortho' (1,2), 'meta' (1,3), 'para' (1,4), or 'other'
    """
    if mol is None:
        return []

    if ring_atom_indices is None:
        # 自动检测芳香环
        ring_info = mol.GetRingInfo()
        aromatic_rings = []
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings.append(ring)
        if not aromatic_rings:
            return []
        # 取最大的芳香环作为参考
        ring_atom_indices = set()
        for ring in aromatic_rings:
            ring_atom_indices.update(ring)

    ring_atoms = sorted(ring_atom_indices)
    if len(ring_atoms) < 5:
        return []

    # 找到环上的取代基 (非环原子连接到环原子)
    substituents = []
    for ring_atom_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        for neighbor in ring_atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_atom_indices:
                # 这是一个取代基的起始原子
                # 尝试匹配 Hammett 表
                matched = False
                for patt, sigma_m, sigma_p, name, smarts in _COMPILED_PATTERNS:
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        if neighbor.GetIdx() in match:
                            # 确定位置 (相对于环上第一个原子的距离)
                            pos = _determine_position(mol, ring_atom_idx, ring_atoms)
                            substituents.append({
                                'atom_idx': neighbor.GetIdx(),
                                'ring_atom_idx': ring_atom_idx,
                                'smarts': smarts,
                                'sigma_m': sigma_m,
                                'sigma_p': sigma_p,
                                'name': name,
                                'position': pos,
                            })
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    # 未知取代基, σ=0
                    pos = _determine_position(mol, ring_atom_idx, ring_atoms)
                    # 尝试获取取代基名称
                    smarts_sym = neighbor.GetSymbol()
                    substituents.append({
                        'atom_idx': neighbor.GetIdx(),
                        'ring_atom_idx': ring_atom_idx,
                        'smarts': smarts_sym,
                        'sigma_m': 0.0,
                        'sigma_p': 0.0,
                        'name': smarts_sym,
                        'position': pos,
                    })

    return substituents


def _determine_position(mol, ring_atom_idx, ring_atoms):
    """确定取代基相对于参考原子的位置 (ortho/meta/para)

    简化实现: 基于在环上的拓扑距离
    ortho: 距离参考原子1步 (邻位)
    meta:  距离参考原子2步 (间位)
    para:  距离参考原子3步 (对位)
    """
    # 以环上第一个原子为参考
    ref_atom = ring_atoms[0]
    if ring_atom_idx == ref_atom:
        return 'ipso'

    # 计算 BFS 距离 (沿环)
    ring_set = set(ring_atoms)
    from collections import deque
    queue = deque([(ref_atom, 0)])
    visited = {ref_atom}
    while queue:
        atom_idx, dist = queue.popleft()
        if atom_idx == ring_atom_idx:
            if dist <= 1:
                return 'ortho'
            elif dist == 2:
                return 'meta'
            elif dist == 3:
                return 'para'
            else:
                return 'other'
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in ring_set and nidx not in visited:
                visited.add(nidx)
                queue.append((nidx, dist + 1))
    return 'other'


def get_hammett_features(mol, n_atoms, node_vec_len=60):
    """生成分子的 Hammett σ 特征向量

    为每个原子生成 [sigma_m, sigma_p, is_substituent, position_ortho, position_meta, position_para] 6维特征

    Args:
        mol: RDKit Mol
        n_atoms: 原子数 (含H)
        node_vec_len: 节点特征长度 (用于padding)

    Returns:
        np.ndarray: shape (n_atoms, 6) 的 Hammett 特征矩阵
    """
    # 6维: [sigma_m, sigma_p, is_substituent, ortho, meta, para]
    hammett_dim = 6
    features = np.zeros((node_vec_len, hammett_dim), dtype=np.float32)

    if mol is None:
        return features

    mol_h = Chem.AddHs(mol)
    actual_atoms = mol_h.GetNumAtoms()

    substituents = identify_substituents(mol)

    for sub in substituents:
        atom_idx = sub['atom_idx']
        if atom_idx < min(actual_atoms, node_vec_len):
            features[atom_idx, 0] = sub['sigma_m']
            features[atom_idx, 1] = sub['sigma_p']
            features[atom_idx, 2] = 1.0  # is_substituent

            pos = sub['position']
            if pos == 'ortho':
                features[atom_idx, 3] = 1.0
            elif pos == 'meta':
                features[atom_idx, 4] = 1.0
            elif pos == 'para':
                features[atom_idx, 5] = 1.0

    return features


def get_molecular_hammett_vector(mol):
    """获取分子级 Hammett 向量 (所有取代基 σ 的聚合)

    Returns:
        np.ndarray: [sum_sigma_m, sum_sigma_p, mean_sigma_m, mean_sigma_p,
                      n_substituents, max_sigma_m, min_sigma_m, max_sigma_p, min_sigma_p]
    """
    subs = identify_substituents(mol)
    if not subs:
        return np.zeros(9, dtype=np.float32)

    sigma_ms = [s['sigma_m'] for s in subs]
    sigma_ps = [s['sigma_p'] for s in subs]

    return np.array([
        sum(sigma_ms), sum(sigma_ps),
        np.mean(sigma_ms), np.mean(sigma_ps),
        len(subs),
        max(sigma_ms), min(sigma_ms),
        max(sigma_ps), min(sigma_ps),
    ], dtype=np.float32)


# 取代基电子效应排序 (用于 Method 2 单调性约束)
# 从强给电子到强吸电子
ELECTRONIC_ORDER = [
    'NMe2', 'NH2', 'NiPr', 'OH', 'OCH3', 'OiPr',
    'tBu', 'iPr', 'C2H5', 'CH3',
    'TMS', 'PH2', 'H', 'Ph', 'SCH3', 'SH',
    'F', 'I', 'Br', 'Cl',
    'CF3', 'CCl3', 'CHO', 'CONH2', 'COOH', 'COOMe',
    'SO2NH2', 'SO3H', 'SO2CH3', 'CN', 'NO2',
]

def get_electronic_rank(name):
    """获取取代基在电子效应排序中的位置 (0=最强给电子, 1=最强吸电子)"""
    if name in ELECTRONIC_ORDER:
        return ELECTRONIC_ORDER.index(name) / max(1, len(ELECTRONIC_ORDER) - 1)
    return 0.5  # 未知, 中性


def get_position_encoding(mol, ring_atom_indices=None, n_atoms=75):
    """生成取代基位置编码 (Method 6)

    为每个原子生成 [is_ring_atom, is_ortho, is_meta, is_para, is_substituent, ring_pos] 6维编码

    Args:
        mol: RDKit Mol
        ring_atom_indices: 芳香环原子索引
        n_atoms: 最大原子数

    Returns:
        np.ndarray: shape (n_atoms, 6)
    """
    pos_dim = 6
    encoding = np.zeros((n_atoms, pos_dim), dtype=np.float32)

    if mol is None:
        return encoding

    if ring_atom_indices is None:
        ring_info = mol.GetRingInfo()
        aromatic_rings = []
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings.append(ring)
        if not aromatic_rings:
            return encoding
        ring_atom_indices = set()
        for ring in aromatic_rings:
            ring_atom_indices.update(ring)

    ring_atoms = sorted(ring_atom_indices)

    # 标记环原子
    for idx in ring_atoms:
        if idx < n_atoms:
            encoding[idx, 0] = 1.0  # is_ring_atom

    # 标记取代基及其位置
    subs = identify_substituents(mol, ring_atom_indices)
    for sub in subs:
        atom_idx = sub['atom_idx']
        if atom_idx < n_atoms:
            encoding[atom_idx, 4] = 1.0  # is_substituent
            pos = sub['position']
            if pos == 'ortho':
                encoding[atom_idx, 1] = 1.0
            elif pos == 'meta':
                encoding[atom_idx, 2] = 1.0
            elif pos == 'para':
                encoding[atom_idx, 3] = 1.0
            # 环上位置索引
            ring_atom_idx = sub['ring_atom_idx']
            if ring_atom_idx < n_atoms:
                encoding[ring_atom_idx, 5] = ring_atoms.index(ring_atom_idx) / max(1, len(ring_atoms) - 1)

    return encoding


if __name__ == '__main__':
    # 测试
    test_smiles = ['c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc([N+](=O)[O-])cc1',
                   'c1ccccc1C', 'c1ccc(OC)cc1', 'c1ccc(Cl)cc1']
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        subs = identify_substituents(mol)
        vec = get_molecular_hammett_vector(mol)
        print(f'{smi}:')
        for s in subs:
            print(f'  {s["name"]} (pos={s["position"]}, σ_m={s["sigma_m"]:.2f}, σ_p={s["sigma_p"]:.2f})')
        print(f'  Molecular vector: {vec}')
        print()
