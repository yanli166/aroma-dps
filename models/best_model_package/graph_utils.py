"""图数据构建（自包含、修复 atom_on_ring -1 偏移 bug）

与原始 unified_models/common/graphs.py 保持特征布局一致，
但修复 target_ring_atoms 的 -1 偏移：
  - CSV 中 atom_on_ring 已是 0-based（直接匹配 RDKit AddHs 后 GetRingInfo().AtomRings()）
  - 原始代码 `[idx - 1 for idx in atom_on_ring]` 会把目标环标记错位 1 个原子

本模块可直接独立调用，不依赖原始数据管线。
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdDistGeom as molDG

NODE_VEC_LEN = 60
MAX_ATOMS = 75


def build_graph(smiles, atom_on_ring, node_vec_len=NODE_VEC_LEN,
                max_atoms=MAX_ATOMS, ring_flag_value=1):
    """构建单个分子的图表示（修复 -1 bug 版本）。

    Args:
        smiles: SMILES 字符串
        atom_on_ring: 目标环原子 0-based 索引列表（H-added 分子上，与 CSV 一致）
        node_vec_len: 节点特征维度 (默认 60)
        max_atoms: 最大原子数 (默认 75)
        ring_flag_value: 目标环标记注入值 (1=binary, 10=amplitude)

    Returns:
        dict: node_mat, adj_mat, mask_mat, ring_indices
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)

    atoms = mol.GetAtoms()
    n_atoms = max_atoms
    node_mat = np.zeros((n_atoms, node_vec_len))
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    atom_ring_sizes = {}
    for ring in atom_rings:
        for idx in ring:
            atom_ring_sizes.setdefault(idx, set()).add(len(ring))

    # 修复：atom_on_ring 已是 0-based，不再 -1
    target_ring_atoms = [int(idx) for idx in atom_on_ring]

    for atom in atoms:
        i = atom.GetIdx()
        if i >= n_atoms:
            continue
        atom_no = atom.GetAtomicNum()
        node_mat[i, atom_no] = 10
        node_mat[i, node_vec_len - 1] = float(atom.GetProp('_GasteigerCharge'))

        hyb = atom.GetHybridization()
        if hyb == Chem.rdchem.HybridizationType.SP:
            node_mat[i, node_vec_len - 2] = 1
        elif hyb == Chem.rdchem.HybridizationType.SP2:
            node_mat[i, node_vec_len - 3] = 1
        elif hyb == Chem.rdchem.HybridizationType.SP3:
            node_mat[i, node_vec_len - 4] = 1

        node_mat[i, node_vec_len - 5] = int(atom.GetIsAromatic())
        node_mat[i, node_vec_len - 6] = len(atom.GetNeighbors())
        node_mat[i, node_vec_len - 7] = atom.GetTotalNumHs()

        has_o = any(nb.GetSymbol() == "O" for nb in atom.GetNeighbors())
        has_n = any(nb.GetSymbol() == "N" for nb in atom.GetNeighbors())
        has_double_o = False
        has_double_n = False
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
            if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                if nb.GetSymbol() == "O":
                    has_double_o = True
                elif nb.GetSymbol() == "N":
                    has_double_n = True

        node_mat[i, node_vec_len - 8] = int(has_o) * 10
        node_mat[i, node_vec_len - 9] = int(has_double_o) * 10
        node_mat[i, node_vec_len - 10] = int(has_n) * 10
        node_mat[i, node_vec_len - 11] = int(has_double_n) * 10

        ring_sizes = atom_ring_sizes.get(i, set())
        if 6 in ring_sizes:
            node_mat[i, node_vec_len - 12] = 1
        if 5 in ring_sizes:
            node_mat[i, node_vec_len - 13] = 1
        node_mat[i, node_vec_len - 14] = len(ring_sizes)

        # 目标环标记 (binary 0/1 或 amplitude)
        if node_vec_len >= 15:
            is_in_target = 1 if i in target_ring_atoms else 0
            node_mat[i, node_vec_len - 15] = is_in_target * ring_flag_value

    # 邻接矩阵 (加权)
    adj_mat = rdmolops.GetAdjacencyMatrix(mol).astype(float)
    dist_mat = molDG.GetMoleculeBoundsMatrix(mol)
    dist_mat[dist_mat == 0.0] = 1
    adj_mat = adj_mat * (1 / dist_mat)
    dim_add = n_atoms - adj_mat.shape[0]
    if dim_add > 0:
        adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")
    adj_mat = adj_mat + np.eye(n_atoms)

    # mask 矩阵: 标记目标环原子
    mask_mat = np.zeros((n_atoms, node_vec_len))
    for idx in target_ring_atoms:
        if 0 <= idx < n_atoms:
            mask_mat[idx, :] = 1.0

    # ring_indices: 目标环原子位置标记
    ring_indices = np.full(n_atoms, -1, dtype=np.int64)
    for pos, idx in enumerate(target_ring_atoms):
        if 0 <= idx < n_atoms:
            ring_indices[idx] = pos

    return {
        'node_mat': node_mat,
        'adj_mat': adj_mat,
        'mask_mat': mask_mat,
        'ring_indices': ring_indices,
    }


def smiles_to_graphs(smiles_list, atom_on_ring_list, node_vec_len=NODE_VEC_LEN,
                     max_atoms=MAX_ATOMS, ring_flag_value=1):
    """批量构建图，返回堆叠后的 numpy 数组（用于训练数据加载）。"""
    node_mats, adj_mats, mask_mats, ring_indices = [], [], [], []
    for smi, aor in zip(smiles_list, atom_on_ring_list):
        g = build_graph(smi, aor, node_vec_len, max_atoms, ring_flag_value)
        node_mats.append(g['node_mat'])
        adj_mats.append(g['adj_mat'])
        mask_mats.append(g['mask_mat'])
        ring_indices.append(g['ring_indices'])
    return {
        'node_mats': np.array(node_mats, dtype=np.float32),
        'adj_mats': np.array(adj_mats, dtype=np.float32),
        'mask_mats': np.array(mask_mats, dtype=np.float32),
        'ring_indices': np.array(ring_indices, dtype=np.int64),
    }
