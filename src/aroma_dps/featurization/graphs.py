"""
统一的图数据处理模块
支持三种环信息编码方式: label, mask, pool
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolDescriptors
from rdkit.Chem import rdDistGeom as molDG


def process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col='homa_value',
                          ring_flag_value=10):
    """读取CSV并生成图数据字典

    ring_flag_value: 目标环原子的标记值 (默认 10, 与原始 GNN-label 一致;
                     设为 1 则用二值标记, 用于 ablation 对比)
    """
    df = pd.read_csv(dataset_path)
    data = {
        'node_mats': [],
        'adj_mats': [],
        'mask_mats': [],
        'ring_indices': [],
        'outputs': [],
        'smiles': df["smiles"].to_list(),
        'atom_on_ring': df['atom_on_ring'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist(),
        'outputs_list': df[target_col].to_list(),
    }

    for i in range(len(df)):
        smile = df.iloc[i]['smiles']
        atom_on_ring = data['atom_on_ring'][i]
        output = data['outputs_list'][i]

        graph = Graph(smile, atom_on_ring, node_vec_len, max_atoms,
                      ring_flag_value=ring_flag_value)
        data['node_mats'].append(graph.node_mat)
        data['adj_mats'].append(graph.adj_mat)
        data['mask_mats'].append(graph.mask_mat)
        data['ring_indices'].append(graph.ring_indices)
        data['outputs'].append([output])

    return data


class Graph:
    """分子图表示，统一支持label/mask/pool三种编码方式"""

    def __init__(self, molecule_smiles, atom_on_ring, node_vec_len, max_atoms=None,
                 ring_flag_value=10):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.atom_on_ring = atom_on_ring if atom_on_ring else []
        self.ring_flag_value = ring_flag_value
        self.mol = None
        self._build()

    def _build(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        self.mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(self.mol)
        self._smiles_to_graph()

    def _smiles_to_graph(self):
        atoms = self.mol.GetAtoms()
        n_atoms = len(list(atoms)) if self.max_atoms is None else self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))
        ring_info = self.mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # 每个原子所属的环大小集合
        atom_ring_sizes = {}
        for ring in atom_rings:
            for idx in ring:
                atom_ring_sizes.setdefault(idx, set()).add(len(ring))

        # 目标环原子索引 (CSV atom_on_ring 已是 0-based, 直接使用)
        target_ring_atoms = list(self.atom_on_ring)

        for atom in atoms:
            i = atom.GetIdx()
            if i >= n_atoms:
                continue
            atom_no = atom.GetAtomicNum()
            node_mat[i, atom_no] = 10
            node_mat[i, self.node_vec_len - 1] = float(atom.GetProp('_GasteigerCharge'))

            # 杂化
            hyb = atom.GetHybridization()
            if hyb == Chem.rdchem.HybridizationType.SP:
                node_mat[i, self.node_vec_len - 2] = 1
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                node_mat[i, self.node_vec_len - 3] = 1
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                node_mat[i, self.node_vec_len - 4] = 1

            node_mat[i, self.node_vec_len - 5] = int(atom.GetIsAromatic())
            node_mat[i, self.node_vec_len - 6] = len(atom.GetNeighbors())
            node_mat[i, self.node_vec_len - 7] = atom.GetTotalNumHs()

            # O/N 邻居和双键
            has_o = any(nb.GetSymbol() == "O" for nb in atom.GetNeighbors())
            has_n = any(nb.GetSymbol() == "N" for nb in atom.GetNeighbors())
            has_double_o = False
            has_double_n = False
            for nb in atom.GetNeighbors():
                bond = self.mol.GetBondBetweenAtoms(i, nb.GetIdx())
                if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    if nb.GetSymbol() == "O":
                        has_double_o = True
                    elif nb.GetSymbol() == "N":
                        has_double_n = True

            node_mat[i, self.node_vec_len - 8] = int(has_o) * 10
            node_mat[i, self.node_vec_len - 9] = int(has_double_o) * 10
            node_mat[i, self.node_vec_len - 10] = int(has_n) * 10
            node_mat[i, self.node_vec_len - 11] = int(has_double_n) * 10

            # 环大小
            ring_sizes = atom_ring_sizes.get(i, set())
            if 6 in ring_sizes:
                node_mat[i, self.node_vec_len - 12] = 1
            if 5 in ring_sizes:
                node_mat[i, self.node_vec_len - 13] = 1
            node_mat[i, self.node_vec_len - 14] = len(ring_sizes)

            # label版本: 在特征中加入目标环标记
            if self.node_vec_len >= 15:
                is_in_target = 1 if i in target_ring_atoms else 0
                node_mat[i, self.node_vec_len - 15] = is_in_target * self.ring_flag_value

        # 邻接矩阵 (加权)
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol).astype(float)
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1
        adj_mat = adj_mat * (1 / dist_mat)
        dim_add = n_atoms - adj_mat.shape[0]
        if dim_add > 0:
            adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")
        adj_mat = adj_mat + np.eye(n_atoms)

        # mask矩阵: 标记目标环原子 (用于mask版本)
        mask_mat = np.zeros((n_atoms, self.node_vec_len))
        for idx in target_ring_atoms:
            if 0 <= idx < n_atoms:
                mask_mat[idx, :] = 1.0

        # ring_indices: 用于pool版本的高级池化 (标记目标环位置)
        ring_indices = np.full(n_atoms, -1, dtype=np.int64)
        for pos, idx in enumerate(target_ring_atoms):
            if 0 <= idx < n_atoms:
                ring_indices[idx] = pos

        self.node_mat = node_mat
        self.adj_mat = adj_mat
        self.mask_mat = mask_mat
        self.ring_indices = ring_indices


class GraphData(Dataset):
    """PyTorch Dataset 包装类"""

    def __init__(self, data, node_vec_len, max_atoms):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.data = data

    def __len__(self):
        return len(self.data['node_mats'])

    def __getitem__(self, i):
        node_mat = torch.Tensor(self.data['node_mats'][i])
        adj_mat = torch.Tensor(self.data['adj_mats'][i])
        mask_mat = torch.Tensor(self.data['mask_mats'][i])
        ring_indices = torch.LongTensor(self.data['ring_indices'][i])
        output = torch.Tensor(self.data['outputs'][i])
        smile = self.data['smiles'][i]
        return (node_mat, adj_mat, mask_mat, ring_indices), output, smile


def collate_graph_dataset(dataset):
    """统一批处理函数 - 返回所有可能的张量供不同模型使用"""
    node_mats, adj_mats, mask_mats, ring_indices = [], [], [], []
    outputs, smiles = [], []
    for i in range(len(dataset)):
        (node_mat, adj_mat, mask_mat, ring_idx), output, smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        mask_mats.append(mask_mat)
        ring_indices.append(ring_idx)
        outputs.append(output)
        smiles.append(smile)

    node_mats_tensor = torch.stack(node_mats)
    adj_mats_tensor = torch.stack(adj_mats)
    mask_mats_tensor = torch.stack(mask_mats)
    ring_indices_tensor = torch.stack(ring_indices)
    outputs_tensor = torch.stack(outputs).squeeze(1)

    return (node_mats_tensor, adj_mats_tensor, mask_mats_tensor, ring_indices_tensor), outputs_tensor, smiles
