"""Data processing and graph representation for GNN."""
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset

def generate_atom_on_ring_mask(mol, atom_on_ring_list):
    """
    生成原子在环上的掩码
    
    参数:
        mol: RDKit分子对象
        atom_on_ring_list: 环上原子索引列表
        
    返回:
        atom_on_ring: 表示原子是否在环上的数组
    """
    atom_on_ring = np.zeros((mol.GetNumAtoms(),), dtype=int)
    for atom_idx in atom_on_ring_list:
        atom_on_ring[atom_idx - 1] = 1
    return atom_on_ring

def create_mask_matrix(atom_on_ring, node_vec_len, n_atoms):
    """
    创建掩码矩阵
    
    参数:
        atom_on_ring: 原子在环上的标记数组
        node_vec_len: 节点向量长度
        n_atoms: 原子数
        
    返回:
        mask_matrix: 掩码矩阵
    """
    extended_atom_on_ring = np.zeros((n_atoms,), dtype=int)
    extended_atom_on_ring[:len(atom_on_ring)] = atom_on_ring
    mask_matrix = np.zeros((n_atoms, node_vec_len), dtype=int)
    mask_matrix[extended_atom_on_ring == 1, :] = 1
    return mask_matrix

def process_and_save_data(dataset_path, node_vec_len, max_atoms):
    """
    处理并保存分子数据
    
    参数:
        dataset_path: 数据集CSV文件路径
        node_vec_len: 节点向量长度
        max_atoms: 最大原子数
        
    返回:
        data2: 包含分子图数据的字典
    """
    df = pd.read_csv(dataset_path)
    data2 = {
        'node_mats': [],
        'adj_mats': [],
        'outputs': [],
        'mask_mats': [],
        'indices': df.index.to_list(),
        'smiles': df["smiles"].to_list(),
        'atom_on_ring': df['atom_on_ring'].apply(lambda x: eval(x)).tolist(),
        'outputs_list': df["homa_value"].to_list()
    }
    
    for i in range(len(df)):
        smile = df.iloc[i]['smiles']
        atom_on_ring = data2['atom_on_ring'][i]
        output = data2['outputs_list'][i]

        mol = Graph(smile, atom_on_ring, node_vec_len, max_atoms)
        node_mat = mol.node_mat
        adj_mat = mol.adj_mat
        mask_mat = mol.mask_mat

        # 添加到数据字典
        data2['node_mats'].append(node_mat)
        data2['adj_mats'].append(adj_mat)
        data2['outputs'].append([output])
        data2['mask_mats'].append(mask_mat)

    return data2

class Graph:
    """
    分子图表示类
    """
    def __init__(self, molecule_smiles: str, atom_on_ring: list, node_vec_len: int, max_atoms: int = None):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.atom_on_ring = atom_on_ring
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()
        else:
            raise ValueError(f"Invalid SMILES: {molecule_smiles}")

    def smiles_to_mol(self):
        """从SMILES字符串创建RDKit分子对象"""
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return
        self.mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(self.mol)

    def smiles_to_graph(self):
        """将分子转换为图表示"""
        atoms = self.mol.GetAtoms()
        n_atoms = len(atoms) if self.max_atoms is None else self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # 填充节点特征矩阵
        for atom in atoms:
            atom_index = atom.GetIdx()
            if atom_index >= n_atoms:
                continue
                
            # 原子序数
            atom_no = atom.GetAtomicNum()
            node_mat[atom_index, atom_no] = 10
            
            # Gasteiger电荷
            node_mat[atom_index, self.node_vec_len - 1] = float(atom.GetProp('_GasteigerCharge'))
            
            # 杂化状态
            hybridization = atom.GetHybridization()
            if hybridization == Chem.rdchem.HybridizationType.SP:
                node_mat[atom_index, self.node_vec_len - 2] = 1
            elif hybridization == Chem.rdchem.HybridizationType.SP2:
                node_mat[atom_index, self.node_vec_len - 3] = 1
            elif hybridization == Chem.rdchem.HybridizationType.SP3:
                node_mat[atom_index, self.node_vec_len - 4] = 1
            
            # 芳香性
            node_mat[atom_index, self.node_vec_len - 5] = int(atom.GetIsAromatic())
            
            # 相邻原子数量
            node_mat[atom_index, self.node_vec_len - 6] = len(atom.GetNeighbors())
            
            # 氢原子数量
            node_mat[atom_index, self.node_vec_len - 7] = atom.GetTotalNumHs()
            
            # 氧/氮相关特征
            has_oxygen_neighbor = any(neighbor.GetSymbol() == "O" for neighbor in atom.GetNeighbors())
            has_nitrogen_neighbor = any(neighbor.GetSymbol() == "N" for neighbor in atom.GetNeighbors())
            
            # 氧/氮双键特征
            has_double_bond_with_oxygen = False
            has_double_bond_with_nitrogen = False
            for neighbor in atom.GetNeighbors():
                bond = self.mol.GetBondBetweenAtoms(atom_index, neighbor.GetIdx())
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    if neighbor.GetSymbol() == "O":
                        has_double_bond_with_oxygen = True
                    elif neighbor.GetSymbol() == "N":
                        has_double_bond_with_nitrogen = True
            
            # 设置特征值
            node_mat[atom_index, self.node_vec_len - 8] = int(has_oxygen_neighbor) * 10
            node_mat[atom_index, self.node_vec_len - 9] = int(has_double_bond_with_oxygen) * 10
            node_mat[atom_index, self.node_vec_len - 10] = int(has_nitrogen_neighbor) * 10
            node_mat[atom_index, self.node_vec_len - 11] = int(has_double_bond_with_nitrogen) * 10

        # 创建邻接矩阵
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)
        
        # 创建距离矩阵并规范化
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1
        adj_mat = adj_mat * (1 / dist_mat)
        
        # 填充矩阵以适应最大原子数
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")
        adj_mat = adj_mat + np.eye(n_atoms)  # 添加自连接

        # 保存矩阵
        self.node_mat = node_mat
        self.adj_mat = adj_mat

        # 创建掩码矩阵
        atom_on_ring_mask = generate_atom_on_ring_mask(self.mol, self.atom_on_ring)
        self.mask_mat = create_mask_matrix(atom_on_ring_mask, self.node_vec_len, n_atoms)

class GraphData(Dataset):
    """
    分子图数据集类
    """
    def __init__(self, data2: dict, node_vec_len: int, max_atoms: int):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.data2 = data2

    def __len__(self):
        return len(self.data2['node_mats'])

    def __getitem__(self, i: int):
        node_mat = torch.Tensor(self.data2['node_mats'][i])
        adj_mat = torch.Tensor(self.data2['adj_mats'][i])
        output = torch.Tensor(self.data2['outputs'][i])
        smile = self.data2['smiles'][i]
        mask_mat = torch.Tensor(self.data2['mask_mats'][i])
        return (node_mat, adj_mat), output, smile, mask_mat

def collate_graph_dataset(dataset: Dataset):
    """
    整理图数据集
    
    参数:
        dataset: 图数据集对象
        
    返回:
        node_mats_tensor: 节点矩阵张量
        adj_mats_tensor: 邻接矩阵张量
        outputs_tensor: 输出张量
        smiles: SMILES字符串列表
    """
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []
    mask_mats = []
    
    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile, mask_mat = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)
        mask_mats.append(mask_mat)
    
    # 创建张量
    node_mats_tensor = torch.stack(node_mats)
    adj_mats_tensor = torch.stack(adj_mats)
    mask_mats_tensor = torch.stack(mask_mats)
    outputs_tensor = torch.stack(outputs).squeeze(1)

    return (node_mats_tensor, adj_mats_tensor, mask_mats_tensor), outputs_tensor, smiles