"""
图数据加载 (第二、三层共用)

复用原始 unified_models/common/graphs.py 的 Graph/process_and_save_data,
不改动原代码, 仅做薄层封装以支持:
  - 自定义 GNN 模型 (adj 矩阵格式, GPU 张量批索引, 与 three_task_eval.py 一致)
  - PyG 模型 (edge_index 格式, 用于 AttentiveFP/DMPNN)

ring_flag_value:
  - 10 : 注入目标环标记 (label编码, 第二层GNN基线 + 第三层ring labeling, 与原 GNN-label 一致)
  - 0  : 不注入目标环标记 (仅用于ablation对比, 当前实验不使用)
"""
import os
import sys
import numpy as np
import torch

# 复用原始代码 (不改动)
ORIG_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'
sys.path.insert(0, ORIG_ROOT)
from unified_models.common.graphs import process_and_save_data  # noqa: E402


def load_adj_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu'):
    """加载邻接矩阵格式数据为 GPU 张量 (供自定义 GNN 模型使用)

    与 three_task_eval.py 的 load_data_to_tensor 等价。
    """
    data = process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col,
                                 ring_flag_value=ring_flag_value)
    n = len(data['node_mats'])
    node_mats = torch.tensor(np.array(data['node_mats']), dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    mask_mats = torch.tensor(np.array(data['mask_mats']), dtype=torch.float32, device=device)
    ring_indices = torch.tensor(np.array(data['ring_indices']), dtype=torch.long, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()
    smiles = data['smiles']
    return {
        'node_mats': node_mats, 'adj_mats': adj_mats, 'mask_mats': mask_mats,
        'ring_indices': ring_indices, 'outputs': outputs, 'smiles': smiles, 'n': n,
        'node_vec_len': node_vec_len, 'max_atoms': max_atoms,
    }


def load_pyg_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu'):
    """加载 PyG Data 列表 (供 AttentiveFP/DMPNN 使用)

    节点特征复用原始 Graph 的 node_mat (保持与自定义模型特征一致),
    额外构建 edge_index + edge_attr (键类型 one-hot)。
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolops
    from torch_geometric.data import Data
    import pandas as pd

    df = pd.read_csv(dataset_path)
    data_list = []
    # 原始 Graph 提供与自定义模型一致的 node_mat
    raw = process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col,
                                ring_flag_value=ring_flag_value)
    node_mats = raw['node_mats']
    outputs = raw['outputs_list']

    for i in range(len(df)):
        smile = df.iloc[i]['smiles']
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        try:
            AllChem.Compute2DCoords(mol_h)
        except Exception:
            pass

        n_atoms = mol_h.GetNumAtoms()
        # 节点特征: 截取实际原子数对应的 node_mat 行 (原始 max_atoms padding 截断)
        node_feat = np.array(node_mats[i][:n_atoms], dtype=np.float32)
        x = torch.tensor(node_feat, dtype=torch.float)

        # edge_index + edge_attr
        edges = []
        edge_attrs = []
        for bond in mol_h.GetBonds():
            i_, j_ = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            # one-hot: [single, double, triple, aromatic]
            vec = [0.0, 0.0, 0.0, 0.0]
            if bt == Chem.BondType.SINGLE:
                vec[0] = 1.0
            elif bt == Chem.BondType.DOUBLE:
                vec[1] = 1.0
            elif bt == Chem.BondType.TRIPLE:
                vec[2] = 1.0
            elif bt == Chem.BondType.AROMATIC:
                vec[3] = 1.0
            edges.append([i_, j_]); edges.append([j_, i_])
            edge_attrs.append(vec); edge_attrs.append(vec)

        if len(edges) == 0:
            edges = [[0, 0]]; edge_attrs = [[0, 0, 0, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([outputs[i]], dtype=torch.float)

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smile)
        data_list.append(d)

    return data_list, node_vec_len
