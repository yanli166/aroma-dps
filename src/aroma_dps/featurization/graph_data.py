"""
图数据加载 (四阶段共用)

ring_flag_value:
  - 10 : 注入目标环成员关系 (Stage2 Membership Encoding)
  - 1  : 二值 0/1 标记 (P1-1 ring_flag sensitivity test)
  - 0  : 不注入目标环标记 (Stage1 Base, 消息传递阶段无环条件化)

feature_mode:
  - 'standard'                     : 保留所有节点特征 (含 atom.GetIsAromatic())
  - 'explicit_aromaticity_ablated': 强制清零 node_vec_len-5 位 (原子芳香性 flag) +
                                     邻接矩阵中标记 aromatic bond 的第 4 维 (PyG edge_attr)
"""
import os
import numpy as np
import torch

from aroma_dps.data.tasks import clean_dataset_csv
from aroma_dps.featurization.graphs import process_and_save_data

DATA_CACHE = {}

# 节点特征中芳香性标志位 (Graph._smiles_to_graph 中 node_vec_len - 5 = atom.GetIsAromatic())
AROM_NODE_BIT_IDX_FROM_END = 5


def _ablate_aromaticity_node(node_mats: np.ndarray) -> np.ndarray:
    """强制清零 atom.GetIsAromatic() 写入的位 (node_vec_len - 5)。

    node_mats: (n_samples, max_atoms, node_vec_len)
    """
    out = node_mats.copy()
    n_dim = out.shape[-1]
    arom_idx = n_dim - AROM_NODE_BIT_IDX_FROM_END
    out[..., arom_idx] = 0.0
    return out


def _ablate_aromaticity_edge(edge_attrs: np.ndarray) -> np.ndarray:
    """PyG 模式下清零 edge_attr 的 aromatic bond bit (第 4 维, 索引 3)。

    edge_attrs: (n_samples, max_atoms*2, 4)
    """
    out = edge_attrs.copy()
    out[..., 3] = 0.0
    return out


def load_adj_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu', feature_mode='standard'):
    """加载邻接矩阵格式数据为 GPU 张量 (供自定义 GNN 模型使用)"""
    clean_path = clean_dataset_csv(dataset_path, target_col)
    cache_key = (clean_path, target_col, node_vec_len, max_atoms,
                 ring_flag_value, str(device), feature_mode)

    if cache_key in DATA_CACHE:
        return DATA_CACHE[cache_key]

    print(f"  [load_adj_format] 加载 {os.path.basename(clean_path)} "
          f"(ring_flag={ring_flag_value}, max_atoms={max_atoms}, "
          f"feature_mode={feature_mode})...", flush=True)
    data = process_and_save_data(clean_path, node_vec_len, max_atoms, target_col,
                                 ring_flag_value=ring_flag_value)
    node_mats_np = np.array(data['node_mats'], dtype=np.float32)
    if feature_mode == 'explicit_aromaticity_ablated':
        node_mats_np = _ablate_aromaticity_node(node_mats_np)
    n = len(node_mats_np)
    node_mats = torch.tensor(node_mats_np, dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    mask_mats = torch.tensor(np.array(data['mask_mats']), dtype=torch.float32, device=device)
    ring_indices = torch.tensor(np.array(data['ring_indices']), dtype=torch.long, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()
    smiles = data['smiles']
    result = {
        'node_mats': node_mats, 'adj_mats': adj_mats, 'mask_mats': mask_mats,
        'ring_indices': ring_indices, 'outputs': outputs, 'smiles': smiles, 'n': n,
        'node_vec_len': node_vec_len, 'max_atoms': max_atoms,
        'feature_mode': feature_mode, 'ring_flag_value': ring_flag_value,
    }
    DATA_CACHE[cache_key] = result
    return result


def load_pyg_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu', feature_mode='standard'):
    """加载 PyG Data 列表 (供 DMPNN 等使用)

    在 Data 中额外携带 ring_indices (标记目标环原子), 供 Stage1 固定环平均池化使用。
explicit_aromaticity_ablated 模式下清零 atom.GetIsAromatic() 位与 edge_attr 中 aromatic bond bit。
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolops
    from torch_geometric.data import Data
    import pandas as pd

    clean_path = clean_dataset_csv(dataset_path, target_col)
    cache_key = ('pyg', clean_path, target_col, node_vec_len, max_atoms,
                 ring_flag_value, feature_mode)

    if cache_key in DATA_CACHE:
        return DATA_CACHE[cache_key]

    print(f"  [load_pyg_format] 加载 {os.path.basename(clean_path)} "
          f"(feature_mode={feature_mode})...", flush=True)
    df = pd.read_csv(clean_path)
    data_list = []
    raw = process_and_save_data(clean_path, node_vec_len, max_atoms, target_col,
                                ring_flag_value=ring_flag_value)
    node_mats = np.array(raw['node_mats'], dtype=np.float32)
    if feature_mode == 'explicit_aromaticity_ablated':
        node_mats = _ablate_aromaticity_node(node_mats)
    outputs = raw['outputs_list']
    ring_indices_all = raw['ring_indices']  # 每个分子的目标环原子位置标记

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
        node_feat = node_mats[i][:n_atoms]
        x = torch.tensor(node_feat, dtype=torch.float)

        edges = []
        edge_attrs = []
        for bond in mol_h.GetBonds():
            i_, j_ = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
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

        if feature_mode == 'explicit_aromaticity_ablated':
            edge_attrs = _ablate_aromaticity_edge(np.array(edge_attrs, dtype=np.float32)).tolist()

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([outputs[i]], dtype=torch.float)

        # ring_mask: 标记哪些原子属于目标环 (用于固定环平均池化)
        ri = ring_indices_all[i][:n_atoms]
        ring_mask = torch.tensor((ri >= 0).astype(np.float32), dtype=torch.float)

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smile,
                 ring_mask=ring_mask)
        data_list.append(d)

    DATA_CACHE[cache_key] = (data_list, node_vec_len)
    return data_list, node_vec_len
