"""
图数据加载 (第二、三层共用) — 修复版

修复:
  - m4: 路径配置化 (从 constants.py 读取 ORIG_MODELS_ROOT)
  - 数据清洗: 自动处理 NaN / Windows 换行符 / 多余空列
  - 数据缓存: 同一 (path, target, ring_flag, max_atoms) 只加载一次

ring_flag_value:
  - 10 : 注入目标环标记 (label编码)
  - 0  : 不注入目标环标记 (用于 none 编码 ablation 对照, M3)
"""
import os
import sys
import numpy as np
import torch

from common.constants import ORIG_MODELS_ROOT
from common.tasks import clean_dataset_csv
sys.path.insert(0, ORIG_MODELS_ROOT)
from unified_models.common.graphs import process_and_save_data  # noqa: E402

# 数据缓存: 避免同一组合重复加载 (Layer 3 有 90 种 model×encoding 组合)
_DATA_CACHE = {}


def load_adj_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu'):
    """加载邻接矩阵格式数据为 GPU 张量 (供自定义 GNN 模型使用)"""
    clean_path = clean_dataset_csv(dataset_path, target_col)
    cache_key = (clean_path, target_col, node_vec_len, max_atoms, ring_flag_value, str(device))

    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    print(f"  [load_adj_format] 加载 {os.path.basename(clean_path)} (ring_flag={ring_flag_value}, max_atoms={max_atoms})...", flush=True)
    data = process_and_save_data(clean_path, node_vec_len, max_atoms, target_col,
                                 ring_flag_value=ring_flag_value)
    n = len(data['node_mats'])
    node_mats = torch.tensor(np.array(data['node_mats']), dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    mask_mats = torch.tensor(np.array(data['mask_mats']), dtype=torch.float32, device=device)
    ring_indices = torch.tensor(np.array(data['ring_indices']), dtype=torch.long, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()
    smiles = data['smiles']
    result = {
        'node_mats': node_mats, 'adj_mats': adj_mats, 'mask_mats': mask_mats,
        'ring_indices': ring_indices, 'outputs': outputs, 'smiles': smiles, 'n': n,
        'node_vec_len': node_vec_len, 'max_atoms': max_atoms,
    }
    _DATA_CACHE[cache_key] = result
    return result


def load_pyg_format(dataset_path, target_col, node_vec_len=60, max_atoms=75,
                    ring_flag_value=0, device='cpu'):
    """加载 PyG Data 列表 (供 AttentiveFP/DMPNN 使用)"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolops
    from torch_geometric.data import Data
    import pandas as pd

    clean_path = clean_dataset_csv(dataset_path, target_col)
    cache_key = ('pyg', clean_path, target_col, node_vec_len, max_atoms, ring_flag_value)

    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    print(f"  [load_pyg_format] 加载 {os.path.basename(clean_path)}...", flush=True)
    df = pd.read_csv(clean_path)
    data_list = []
    raw = process_and_save_data(clean_path, node_vec_len, max_atoms, target_col,
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
        node_feat = np.array(node_mats[i][:n_atoms], dtype=np.float32)
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

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([outputs[i]], dtype=torch.float)

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smile)
        data_list.append(d)

    _DATA_CACHE[cache_key] = (data_list, node_vec_len)
    return data_list, node_vec_len
