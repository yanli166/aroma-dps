"""
Method 6: 取代基位置编码 (输入层)

思路:
    在 GNN 输入层拼接取代基位置编码 (邻/间/对)。
    原始节点特征维度 node_vec_len=60, 拼接 6 维位置特征后扩展为 60+6=66。
    6 维位置特征: [is_ring_atom, is_ortho, is_meta, is_para, is_substituent, ring_pos]
    使用 hammett_constants.get_position_encoding 生成。

    与 Method 1 (Hammett σ 嵌入) 结构类似, 但用的是位置编码而非 Hammett σ,
    侧重于让模型显式感知取代基的相对位置 (ortho/meta/para)。

核心组件:
    - add_position_features(data): 数据预处理, 为每个分子生成位置编码并拼接到 node_mats
    - PositionGNN: 继承 GNNModel, 修改第一层输入维度为 node_vec_len + POS_DIM
    - build_model: 返回模型实例
"""
import os
import sys


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# 统一导入路径 (与任务约定一致)
CODE_END_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
for _p in (CODE_END_ROOT, ORIG_MODELS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from layer4_substituent.code.hammett_constants import get_position_encoding
from unified_models.gnn.model import GNNModel

# 位置编码维度 (与 hammett_constants.get_position_encoding 一致)
POS_DIM = 6


# ============== 数据预处理 ==============
def add_position_features(data, max_atoms=75, node_vec_len=60, device='cpu'):
    """为数据字典添加位置编码特征, 将 node_mats 从 (N, max_atoms, node_vec_len)
    扩展为 (N, max_atoms, node_vec_len + POS_DIM)

    Args:
        data: load_adj_format 返回的字典, 需包含 'node_mats' 和 'smiles'
        max_atoms: 最大原子数 (与原始数据一致, 默认 75)
        node_vec_len: 原始节点特征长度 (默认 60)
        device: 输出张量所在设备

    Returns:
        新的 data 字典 (浅拷贝, 不修改原始数据):
            - 'node_mats': 扩展后的张量 (N, max_atoms, node_vec_len+POS_DIM)
            - 'node_vec_len': node_vec_len + POS_DIM
            - 'orig_node_vec_len': 原始节点特征长度
            - 'pos_dim': POS_DIM
            其余字段原样保留
    """
    from rdkit import Chem

    smiles = data['smiles']
    n = len(smiles)

    # 原始 node_mats: torch.Tensor (N, max_atoms, node_vec_len)
    orig_node_mats = data['node_mats']
    if not torch.is_tensor(orig_node_mats):
        orig_node_mats = torch.as_tensor(np.array(orig_node_mats), dtype=torch.float32)

    pos_mats = np.zeros((n, max_atoms, POS_DIM), dtype=np.float32)

    for i in range(n):
        mol = Chem.MolFromSmiles(smiles[i])
        pe = get_position_encoding(mol, None, max_atoms)  # (max_atoms, POS_DIM)
        pos_mats[i] = pe[:max_atoms, :POS_DIM]

    pos_tensor = torch.tensor(pos_mats, dtype=torch.float32, device=orig_node_mats.device)
    # 拼接: 在最后一维 (特征维) 拼接
    extended_node_mats = torch.cat([orig_node_mats, pos_tensor], dim=-1)

    # 构造新 data (浅拷贝其余字段)
    new_data = dict(data)
    new_data['node_mats'] = extended_node_mats.to(device)
    new_data['node_vec_len'] = node_vec_len + POS_DIM
    new_data['orig_node_vec_len'] = node_vec_len
    new_data['pos_dim'] = POS_DIM
    return new_data


# ============== 模型 ==============
class PositionGNN(GNNModel):
    """在 GNN 输入层拼接位置编码的 GNN

    继承 GNNModel, 将第一层 init_transform 的输入维度从 node_vec_len 扩展到
    node_vec_len + pos_dim。forward 接口与 GNNModel 完全一致 (位置编码已在
    数据预处理阶段拼接到 node_mat 中)。
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label', pos_dim=POS_DIM):
        # 父类使用扩展后的 node_vec_len 构造第一层 Linear
        super().__init__(
            node_vec_len=node_vec_len + pos_dim,
            hidden_dim=hidden_dim,
            n_conv=n_conv,
            n_hidden=n_hidden,
            n_outputs=n_outputs,
            p_dropout=p_dropout,
            mode=mode,
        )
        self.pos_dim = pos_dim
        # 记录原始 (未拼接) 节点特征长度, 便于外部查询
        self.orig_node_vec_len = node_vec_len


# ============== 构建模型 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=2, n_outputs=1, p_dropout=0.2, mode='label',
                pos_dim=POS_DIM, **kwargs):
    """返回 Method 6 模型实例

    Args:
        base: 'gnn' (基于 GNNModel)
        node_vec_len: 原始 (未拼接位置编码) 节点特征长度
        pos_dim: 位置编码维度 (默认 6)
        其余参数同 GNNModel

    Returns:
        PositionGNN 实例, 其内部 node_vec_len 已扩展为 node_vec_len+pos_dim
    """
    base = base.lower()
    if base == 'gnn':
        return PositionGNN(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                           p_dropout, mode, pos_dim)
    else:
        raise ValueError(f"Method 6 currently supports base='gnn' only, got: {base}")


if __name__ == '__main__':
    # 简单自测: 实例化模型并做一次前向传播
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_atoms, nvl = 75, 60
    model = build_model(base='gnn', node_vec_len=nvl, hidden_dim=64, n_conv=2,
                        n_hidden=1, n_outputs=1, mode='label').to(dev)
    # 模拟预处理后的输入 (已拼接位置编码, 特征维 = nvl + POS_DIM)
    node_mat = torch.randn(4, n_atoms, nvl + POS_DIM, device=dev)
    adj_mat = torch.eye(n_atoms, device=dev).unsqueeze(0).expand(4, -1, -1).clone()
    out = model(node_mat, adj_mat)
    print(f"[m6] PositionGNN output shape: {out.shape} (expected: [4, 1])")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    print("[m6] OK")
