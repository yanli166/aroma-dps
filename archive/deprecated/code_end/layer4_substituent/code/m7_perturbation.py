"""
Method 7: 取代基扰动学习框架 (聚合层 + 双输出头)

思路:
    核心思想: P_total = P_ring + ΔP_substituent
    模型有两个输出头:
        - ring_head: 只用环原子特征 (position_encoding 中 is_ring_atom=1 的原子) 预测环本身贡献 P_ring
        - substituent_head: 只用取代基原子特征 (is_substituent=1 的原子) 预测取代基扰动 ΔP_sub
    最终预测: P_total = P_ring + ΔP_sub

    训练时使用 get_position_encoding 产生的 6 维位置编码区分环原子和取代基原子。

核心组件:
    - add_position_encodings(data): 数据预处理, 生成位置编码矩阵
    - PerturbationGNN: 共享 GNN encoder + ring_pool (环原子注意力池化) + sub_pool
      (取代基原子注意力池化) + ring_head + sub_head
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
import torch.nn as nn
import torch.nn.functional as F

from layer4_substituent.code.hammett_constants import get_position_encoding
from unified_models.gnn.model import GNNModel

# 位置编码维度 (与 hammett_constants.get_position_encoding 一致)
POS_DIM = 6


# ============== 数据预处理 ==============
def add_position_encodings(data, max_atoms=75, device='cpu'):
    """为每个分子生成取代基位置编码矩阵

    Args:
        data: load_adj_format 返回的字典, 需包含 'smiles'
        max_atoms: 最大原子数 (默认 75)
        device: 输出张量所在设备

    Returns:
        更新后的 data 字典, 新增字段:
            - 'pos_encs': torch.Tensor (N, max_atoms, POS_DIM)
            - 'pos_dim': POS_DIM
    """
    from rdkit import Chem

    smiles = data['smiles']
    n = len(smiles)
    pos_encs = np.zeros((n, max_atoms, POS_DIM), dtype=np.float32)

    for i in range(n):
        mol = Chem.MolFromSmiles(smiles[i])
        pe = get_position_encoding(mol, None, max_atoms)  # (max_atoms, POS_DIM)
        pos_encs[i] = pe[:max_atoms, :POS_DIM]

    pos_tensor = torch.tensor(pos_encs, dtype=torch.float32, device=device)
    new_data = dict(data)
    new_data['pos_encs'] = pos_tensor
    new_data['pos_dim'] = POS_DIM
    return new_data


# ============== 掩码注意力池化 ==============
class _MaskedAttentionPool(nn.Module):
    """对指定类型原子做带掩码的注意力池化

    type_flag_index 指定 pos_enc 中的列索引:
        0 → is_ring_atom (环原子)
        4 → is_substituent (取代基原子)
    """

    def __init__(self, hidden_dim, pos_dim=POS_DIM):
        super().__init__()
        # 拼接位置编码增强可区分性
        self.score = nn.Linear(hidden_dim + pos_dim, 1)

    def forward(self, node_emb, pos_enc, atom_mask, type_flag_index):
        # node_emb: (B, N, H), pos_enc: (B, N, P), atom_mask: (B, N)
        B, N, H = node_emb.shape
        combined = torch.cat([node_emb, pos_enc], dim=-1)  # (B, N, H+P)
        scores = self.score(combined).squeeze(-1)  # (B, N)

        type_mask = pos_enc[..., type_flag_index]
        valid = atom_mask * type_mask
        has_any = valid.any(dim=1).float()  # (B,)

        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid.bool(), neg_inf)
        attn = F.softmax(masked_scores, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0).unsqueeze(-1)  # (B, N, 1)
        repr_ = (node_emb * attn).sum(dim=1)  # (B, H)

        # 退化兜底: 无该类型原子时用全局有效均值 (避免 NaN)
        global_mean = (node_emb * atom_mask.unsqueeze(-1)).sum(dim=1) / \
                      (atom_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, H)
        repr_ = torch.where(has_any.unsqueeze(-1).bool(), repr_, global_mean)
        return repr_


# ============== 模型 ==============
class PerturbationGNN(GNNModel):
    """取代基扰动学习 GNN: P_total = P_ring + ΔP_sub

    共享 GNN encoder (init_transform + conv_layers), 之后:
        - ring_pool: 对环原子 (is_ring_atom=1) 做注意力池化 → ring_repr
        - sub_pool:  对取代基原子 (is_substituent=1) 做注意力池化 → sub_repr
        - ring_head: Linear(hidden, 1) → P_ring
        - sub_head:  Linear(hidden, 1) → ΔP_sub
        - 输出: P_ring + ΔP_sub

    forward 额外参数 (通过 kwargs):
        pos_enc:    (B, N, POS_DIM) 位置编码 (必需, 由 add_position_encodings 预计算)
        atom_mask:  (B, N) 有效原子掩码 (可选, 默认从 node_mat 推断)
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label', pos_dim=POS_DIM):
        super().__init__(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                         p_dropout, mode)
        self.pos_dim = pos_dim
        self.ring_pool = _MaskedAttentionPool(hidden_dim, pos_dim)
        self.sub_pool = _MaskedAttentionPool(hidden_dim, pos_dim)
        # 双输出头 (取代 GNNModel 的 output_layer)
        self.ring_head = nn.Linear(hidden_dim, 1)
        self.sub_head = nn.Linear(hidden_dim, 1)

    def _derive_atom_mask(self, node_mat):
        """从 node_mat 推断有效原子掩码: padding 行全为 0"""
        return (node_mat.abs().sum(dim=-1) > 0).float()

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None,
                pos_enc=None, atom_mask=None):
        batch_size, n_atoms, _ = node_mat.shape

        # === 共享 GNN encoder: init_transform + conv_layers ===
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        if self.mode == 'mask' and mask_mat is not None:
            if mask_mat.shape[-1] != node_fea.shape[-1]:
                mask_mat = self.mask_proj(mask_mat.float())
                mask_mat = (mask_mat > 0).float()
            node_fea = node_fea * mask_mat

        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        # === 位置编码 + 掩码 ===
        if atom_mask is None:
            atom_mask = self._derive_atom_mask(node_mat)
        if pos_enc is None:
            # 未提供位置编码时退化为零向量
            pos_enc = torch.zeros(batch_size, n_atoms, self.pos_dim,
                                  device=node_mat.device)

        # === 双输出头: P_total = P_ring + ΔP_sub ===
        ring_repr = self.ring_pool(node_fea, pos_enc, atom_mask, type_flag_index=0)
        sub_repr = self.sub_pool(node_fea, pos_enc, atom_mask, type_flag_index=4)
        ring_pred = self.ring_head(ring_repr)  # (B, 1)
        sub_pred = self.sub_head(sub_repr)     # (B, 1)
        return ring_pred + sub_pred            # (B, 1)


# ============== 构建模型 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=2, n_outputs=1, p_dropout=0.2, mode='label',
                pos_dim=POS_DIM, **kwargs):
    """返回 Method 7 模型实例

    Args:
        base: 目前支持 'gnn' (基于 GNNModel)
        node_vec_len: 节点特征长度 (默认 60)
        pos_dim: 位置编码维度 (默认 6)
        其余参数同 GNNModel
    """
    base = base.lower()
    if base == 'gnn':
        return PerturbationGNN(node_vec_len, hidden_dim, n_conv, n_hidden,
                               n_outputs, p_dropout, mode, pos_dim)
    else:
        raise ValueError(f"Method 7 currently supports base='gnn' only, got: {base}")


if __name__ == '__main__':
    # 自测: 实例化并前向传播
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_atoms, nvl = 75, 60
    model = build_model(node_vec_len=nvl, hidden_dim=64, n_conv=2, n_hidden=1,
                        n_outputs=1, mode='label').to(dev)
    node_mat = torch.randn(4, n_atoms, nvl, device=dev)
    # 模拟有效原子 (前 40 个有效)
    node_mat[:, 40:, :] = 0.0
    adj_mat = torch.eye(n_atoms, device=dev).unsqueeze(0).expand(4, -1, -1).clone()
    # 模拟位置编码: 前6个为环原子, 6-11为取代基原子
    pos_enc = torch.zeros(4, n_atoms, POS_DIM, device=dev)
    pos_enc[:, :6, 0] = 1.0       # is_ring_atom
    pos_enc[:, 6:12, 4] = 1.0     # is_substituent
    out = model(node_mat, adj_mat, pos_enc=pos_enc)
    print(f"[m7] PerturbationGNN output shape: {out.shape} (expected: [4, 1])")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    print("[m7] OK")
