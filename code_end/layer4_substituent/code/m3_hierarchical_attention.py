"""
Method 3: 分层交叉注意力 (聚合层)

思路:
    在 GNN 消息传递完成后、全局池化之前, 加入分层注意力机制替代简单的 mean pooling。
        第一层 (官能团级): 分别对 取代基原子 (substituent atoms) 和 环原子 (ring atoms)
                          做加权注意力, 聚合得到两个子表示。
        第二层 (分子级):   将两个子表示融合得到最终分子表示。
    使用 get_position_encoding 产生的 6 维位置编码区分环原子与取代基原子:
        [is_ring_atom, is_ortho, is_meta, is_para, is_substituent, ring_pos]

核心组件:
    - add_position_encodings(data): 数据预处理, 为每个分子生成位置编码矩阵
    - HierarchicalAttention: 分层注意力模块
    - HierarchicalAttentionGNN: 集成分层注意力的 GNN (继承 GNNModel, 覆盖 forward)
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


# ============== 分层注意力模块 ==============
class HierarchicalAttention(nn.Module):
    """分层交叉注意力模块

    输入:
        node_emb:   (B, N, H)   消息传递后的节点嵌入
        pos_enc:    (B, N, P)   位置编码 [is_ring, ortho, meta, para, is_sub, ring_pos]
        atom_mask:  (B, N)      有效原子掩码 (1=有效, 0=padding)

    流程:
        1. 官能团级注意力:
           - 对取代基原子 (pos_enc[...,4]==1) 计算注意力权重, 加权求和 → sub_repr
           - 对环原子 (pos_enc[...,0]==1) 计算注意力权重, 加权求和 → ring_repr
           - 若分子无取代基原子, sub_repr 退化为全局平均 (避免 NaN)
        2. 分子级融合:
           - fused = fusion(cat[sub_repr, ring_repr])
    """

    def __init__(self, hidden_dim, pos_dim=POS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        # 取代基原子注意力打分 (拼接位置编码增强可区分性)
        self.sub_score = nn.Linear(hidden_dim + pos_dim, 1)
        # 环原子注意力打分
        self.ring_score = nn.Linear(hidden_dim + pos_dim, 1)
        # 分子级融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

    def _masked_softmax_attn(self, scores, valid_mask, type_mask):
        """对指定类型原子做带掩码的 softmax 注意力

        Args:
            scores: (B, N) 原始注意力分数
            valid_mask: (B, N) 有效原子 (排除 padding)
            type_mask: (B, N) 指定类型原子 (取代基/环)

        Returns:
            attn: (B, N, 1) 注意力权重 (非指定/无效位置为 0)
            has_any: (B,) 该类型原子是否存在
        """
        combined_mask = (valid_mask * type_mask).bool()  # (B, N)
        has_any = combined_mask.any(dim=1).float()  # (B,)
        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~combined_mask, neg_inf)
        # 对没有任何有效原子的样本, softmax 会产生 NaN; 用 where 兜底
        attn = F.softmax(masked_scores, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return attn.unsqueeze(-1), has_any

    def forward(self, node_emb, pos_enc, atom_mask=None):
        # node_emb: (B, N, H), pos_enc: (B, N, P), atom_mask: (B, N)
        B, N, H = node_emb.shape
        if atom_mask is None:
            # 默认所有原子有效
            atom_mask = torch.ones(B, N, device=node_emb.device)

        combined = torch.cat([node_emb, pos_enc], dim=-1)  # (B, N, H+P)

        # 类型掩码
        is_ring = pos_enc[..., 0]      # is_ring_atom
        is_sub = pos_enc[..., 4]       # is_substituent
        valid = atom_mask.float()

        # 取代基原子注意力
        sub_scores = self.sub_score(combined).squeeze(-1)  # (B, N)
        sub_attn, has_sub = self._masked_softmax_attn(sub_scores, valid, is_sub)
        sub_repr = (node_emb * sub_attn).sum(dim=1)        # (B, H)

        # 环原子注意力
        ring_scores = self.ring_score(combined).squeeze(-1)
        ring_attn, has_ring = self._masked_softmax_attn(ring_scores, valid, is_ring)
        ring_repr = (node_emb * ring_attn).sum(dim=1)      # (B, H)

        # 退化兜底: 若某分子无取代基原子, sub_repr 用全局有效均值替代
        global_mean = (node_emb * valid.unsqueeze(-1)).sum(dim=1) / \
                      (valid.sum(dim=1, keepdim=True) + 1e-8)  # (B, H)
        sub_repr = torch.where(has_sub.unsqueeze(-1).bool(), sub_repr, global_mean)
        # 若无环原子, ring_repr 同样用全局均值兜底
        ring_repr = torch.where(has_ring.unsqueeze(-1).bool(), ring_repr, global_mean)

        # 分子级融合
        fused = self.fusion(torch.cat([sub_repr, ring_repr], dim=-1))  # (B, H)
        return fused


# ============== 模型 ==============
class HierarchicalAttentionGNN(GNNModel):
    """集成分层注意力的 GNN

    继承 GNNModel, 复用其 init_transform / conv_layers / hidden_layers / output_layer,
    仅覆盖 forward: 在消息传递后用 HierarchicalAttention 替代 mean pooling。

    forward 额外参数 (通过 kwargs):
        pos_enc:     (B, N, POS_DIM) 位置编码 (必需, 由 add_position_encodings 预计算)
        atom_mask:   (B, N) 有效原子掩码 (可选, 默认从 node_mat 推断)
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label', pos_dim=POS_DIM):
        super().__init__(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                         p_dropout, mode)
        self.pos_dim = pos_dim
        self.hier_attn = HierarchicalAttention(hidden_dim, pos_dim)

    def _derive_atom_mask(self, node_mat):
        """从 node_mat 推断有效原子掩码: padding 行全为 0"""
        return (node_mat.abs().sum(dim=-1) > 0).float()

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None,
                pos_enc=None, atom_mask=None):
        batch_size, n_atoms, _ = node_mat.shape

        # === 复用 GNNModel 的消息传递部分 (init_transform + conv) ===
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

        # === 分层注意力池化 (替代 mean pooling) ===
        if atom_mask is None:
            atom_mask = self._derive_atom_mask(node_mat)
        if pos_enc is None:
            # 未提供位置编码时退化为零向量 (注意力退化为普通注意力, 不再区分类型)
            pos_enc = torch.zeros(batch_size, n_atoms, self.pos_dim,
                                  device=node_mat.device)
        pooled = self.hier_attn(node_fea, pos_enc, atom_mask)
        pooled = self.pooling_activation(pooled)

        # === 复用 GNNModel 的 hidden + output ===
        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)


# ============== 构建模型 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=2, n_outputs=1, p_dropout=0.2, mode='label',
                pos_dim=POS_DIM, **kwargs):
    """返回 Method 3 模型实例

    Args:
        base: 目前支持 'gnn' (基于 GNNModel)
        node_vec_len: 节点特征长度 (默认 60)
        pos_dim: 位置编码维度 (默认 6)
        其余参数同 GNNModel
    """
    base = base.lower()
    if base == 'gnn':
        return HierarchicalAttentionGNN(node_vec_len, hidden_dim, n_conv, n_hidden,
                                        n_outputs, p_dropout, mode, pos_dim)
    else:
        raise ValueError(f"Method 3 currently supports base='gnn' only, got: {base}")


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
    # 模拟位置编码: 部分原子为环原子, 部分为取代基
    pos_enc = torch.zeros(4, n_atoms, POS_DIM, device=dev)
    pos_enc[:, :6, 0] = 1.0       # 前6个为环原子
    pos_enc[:, 6:12, 4] = 1.0     # 6-11为取代基原子
    out = model(node_mat, adj_mat, pos_enc=pos_enc)
    print(f"[m3] HierarchicalAttentionGNN output shape: {out.shape} (expected: [4, 1])")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    print("[m3] OK")
