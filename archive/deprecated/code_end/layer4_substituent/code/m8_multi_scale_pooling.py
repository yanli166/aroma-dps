"""
Method 8: 多尺度取代基池化 (聚合层)

思路:
    在聚合层实现多尺度池化, 与环池化融合。三个尺度的池化:
        1. 原子级 (AtomPool):        标准全局池化 (mean)
        2. 取代基级 (SubstituentPool): 对每个取代基做局部池化, 再聚合
        3. 环级 (RingPool):          对环原子做注意力池化
    多尺度融合: 可学习的门控融合
        gate = sigmoid(W · cat(atom, sub, ring))
        fused = proj(gate ⊙ cat(atom, sub, ring))

核心组件:
    - add_position_encodings(data): 数据预处理 (位置编码, 用于区分环原子)
    - add_substituent_groups(data): 数据预处理, 为每个原子计算取代基组 ID
    - MultiScalePoolGNN: 共享 GNN encoder + AtomPool + SubstituentPool + RingPool + FusionLayer
    - build_model: 返回模型实例
"""
import os
import sys
from collections import deque


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
from rdkit import Chem

from layer4_substituent.code.hammett_constants import get_position_encoding
from unified_models.gnn.model import GNNModel

# 位置编码维度 (与 hammett_constants.get_position_encoding 一致)
POS_DIM = 6
# 取代基组数的上限 (用于 padding / one-hot)
MAX_GROUPS = 20


# ============== 数据预处理 ==============
def add_position_encodings(data, max_atoms=75, device='cpu'):
    """为每个分子生成位置编码矩阵 (与 Method 3/7 一致)

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
        pe = get_position_encoding(mol, None, max_atoms)
        pos_encs[i] = pe[:max_atoms, :POS_DIM]

    pos_tensor = torch.tensor(pos_encs, dtype=torch.float32, device=device)
    new_data = dict(data)
    new_data['pos_encs'] = pos_tensor
    new_data['pos_dim'] = POS_DIM
    return new_data


def compute_substituent_groups(mol, max_atoms=75):
    """为单个分子计算每个原子的取代基组 ID

    组 ID 含义:
        0           = 环原子 / 非取代基原子 (含直接连在环上的 H)
        1, 2, ..., K = 第 K 个取代基组 (BFS 从环上非 H 取代基起始原子出发, 覆盖
                      所有连通的非环原子, 包含该取代基上的 H)

    Args:
        mol: RDKit Mol (未加氢; 内部统一 AddHs 以匹配 node_mat 索引)
        max_atoms: 最大原子数

    Returns:
        np.ndarray: (max_atoms,) int64
    """
    groups = np.zeros(max_atoms, dtype=np.int64)
    if mol is None:
        return groups

    mol_h = Chem.AddHs(mol)
    ring_info = mol_h.GetRingInfo()
    aromatic_rings = [r for r in ring_info.AtomRings()
                      if all(mol_h.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not aromatic_rings:
        return groups
    ring_atom_indices = set()
    for r in aromatic_rings:
        ring_atom_indices.update(r)

    visited = set()
    group_id = 0
    for ring_idx in sorted(ring_atom_indices):
        ring_atom = mol_h.GetAtomWithIdx(ring_idx)
        for nb in ring_atom.GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in ring_atom_indices or nidx in visited:
                continue
            if nb.GetSymbol() == 'H':
                # 直接连在环上的 H 不算取代基
                visited.add(nidx)
                continue
            # 新取代基组: BFS 覆盖该取代基的所有非环原子 (含其上的 H)
            group_id += 1
            queue = deque([nidx])
            visited.add(nidx)
            while queue:
                cur = queue.popleft()
                if cur < max_atoms:
                    groups[cur] = group_id
                for nb2 in mol_h.GetAtomWithIdx(cur).GetNeighbors():
                    if nb2.GetIdx() not in ring_atom_indices and nb2.GetIdx() not in visited:
                        visited.add(nb2.GetIdx())
                        queue.append(nb2.GetIdx())
    return groups


def add_substituent_groups(data, max_atoms=75, max_groups=MAX_GROUPS, device='cpu'):
    """为整个数据集计算取代基组 ID 张量

    Args:
        data: load_adj_format 返回的字典, 需包含 'smiles'
        max_atoms: 最大原子数 (默认 75)
        max_groups: 取代基组数上限 (用于 padding)
        device: 输出张量所在设备

    Returns:
        更新后的 data 字典, 新增字段:
            - 'sub_groups': torch.LongTensor (N, max_atoms), 值域 [0, max_groups)
            - 'max_groups': max_groups
    """
    from rdkit import Chem

    smiles = data['smiles']
    n = len(smiles)
    groups = np.zeros((n, max_atoms), dtype=np.int64)
    for i in range(n):
        mol = Chem.MolFromSmiles(smiles[i])
        g = compute_substituent_groups(mol, max_atoms)
        groups[i] = np.clip(g, 0, max_groups - 1)

    new_data = dict(data)
    new_data['sub_groups'] = torch.tensor(groups, dtype=torch.long, device=device)
    new_data['max_groups'] = max_groups
    return new_data


# ============== 多尺度池化模块 ==============
class _SubstituentPool(nn.Module):
    """取代基级池化: 对每个取代基组做局部均值池化, 再对组间做注意力聚合

    输入:
        node_emb:   (B, N, H)
        sub_groups: (B, N) long, 0=非取代基, 1..K=取代基组
        atom_mask:  (B, N)
    输出:
        sub_repr:   (B, H)
    """

    def __init__(self, hidden_dim, max_groups=MAX_GROUPS):
        super().__init__()
        self.max_groups = max_groups
        # 组间注意力打分
        self.group_attn = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb, sub_groups, atom_mask):
        B, N, H = node_emb.shape
        G = self.max_groups

        # 组 one-hot: (B, N, G), 仅对组 ID > 0 的有效原子置 1
        group_oh = F.one_hot(sub_groups.clamp(0, G - 1), G).float()  # (B, N, G)
        is_sub = (sub_groups > 0).float()                            # (B, N)
        group_oh = group_oh * is_sub.unsqueeze(-1) * atom_mask.unsqueeze(-1)  # (B, N, G)

        # 组内均值池化: group_repr[g] = sum(node_emb * mask) / count
        group_counts = group_oh.sum(dim=1).clamp(min=1.0)            # (B, G)
        group_sums = torch.einsum('bnh,bng->bgh', node_emb, group_oh)  # (B, G, H)
        group_repr = group_sums / group_counts.unsqueeze(-1)          # (B, G, H)

        # 组间注意力聚合
        group_scores = self.group_attn(group_repr).squeeze(-1)        # (B, G)
        valid_groups = (group_counts > 0)                             # (B, G)
        neg_inf = torch.finfo(group_scores.dtype).min
        group_scores = group_scores.masked_fill(~valid_groups, neg_inf)
        group_attn = F.softmax(group_scores, dim=1).unsqueeze(-1)     # (B, G, 1)
        group_attn = torch.nan_to_num(group_attn, nan=0.0)
        sub_repr = (group_repr * group_attn).sum(dim=1)               # (B, H)

        # 退化兜底: 无取代基组时用全局有效均值 (避免 NaN)
        has_sub = valid_groups.any(dim=1).float()                    # (B,)
        global_mean = (node_emb * atom_mask.unsqueeze(-1)).sum(dim=1) / \
                      (atom_mask.sum(dim=1, keepdim=True) + 1e-8)     # (B, H)
        sub_repr = torch.where(has_sub.unsqueeze(-1).bool(), sub_repr, global_mean)
        return sub_repr


class _RingPool(nn.Module):
    """环级池化: 对环原子做注意力池化"""

    def __init__(self, hidden_dim, pos_dim=POS_DIM):
        super().__init__()
        self.score = nn.Linear(hidden_dim + pos_dim, 1)

    def forward(self, node_emb, pos_enc, atom_mask):
        combined = torch.cat([node_emb, pos_enc], dim=-1)         # (B, N, H+P)
        scores = self.score(combined).squeeze(-1)                  # (B, N)
        is_ring = pos_enc[..., 0]                                  # is_ring_atom
        valid = atom_mask * is_ring
        has_any = valid.any(dim=1).float()
        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid.bool(), neg_inf)
        attn = F.softmax(masked_scores, dim=1).unsqueeze(-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        repr_ = (node_emb * attn).sum(dim=1)                       # (B, H)
        global_mean = (node_emb * atom_mask.unsqueeze(-1)).sum(dim=1) / \
                      (atom_mask.sum(dim=1, keepdim=True) + 1e-8)
        repr_ = torch.where(has_any.unsqueeze(-1).bool(), repr_, global_mean)
        return repr_


class _FusionLayer(nn.Module):
    """可学习门控融合三个尺度表示

    gate = sigmoid(W · cat(atom, sub, ring))
    fused = proj(gate ⊙ cat(atom, sub, ring))
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, atom_repr, sub_repr, ring_repr):
        cat = torch.cat([atom_repr, sub_repr, ring_repr], dim=-1)  # (B, 3H)
        gate = torch.sigmoid(self.gate(cat))                       # (B, 3H)
        fused = self.proj(gate * cat)                              # (B, H)
        return fused


# ============== 模型 ==============
class MultiScalePoolGNN(GNNModel):
    """多尺度取代基池化 GNN

    共享 GNN encoder (init_transform + conv_layers), 之后三尺度池化融合:
        - AtomPool:        全局 mean pooling (原子级)
        - SubstituentPool: 按取代基组局部池化 + 组间注意力 (取代基级)
        - RingPool:        环原子注意力池化 (环级)
        - FusionLayer:     可学习门控融合

    forward 额外参数 (通过 kwargs):
        pos_enc:    (B, N, POS_DIM) 位置编码 (用于 RingPool)
        sub_groups: (B, N) 取代基组 ID (用于 SubstituentPool)
        atom_mask:  (B, N) 有效原子掩码 (可选, 默认从 node_mat 推断)
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label', pos_dim=POS_DIM,
                 max_groups=MAX_GROUPS):
        super().__init__(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                         p_dropout, mode)
        self.pos_dim = pos_dim
        self.max_groups = max_groups
        self.sub_pool = _SubstituentPool(hidden_dim, max_groups)
        self.ring_pool = _RingPool(hidden_dim, pos_dim)
        self.fusion = _FusionLayer(hidden_dim)

    def _derive_atom_mask(self, node_mat):
        """从 node_mat 推断有效原子掩码: padding 行全为 0"""
        return (node_mat.abs().sum(dim=-1) > 0).float()

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None,
                pos_enc=None, sub_groups=None, atom_mask=None):
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

        # === 掩码 & 位置编码 & 取代基组 ===
        if atom_mask is None:
            atom_mask = self._derive_atom_mask(node_mat)
        if pos_enc is None:
            pos_enc = torch.zeros(batch_size, n_atoms, self.pos_dim,
                                  device=node_mat.device)
        if sub_groups is None:
            sub_groups = torch.zeros(batch_size, n_atoms, dtype=torch.long,
                                     device=node_mat.device)

        # === 三尺度池化 ===
        # 1. 原子级: 全局 mean pooling
        atom_repr = (node_fea * atom_mask.unsqueeze(-1)).sum(dim=1) / \
                    (atom_mask.sum(dim=1, keepdim=True) + 1e-8)        # (B, H)
        # 2. 取代基级: 按组局部池化 + 组间注意力
        sub_repr = self.sub_pool(node_fea, sub_groups, atom_mask)      # (B, H)
        # 3. 环级: 环原子注意力池化
        ring_repr = self.ring_pool(node_fea, pos_enc, atom_mask)       # (B, H)

        # === 门控融合 ===
        pooled = self.fusion(atom_repr, sub_repr, ring_repr)          # (B, H)
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
                pos_dim=POS_DIM, max_groups=MAX_GROUPS, **kwargs):
    """返回 Method 8 模型实例

    Args:
        base: 目前支持 'gnn' (基于 GNNModel)
        node_vec_len: 节点特征长度 (默认 60)
        pos_dim: 位置编码维度 (默认 6)
        max_groups: 取代基组数上限 (默认 20)
        其余参数同 GNNModel
    """
    base = base.lower()
    if base == 'gnn':
        return MultiScalePoolGNN(node_vec_len, hidden_dim, n_conv, n_hidden,
                                 n_outputs, p_dropout, mode, pos_dim, max_groups)
    else:
        raise ValueError(f"Method 8 currently supports base='gnn' only, got: {base}")


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
    # 模拟取代基组 ID: 6-8 为组 1, 9-11 为组 2
    sub_groups = torch.zeros(4, n_atoms, dtype=torch.long, device=dev)
    sub_groups[:, 6:9] = 1
    sub_groups[:, 9:12] = 2
    out = model(node_mat, adj_mat, pos_enc=pos_enc, sub_groups=sub_groups)
    print(f"[m8] MultiScalePoolGNN output shape: {out.shape} (expected: [4, 1])")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    print("[m8] OK")
