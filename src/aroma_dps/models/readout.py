"""
Ring-Level Readout 模块

支持三种 readout 模式:
  - 'fixed_avg':   固定目标环原子平均聚合 (Stage1 Base, 无可学习参数)
  - 'attention':   可学习注意力聚合 (Stage2 Learnable Ring-Level Readout)
  - 'global_mean': 全局平均池化 (对照, 不使用目标环信息)

设计要点:
  - fixed_avg: 纯目标环原子特征均值, 不混入全局特征, 保证 (G,R)->A_R 任务一致性
  - attention: 对目标环原子做 learnable attention pooling, 替代固定平均
  - global_mean: 所有原子均值, 不使用 ring_indices
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedRingAvgReadout(nn.Module):
    """固定目标环原子平均聚合 (无可学习参数)

    pooled = mean(node_fea[i] for i in ring_indices if i >= 0)
    """
    def __init__(self):
        super().__init__()

    def forward(self, node_fea, ring_indices):
        """
        Args:
            node_fea:     (batch, n_atoms, hidden_dim)
            ring_indices: (batch, n_atoms) with -1 for non-ring atoms
        Returns:
            pooled: (batch, hidden_dim)
        """
        mask = (ring_indices >= 0).float().unsqueeze(-1)  # (batch, n_atoms, 1)
        ring_fea = (node_fea * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return ring_fea


class AttentionRingReadout(nn.Module):
    """可学习注意力环级聚合 (Learnable Ring-Level Readout)

    对目标环原子计算 attention 权重后加权平均:
      alpha_i = softmax(w^T * tanh(W * h_i))  over ring atoms
      pooled = sum(alpha_i * h_i)

    核心区别 vs fixed_avg:
      - fixed_avg: 所有环原子等权 (1/n_ring)
      - attention: 环原子权重可学习, 能突出任务相关原子
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_fea, ring_indices):
        """
        Args:
            node_fea:     (batch, n_atoms, hidden_dim)
            ring_indices: (batch, n_atoms) with -1 for non-ring atoms
        Returns:
            pooled: (batch, hidden_dim)
        """
        mask = (ring_indices >= 0).float().unsqueeze(-1)  # (batch, n_atoms, 1)

        # 注意力分数 (仅对环原子计算)
        proj = torch.tanh(self.attn_proj(node_fea))  # (batch, n_atoms, hidden_dim)
        scores = self.attn_score(proj)  # (batch, n_atoms, 1)
        # 非环原子 mask 为 -1e9, softmax 后权重为 0
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)  # (batch, n_atoms, 1)
        attn_weights = self.dropout(attn_weights)
        pooled = (node_fea * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class GlobalMeanReadout(nn.Module):
    """全局平均池化 (对照, 不使用目标环信息)"""
    def __init__(self):
        super().__init__()

    def forward(self, node_fea, ring_indices=None):
        return node_fea.mean(dim=1)


def build_readout(mode, hidden_dim, dropout=0.2):
    """工厂函数"""
    if mode == 'fixed_avg':
        return FixedRingAvgReadout()
    elif mode == 'attention':
        return AttentionRingReadout(hidden_dim, dropout)
    elif mode == 'global_mean':
        return GlobalMeanReadout()
    else:
        raise ValueError(f"未知 readout 模式: {mode}, 可选: fixed_avg / attention / global_mean")
