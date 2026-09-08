"""模型架构定义（自包含）

复刻 0831-end-code 的 RingConditionedGNN + MPNN backbone + FixedRingAvgReadout，
并内建可选的 ring_proj (nn.Embedding(2, hidden)) 注入逻辑，便于独立调用。

与 stage6_final_membership 的最优配置完全一致:
  - MPNN backbone, 3 conv layers, hidden=128, 2 hidden MLP layers
  - readout_mode='fixed_avg' (目标环原子均值池化)
  - HOMA:       use_projection=True  (ring_flag=1 + nn.Embedding 投影)
  - NICS_1zz:   use_projection=False (ring_flag=1)
  - MBCO:       use_projection=False (ring_flag=1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """MPNN 消息传递层 (GRU-style 更新), 与 unified_models/mpnn/model.py 一致"""

    def __init__(self, hidden_dim, p_dropout=0.2):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.update_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_r = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_h = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, adj):
        # x: (batch, n_atoms, hidden), adj: (batch, n_atoms, n_atoms)
        neigh_sum = torch.matmul(adj, x)  # 邻居求和
        messages = self.message_mlp(torch.cat([x, neigh_sum], dim=-1))
        z = torch.sigmoid(self.update_z(torch.cat([x, messages], dim=-1)))
        r = torch.sigmoid(self.update_r(torch.cat([x, messages], dim=-1)))
        h_tilde = torch.tanh(self.update_h(torch.cat([x * r, messages], dim=-1)))
        return (1 - z) * x + z * h_tilde


class FixedRingAvgReadout(nn.Module):
    """固定目标环原子平均聚合"""

    def forward(self, node_fea, ring_indices):
        mask = (ring_indices >= 0).float().unsqueeze(-1).to(node_fea.device)
        return (node_fea * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)


class RingConditionedMPNN(nn.Module):
    """Ring-Conditioned MPNN 最终模型

    Args:
        node_vec_len:   基础节点特征维度 (60)
        hidden_dim:     隐藏维度 (128)
        n_conv:         卷积层数 (3)
        n_hidden:       MLP 隐藏层数 (2)
        p_dropout:      dropout (0.2)
        use_projection: 是否使用 ring_proj (nn.Embedding) 投影注入
        ring_flag_value: 目标环标记注入值 (1)
    """

    def __init__(self, node_vec_len=60, hidden_dim=128, n_conv=3, n_hidden=2,
                 p_dropout=0.2, use_projection=False, ring_flag_value=1):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.use_projection = use_projection
        self.ring_flag_value = ring_flag_value

        # ring_proj 投影注入时输入维度增大
        in_dim = node_vec_len + (hidden_dim if use_projection else 0)
        self.init_transform = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        if use_projection:
            self.ring_proj = nn.Embedding(2, hidden_dim)
        else:
            self.ring_proj = None

        self.conv_layers = nn.ModuleList([MPNNLayer(hidden_dim, p_dropout)
                                          for _ in range(n_conv)])
        self.conv_bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)])

        self.readout = FixedRingAvgReadout()
        self.pooling_activation = nn.LeakyReLU(0.2)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                            for _ in range(n_hidden)])
        self.hidden_bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_hidden)])

        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _maybe_inject(self, x, ring_indices):
        """ring_proj 投影注入: 将二值环成员标记映射到 hidden 维并与节点特征 concat"""
        if self.use_projection and self.ring_proj is not None:
            bin_ri = (ring_indices >= 0).long().to(x.device)
            emb = self.ring_proj(bin_ri)
            x = torch.cat([x, emb], dim=-1)
        return x

    def forward(self, node_mat, adj_mat, ring_indices=None):
        batch_size, n_atoms, _ = node_mat.shape
        node_mat = self._maybe_inject(node_mat, ring_indices)
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len + (self.hidden_dim if self.use_projection else 0))
        ).reshape(batch_size, n_atoms, -1)

        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = self.conv_bns[i](node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        pooled = self.readout(node_fea, ring_indices)
        pooled = self.pooling_activation(pooled)

        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)


def build_model(use_projection=False, node_vec_len=60, hidden_dim=128,
                n_conv=3, n_hidden=2, p_dropout=0.2, ring_flag_value=1):
    return RingConditionedMPNN(
        node_vec_len=node_vec_len, hidden_dim=hidden_dim,
        n_conv=n_conv, n_hidden=n_hidden, p_dropout=p_dropout,
        use_projection=use_projection, ring_flag_value=ring_flag_value,
    )
