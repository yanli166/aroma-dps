"""
PyG 模型 (D-MPNN) — 支持环级 readout 的版本

D-MPNN: 基于 PyG scatter 的实现 (Yang et al. 2019)
扩展: 支持固定环平均池化 / 注意力环聚合 / 全局平均池化

节点特征复用原始 Graph 的 node_mat, 保证与自定义模型公平对比。
ring_mask (在 Data 中) 标记目标环原子, 供 ring-level readout 使用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import scatter, softmax

from aroma_dps.models.readout import build_readout


class DMPNNModel(nn.Module):
    """D-MPNN 主模型 (Yang et al. 2019) + 可配置 ring-level readout

    流程:
      1. h_v^0 = ReLU(W_i x_v)
      2. m_{uv}^0 = ReLU(W_h [h_u^0, e_{uv}])
      3. T 步消息传递: m_{uv}^{t+1} = ReLU(W_t [h_v, sum_{w!=u} m_{wv}, e_{uv}])
      4. h_v = ReLU(W_o [h_v^0, sum_w m_{wv}])
      5. Ring-level readout (fixed_avg / attention / global_mean) -> MLP -> 预测
    """

    def __init__(self, in_channels, hidden_channels=128, num_layers=3,
                 dropout=0.2, edge_dim=4, out_channels=1,
                 readout_mode='fixed_avg'):
        super().__init__()
        self.hidden_dim = hidden_channels
        self.readout_mode = readout_mode
        self.in_lin = nn.Linear(in_channels, hidden_channels)
        self.edge_init = nn.Linear(edge_dim + hidden_channels, hidden_channels)
        self.msg_layers = nn.ModuleList([
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_channels * 2, hidden_channels)
        self.drop = nn.Dropout(dropout)

        # Ring-level readout
        self.readout = build_readout(readout_mode, hidden_channels, dropout)
        self.pooling_activation = nn.LeakyReLU(0.2)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, ring_mask=None):
        """
        Args:
            x:         (n_atoms_total, in_channels)
            edge_index:(2, n_edges_total)
            batch:     (n_atoms_total,) 图索引
            edge_attr: (n_edges_total, edge_dim)
            ring_mask: (n_atoms_total,) 1.0=目标环原子, 0.0=非环原子
        """
        row, col = edge_index
        h0 = F.relu(self.in_lin(x))
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 4), device=x.device)

        edge_messages = F.relu(self.edge_init(torch.cat([h0[row], edge_attr], dim=-1)))

        for layer in self.msg_layers:
            node_agg = scatter(edge_messages, col, dim=0, dim_size=h0.size(0), reduce='sum')
            rev_sub = node_agg[col] - edge_messages
            new_msg = F.relu(layer(torch.cat([h0[col], rev_sub, edge_attr], dim=-1)))
            edge_messages = self.drop(new_msg)

        node_msg_agg = scatter(edge_messages, col, dim=0, dim_size=h0.size(0), reduce='sum')
        h_final = F.relu(self.out_lin(torch.cat([h0, node_msg_agg], dim=-1)))
        h_final = self.drop(h_final)

        # 将 ring_mask 转为 ring_indices 格式 (供 readout 使用)
        # ring_indices: >=0 表示环原子, -1 表示非环原子
        if ring_mask is not None:
            ring_indices = ring_mask.long()  # 1=环, 0=非环; 需转为 -1/0 格式
            ring_indices = torch.where(ring_indices > 0, ring_indices - 1, torch.tensor(-1, device=x.device))
        else:
            ring_indices = torch.full((h_final.size(0),), -1, dtype=torch.long, device=x.device)

        # 按图分组做 ring-level readout
        # 需要将 (n_atoms_total, hidden) 按 batch 分组, 每组做 readout
        graph_emb = self._batched_ring_readout(h_final, ring_indices, batch)
        graph_emb = self.pooling_activation(graph_emb)
        return self.head(graph_emb)

    def _batched_ring_readout(self, node_fea, ring_indices, batch):
        """按图分组执行 ring-level readout

        Args:
            node_fea:     (n_atoms_total, hidden_dim)
            ring_indices: (n_atoms_total,) -1=非环原子
            batch:        (n_atoms_total,) 图索引
        Returns:
            graph_emb: (num_graphs, hidden_dim)
        """
        num_graphs = batch.max().item() + 1
        graph_embs = []
        for g in range(num_graphs):
            mask = (batch == g)
            nf = node_fea[mask].unsqueeze(0)  # (1, n_atoms_g, hidden)
            ri = ring_indices[mask].unsqueeze(0)  # (1, n_atoms_g)
            emb = self.readout(nf, ri)  # (1, hidden)
            graph_embs.append(emb)
        return torch.cat(graph_embs, dim=0)


def build_pyg_model(name, in_channels, hidden_dim=128, n_layers=3,
                    dropout=0.2, edge_dim=4, readout_mode='fixed_avg'):
    """工厂函数"""
    if name == 'DMPNN':
        return DMPNNModel(in_channels, hidden_dim, n_layers,
                          dropout=dropout, edge_dim=edge_dim, out_channels=1,
                          readout_mode=readout_mode)
    raise ValueError(f"未知 PyG 模型: {name}")
