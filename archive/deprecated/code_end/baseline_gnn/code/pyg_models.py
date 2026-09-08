"""
PyG 实现的 GNN 基线模型 (第二层)

- AttentiveFP: 基于 GATConv 复现 (Pushing the Boundaries, JCIM 2019)
  原因: torch_geometric.nn.AttentiveFP 在 PyG 2.8 + 本环境 edge_attr 处理存在维度异常,
  故用 GATConv (已验证 edge_dim 正常) 等价复现其核心结构: 门控注意力消息传递 + GRU 迭代 + 注意力池化。
- DMPNN: 基于 PyG scatter 的实现 (Yang et al. 2019)

节点特征复用原始 Graph 的 node_mat (与自定义模型一致), 保证公平对比。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import scatter, softmax


class AttentiveFPModel(nn.Module):
    """AttentiveFP 复现 (基于 GATConv)

    结构:
      1. 初始门控嵌入: GATConv(in -> hidden, edge_dim) + ELU
      2. T 步迭代: GATConv(hidden -> hidden) + GRU 更新 + LayerNorm
      3. 注意力池化: 节点上下文 -> softmax 归一化 -> 全局嵌入

    注意: 原实现用 sigmoid 做注意力归一化导致部分 fold 收敛失败 (R²≈0),
    已改为按图内节点 softmax 归一化 (与原始 AttentiveFP 论文一致), 并加入
    LayerNorm 稳定深层 GAT+GRU 训练。
    """

    def __init__(self, in_channels, hidden_channels=128, num_layers=3,
                 num_timesteps=2, dropout=0.2, edge_dim=4, out_channels=1, heads=4):
        super().__init__()
        self.hidden = hidden_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        # 初始门控层 (heads 路, concat)
        self.gate_conv = GATConv(in_channels, hidden_channels // heads, heads=heads,
                                 edge_dim=edge_dim, concat=True, dropout=dropout)
        # 迭代 GAT 层 (单头, 保持 hidden)
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, edge_dim=edge_dim,
                    concat=True, dropout=dropout)
            for _ in range(num_timesteps)
        ])
        # GRU 迭代更新
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        # LayerNorm 稳定深层 GAT+GRU 训练 (修复某些 fold 梯度爆炸/收敛失败)
        self.norm = nn.LayerNorm(hidden_channels)

        # 注意力池化
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # 跳过 self 避免无限递归; GATConv/Linear 自带 reset_parameters
        for m in self.children():
            if hasattr(m, 'reset_parameters'):
                try:
                    m.reset_parameters()
                except Exception:
                    pass

    def forward(self, x, edge_index, batch, edge_attr=None):
        # 1. 初始嵌入
        h = F.elu(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 2. 迭代消息传递 + GRU + LayerNorm
        for gat in self.gat_layers:
            m = F.elu(gat(h, edge_index, edge_attr))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
        h = self.norm(h)  # 稳定深层输出, 防止梯度爆炸/饱和

        # 3. 注意力池化 (context-aware, 与原始 AttentiveFP 一致使用 softmax)
        # 节点级 attention: g_i = tanh(W1 h_i) ; 与全局上下文做点积
        g = torch.tanh(self.lin1(h))
        global_ctx = global_add_pool(g, batch)  # (num_graphs, hidden)
        alpha = (g * self.lin2(global_ctx[batch])).sum(dim=-1, keepdim=True)
        # 按图内节点 softmax 归一化 (替代原先 sigmoid+二次归一化, 修复收敛失败)
        alpha = softmax(alpha, batch)
        graph_emb = global_add_pool(h * alpha, batch)

        # 输出
        out = F.dropout(graph_emb, p=self.dropout, training=self.training)
        return out  # (num_graphs, hidden), 由外层接 head; 这里直接返回隐藏表示


class AttentiveFPFull(nn.Module):
    """AttentiveFP + 输出 head, 返回 (num_graphs, out_channels)"""

    def __init__(self, in_channels, hidden_channels=128, num_layers=3,
                 num_timesteps=2, dropout=0.2, edge_dim=4, out_channels=1):
        super().__init__()
        self.encoder = AttentiveFPModel(in_channels, hidden_channels, num_layers,
                                        num_timesteps, dropout, edge_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        emb = self.encoder(x, edge_index, batch, edge_attr)
        return self.head(emb)


class DMPNNModel(nn.Module):
    """DMPNN 主模型 (Yang et al. 2019)

    流程:
      1. h_v^0 = ReLU(W_i x_v)
      2. m_{uv}^0 = ReLU(W_h [h_u^0, e_{uv}])
      3. T 步消息传递: m_{uv}^{t+1} = ReLU(W_t [h_v, sum_{w!=u} m_{wv}, e_{uv}])
      4. h_v = ReLU(W_o [h_v^0, sum_w m_{wv}])
      5. 全局池化 -> MLP -> 预测
    """

    def __init__(self, in_channels, hidden_channels=128, num_layers=3,
                 dropout=0.2, edge_dim=4, out_channels=1):
        super().__init__()
        self.hidden_dim = hidden_channels
        self.in_lin = nn.Linear(in_channels, hidden_channels)
        self.edge_init = nn.Linear(edge_dim + hidden_channels, hidden_channels)
        self.msg_layers = nn.ModuleList([
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_channels * 2, hidden_channels)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        row, col = edge_index
        h0 = F.relu(self.in_lin(x))
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 4), device=x.device)

        # 初始边消息 m_{uv}^0 = ReLU(W_h [h_u, e_{uv}])
        edge_messages = F.relu(self.edge_init(torch.cat([h0[row], edge_attr], dim=-1)))

        # T 步消息传递
        for layer in self.msg_layers:
            # 节点聚合: agg[v] = sum_u m_{uv}
            node_agg = scatter(edge_messages, col, dim=0, dim_size=h0.size(0), reduce='sum')
            # sum_{w != u} m_{wv} = node_agg[v] - m_{uv}
            rev_sub = node_agg[col] - edge_messages
            new_msg = F.relu(layer(torch.cat([h0[col], rev_sub, edge_attr], dim=-1)))
            edge_messages = self.drop(new_msg)

        # 节点更新: h_v = ReLU(W_o [h_v^0, sum_w m_{wv}])
        node_msg_agg = scatter(edge_messages, col, dim=0, dim_size=h0.size(0), reduce='sum')
        h_final = F.relu(self.out_lin(torch.cat([h0, node_msg_agg], dim=-1)))
        h_final = self.drop(h_final)

        graph_emb = global_mean_pool(h_final, batch)
        return self.head(graph_emb)


def build_pyg_model(name, in_channels, hidden_dim=128, n_layers=3,
                    dropout=0.2, edge_dim=4):
    """工厂函数"""
    if name == 'AttentiveFP':
        return AttentiveFPFull(in_channels, hidden_dim, n_layers,
                               num_timesteps=2, dropout=dropout,
                               edge_dim=edge_dim, out_channels=1)
    if name == 'DMPNN':
        return DMPNNModel(in_channels, hidden_dim, n_layers,
                          dropout=dropout, edge_dim=edge_dim, out_channels=1)
    raise ValueError(f"未知 PyG 模型: {name}")
