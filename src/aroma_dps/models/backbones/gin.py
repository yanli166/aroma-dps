"""
GIN - Graph Isomorphism Network
支持label/mask/pool三种环信息编码方式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GINLayer(nn.Module):
    """GIN卷积层: out = MLP((1 + eps) * x + sum_neighbors(x))"""
    def __init__(self, in_dim, out_dim, p_dropout=0.2):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

    def forward(self, x, adj):
        # adj已经是带自环的邻接矩阵
        agg = torch.matmul(adj, x)
        out = (1 + self.eps) * x + agg
        # MLP 期望 (batch*n_atoms, dim)
        b, n, d = out.shape
        out = out.reshape(b * n, d)
        out = self.mlp(out)
        return out.reshape(b, n, -1)


class GINModel(nn.Module):
    """GIN主模型，统一支持label/mask/pool三种模式"""
    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label'):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.mode = mode

        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        if mode == 'mask':
            self.mask_proj = nn.Linear(node_vec_len, hidden_dim, bias=False)
            with torch.no_grad():
                nn.init.ones_(self.mask_proj.weight)

        self.conv_layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim, p_dropout)
            for _ in range(n_conv)
        ])
        for i in range(n_conv):
            self.add_module(f'conv_bn_{i}', nn.BatchNorm1d(hidden_dim))

        if mode == 'pool':
            self.pooling = None
        else:
            self.pooling = lambda x: x.mean(dim=1)
        self.pooling_activation = nn.LeakyReLU(0.2)

        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, n_outputs)
        )

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None):
        batch_size, n_atoms, _ = node_mat.shape
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

        if self.mode == 'pool' and ring_indices is not None:
            mask = (ring_indices >= 0).float().unsqueeze(-1)
            ring_fea = (node_fea * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            global_fea = node_fea.mean(dim=1)
            pooled = self.pooling_activation(0.7 * ring_fea + 0.3 * global_fea)
        else:
            pooled = self.pooling_activation(self.pooling(node_fea))

        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)
