"""
GAT - Graph Attention Network
支持label/mask/pool三种环信息编码方式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """单头GAT层"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime


class MultiHeadGAT(nn.Module):
    """多头GAT层"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)
        self.bn_out = nn.BatchNorm1d(nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.out_att(x, adj)
        x = x.transpose(1, 2)
        x = self.bn_out(x)
        return x.transpose(1, 2)


class GATModel(nn.Module):
    """GAT主模型，统一支持label/mask/pool三种模式"""
    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, n_heads=4, mode='label'):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.mode = mode

        # 初始变换
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # mask投影层 (仅mask模式使用)
        if mode == 'mask':
            self.mask_proj = nn.Linear(node_vec_len, hidden_dim, bias=False)
            with torch.no_grad():
                nn.init.ones_(self.mask_proj.weight)

        # 图卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.conv_layers.append(
                MultiHeadGAT(nfeat=in_dim, nhid=hidden_dim // n_heads,
                            nclass=hidden_dim, dropout=p_dropout,
                            alpha=0.2, nheads=n_heads)
            )
            self.add_module(f'conv_bn_{i}', nn.BatchNorm1d(hidden_dim))

        # 池化
        if mode == 'pool':
            # pool模式: 突出目标环原子的池化
            self.pooling = None  # 在forward中处理
        else:
            self.pooling = lambda x: x.mean(dim=1)
        self.pooling_activation = nn.LeakyReLU(0.2)

        # 隐藏层
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

        # mask模式: 应用mask过滤
        if self.mode == 'mask' and mask_mat is not None:
            if mask_mat.shape[-1] != node_fea.shape[-1]:
                mask_mat = self.mask_proj(mask_mat.float())
                mask_mat = (mask_mat > 0).float()
            node_fea = node_fea * mask_mat

        # 图卷积
        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea += residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        # 池化
        if self.mode == 'pool' and ring_indices is not None:
            # pool模式: 对目标环原子和所有原子分别池化后拼接
            mask = (ring_indices >= 0).float().unsqueeze(-1)  # (batch, n_atoms, 1)
            ring_fea = (node_fea * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            global_fea = node_fea.mean(dim=1)
            pooled = self.pooling_activation(0.7 * ring_fea + 0.3 * global_fea)
        else:
            pooled = self.pooling_activation(self.pooling(node_fea))

        # 隐藏层
        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)
