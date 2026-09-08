"""
RA-GCN 组合模型: 三类环信息编码范式联合 (Ring labeling + Masking + Pooling)

不修改原始模型骨干, 而是构造一个通用 wrapper, 复用原始模型的卷积层类型,
在同一前向中同时施加三种环编码:
  1. Ring Labeling  -> 数据层 (ring_flag_value=10 注入节点特征, 由 graph_data 负责)
  2. Ring Masking   -> init_transform 后, 用 mask_mat 过滤非目标环原子特征
  3. Ring Pooling   -> 卷积输出后, 按 ring_indices 定向聚合环原子特征

三种编码作用位置互不冲突 (输入层 / 传播层 / 输出层), 可叠加。
支持 GNN/GIN/GAT/MPNN/GraphSAGE 五种骨干 (复用各自卷积层)。
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ORIG_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'
sys.path.insert(0, ORIG_ROOT)
# 复用原始卷积层 (不改动)
from unified_models.gnn.model import ConvolutionLayer
from unified_models.gin.model import GINLayer
from unified_models.gat.model import MultiHeadGAT
from unified_models.mpnn.model import MPNNLayer
from unified_models.graphsage.model import GraphSAGELayer

CONV_LAYERS = {
    'GNN': ConvolutionLayer,
    'GIN': GINLayer,
    'GAT': MultiHeadGAT,        # 需特殊构造 (nheads)
    'MPNN': MPNNLayer,
    'GraphSAGE': GraphSAGELayer,
}


class RAGCNCombined(nn.Module):
    """RA-GCN: 三种环编码联合模型

    backbone: 任一原始卷积层 (GNN/GIN/GAT/MPNN/GraphSAGE)
    编码组合: label(数据层) + mask(传播层) + pool(输出层)
    """

    def __init__(self, backbone, node_vec_len, hidden_dim, n_conv, n_hidden,
                 n_outputs=1, p_dropout=0.2, n_heads=4):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        assert backbone in CONV_LAYERS, f"不支持骨干: {backbone}"

        # 初始变换 (与原始模型一致)
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        # mask 投影层 (与原始 mask 模式一致)
        self.mask_proj = nn.Linear(node_vec_len, hidden_dim, bias=False)
        with torch.no_grad():
            nn.init.ones_(self.mask_proj.weight)

        # 卷积层 (复用原始卷积层类型)
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv):
            if backbone == 'GAT':
                self.conv_layers.append(
                    MultiHeadGAT(nfeat=hidden_dim, nhid=hidden_dim // n_heads,
                                 nclass=hidden_dim, dropout=p_dropout,
                                 alpha=0.2, nheads=n_heads))
            elif backbone == 'GIN':
                self.conv_layers.append(GINLayer(hidden_dim, hidden_dim, p_dropout))
            elif backbone == 'MPNN':
                self.conv_layers.append(MPNNLayer(hidden_dim, p_dropout))
            elif backbone == 'GraphSAGE':
                self.conv_layers.append(GraphSAGELayer(hidden_dim, hidden_dim, p_dropout))
            else:  # GNN
                self.conv_layers.append(ConvolutionLayer(hidden_dim, hidden_dim))
            self.add_module(f'conv_bn_{i}', nn.BatchNorm1d(hidden_dim))

        self.pooling_activation = nn.LeakyReLU(0.2)
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None):
        batch_size, n_atoms, _ = node_mat.shape
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        # 编码2 Ring Masking: 过滤非目标环原子特征 (传播层)
        if mask_mat is not None:
            if mask_mat.shape[-1] != node_fea.shape[-1]:
                mask_mat = self.mask_proj(mask_mat.float())
                mask_mat = (mask_mat > 0).float()
            node_fea = node_fea * mask_mat

        # 卷积传播 (复用原始卷积层, 含残差+BN+激活, 与原始模型一致)
        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        # 编码3 Ring Pooling: 环原子定向聚合 (输出层)
        if ring_indices is not None:
            mask = (ring_indices >= 0).float().unsqueeze(-1)
            ring_fea = (node_fea * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            global_fea = node_fea.mean(dim=1)
            pooled = self.pooling_activation(0.7 * ring_fea + 0.3 * global_fea)
        else:
            pooled = self.pooling_activation(node_fea.mean(dim=1))

        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)
        return self.output_layer(pooled)
