"""
Ring-Conditioned GNN: 统一可配置模型框架

将消息传递 backbone (GNN/GIN/GAT/MPNN) 与环级 readout 解耦,
通过配置组合实现四阶段实验所需的所有变体:

  Stage1 Base:    backbone + fixed_avg readout (ring_flag=0, 无环条件化消息传递)
  Stage2 Membership:    backbone + fixed_avg readout (ring_flag=10, 环成员关系注入输入)
  Stage2 LearnableReadout: backbone + attention readout (ring_flag=0, 可学习环聚合)
  Stage2 Joint:   backbone + attention readout (ring_flag=10, 联合)
  Stage4: 同 Stage2 配置, 跨 MPNN/DMPNN/GIN/GAT backbones

复用原始 unified_models 的卷积层实现 (不修改原始代码), 仅替换 readout。
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)
from unified_models.gnn.model import ConvolutionLayer
from unified_models.gin.model import GINLayer
from unified_models.gat.model import MultiHeadGAT
from unified_models.mpnn.model import MPNNLayer
from unified_models.graphsage.model import GraphSAGELayer

from models.ring_readout import build_readout

# backbone -> 卷积层类映射
CONV_LAYERS = {
    'GNN': ConvolutionLayer,
    'GIN': GINLayer,
    'GAT': MultiHeadGAT,
    'MPNN': MPNNLayer,
    'GraphSAGE': GraphSAGELayer,
}


class RingConditionedGNN(nn.Module):
    """Ring-Conditioned GNN 统一模型

    Args:
        backbone:       'GNN' / 'GIN' / 'GAT' / 'MPNN' / 'GraphSAGE'
        node_vec_len:   节点特征维度
        hidden_dim:     隐藏层维度
        n_conv:         卷积层数
        n_hidden:       MLP 隐藏层数
        n_outputs:      输出维度 (回归=1)
        p_dropout:      dropout 概率
        readout_mode:   'fixed_avg' / 'attention' / 'global_mean'
        n_heads:        GAT 多头数 (仅 GAT 使用)
    """

    def __init__(self, backbone, node_vec_len, hidden_dim, n_conv, n_hidden,
                 n_outputs=1, p_dropout=0.2, readout_mode='fixed_avg', n_heads=4):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.readout_mode = readout_mode
        assert backbone in CONV_LAYERS, f"不支持骨干: {backbone}, 可选: {list(CONV_LAYERS.keys())}"

        # 初始变换
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # 卷积层 (复用原始实现)
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

        # Readout (可配置)
        self.readout = build_readout(readout_mode, hidden_dim, p_dropout)
        self.pooling_activation = nn.LeakyReLU(0.2)

        # MLP 隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, node_mat, adj_mat, ring_indices=None):
        """
        Args:
            node_mat:     (batch, n_atoms, node_vec_len) — 含或不含 ring_flag 取决于数据加载
            adj_mat:      (batch, n_atoms, n_atoms)
            ring_indices: (batch, n_atoms) — 目标环原子位置标记 (-1=非环原子)
        Returns:
            pred: (batch, n_outputs)
        """
        batch_size, n_atoms, _ = node_mat.shape
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        # 消息传递 (无环条件化, 纯 backbone)
        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        # Readout (输出阶段环级聚合)
        pooled = self.readout(node_fea, ring_indices)
        pooled = self.pooling_activation(pooled)

        # MLP head
        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)

    def encode(self, node_mat, adj_mat, ring_indices=None):
        """返回环级表示 (readout 输出, MLP head 之前), 供预训练/分析使用"""
        batch_size, n_atoms, _ = node_mat.shape
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        pooled = self.readout(node_fea, ring_indices)
        pooled = self.pooling_activation(pooled)
        return pooled, node_fea  # 同时返回节点级和环级表示


def build_model(backbone, node_vec_len, hidden_dim=128, n_conv=3, n_hidden=2,
                n_outputs=1, p_dropout=0.2, readout_mode='fixed_avg', n_heads=4):
    """工厂函数: 构造 RingConditionedGNN"""
    return RingConditionedGNN(
        backbone=backbone,
        node_vec_len=node_vec_len,
        hidden_dim=hidden_dim,
        n_conv=n_conv,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        p_dropout=p_dropout,
        readout_mode=readout_mode,
        n_heads=n_heads,
    )
