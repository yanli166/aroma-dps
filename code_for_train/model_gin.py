import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_node_matrix(node_fea, mask_mat):
    """
    应用掩码过滤节点特征矩阵
    参数:
        node_fea: 节点特征张量 (batch_size, n_atoms, features)
        mask_mat: 掩码矩阵 (batch_size, n_atoms, 1)
    返回:
        过滤后的节点特征 (batch_size, n_atoms, features)
    """
    return node_fea * mask_mat

class GINLayer(nn.Module):
    """改进的GIN图同构网络层"""
    def __init__(self, in_features, out_features, eps=0.0, dropout=0.2):
        super().__init__()
        self.eps = eps
        self.dropout = dropout
        
        # 增强的MLP部分（参考文档1的批归一化）
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, adj):
        """
        前向传播（添加了残差连接和高效维度处理）
        参数:
            x: 节点特征 (batch_size, n_atoms, in_features)
            adj: 邻接矩阵 (batch_size, n_atoms, n_atoms)
        返回:
            更新后的节点特征 (batch_size, n_atoms, out_features)
        """
        residual = x  # 残差连接（参考文档1）
        neighbor_sum = torch.bmm(adj, x)
        
        # 添加自连接并应用GIN核心公式
        out = (1 + self.eps) * x + neighbor_sum
        
        # 高效维度处理（参考文档1）
        batch_size, n_atoms, _ = out.shape
        out = out.reshape(-1, out.size(-1))
        out = self.mlp(out)
        out = out.view(batch_size, n_atoms, -1)
        
        # 添加残差连接（参考文档1）
        if residual.size() == out.size():
            out += residual
            
        return out

class GIN(nn.Module):
    """改进的GIN图神经网络模型"""
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.2
    ):
        super().__init__()
        
        # 保存参数
        self.node_vec_len = node_vec_len
        self.node_fea_len = node_fea_len
        self.hidden_fea_len = hidden_fea_len
        self.n_conv = n_conv
        self.n_hidden = n_hidden
        self.p_dropout = p_dropout

        # 初始变换（增强批归一化和激活函数）
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, node_fea_len),
            nn.BatchNorm1d(node_fea_len),
            nn.LeakyReLU(0.2)  # 使用LeakyReLU（参考文档1）
        )
        
        # GIN卷积层（添加dropout参数）
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv):
            in_dim = node_fea_len if i == 0 else hidden_fea_len
            self.conv_layers.append(
                GINLayer(
                    in_features=in_dim,
                    out_features=hidden_fea_len,
                    eps=0.0,
                    dropout=p_dropout
                )
            )
            # 为每层添加批归一化（参考文档1的高效维度处理）
            self.add_module(f'conv_bn_{i}', nn.BatchNorm1d(hidden_fea_len))
        
        # 图池化（全局平均池化）
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 沿原子维度池化
        
        # 隐藏层（增强批归一化和激活函数）
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for i in range(n_hidden):
            in_dim = hidden_fea_len if i == 0 else hidden_fea_len
            self.hidden_layers.append(nn.Linear(in_dim, hidden_fea_len))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_fea_len))
        
        # 输出层（添加dropout）
        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_fea_len, n_outputs)
        )

    def forward(self, node_mat, adj_mat, mask_mat):
        """
        增强的前向传播（添加多阶段掩码过滤）
        参数:
            node_mat: 节点特征矩阵 (batch_size, n_atoms, node_vec_len)
            adj_mat: 邻接矩阵 (batch_size, n_atoms, n_atoms)
            mask_mat: 掩码矩阵 (batch_size, n_atoms, 1)
        返回:
            预测输出 (batch_size, n_outputs)
        """
        batch_size, n_atoms, _ = node_mat.shape
        
        # 初始变换（参考文档1的高效reshape处理）
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).view(batch_size, n_atoms, -1)
        
        # GIN卷积层（添加残差连接和高效批归一化）
        for i, conv in enumerate(self.conv_layers):
            node_fea = conv(node_fea, adj_mat)
            
            # 高效批归一化（参考文档1的维度转置）
            node_fea = node_fea.transpose(1, 2)  # (批次, 特征, 原子)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)  # 恢复(批次, 原子, 特征)
            
            # 激活函数
            node_fea = F.leaky_relu(node_fea, 0.2)  # 使用LeakyReLU
            
            # Dropout（在训练时应用）
            node_fea = F.dropout(node_fea, self.p_dropout, training=self.training)
        
        # 最终掩码过滤
        node_fea = filter_node_matrix(node_fea, mask_mat)
        
        # 图池化（全局平均）- 高效维度处理
        pooled = self.pooling(node_fea.transpose(1, 2)).squeeze(-1)
        
        # 隐藏层处理（添加批归一化和激活函数）
        for i in range(self.n_hidden):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, self.p_dropout, training=self.training)
        
        # 最终输出
        return self.output_layer(pooled)