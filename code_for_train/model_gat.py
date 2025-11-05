import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_node_matrix(node_fea, mask_mat):
 
    return node_fea * mask_mat

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 参数化权重矩阵 (更高效)
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        # 特征变换 (高效矩阵乘法)
        Wh = torch.matmul(h, self.W)  # (batch_size, n_atoms, out_features)
        
        # 计算注意力系数 (使用广播机制)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 特征聚合
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """高效注意力计算 (避免显式复制)"""
        # 分解计算 (Wh1 + Wh2^T)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        return Wh1 + Wh2.transpose(1, 2)

class MultiHeadGAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        
        # 多头注意力层 (并行)
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        
        # 输出层 (带批归一化)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)
        self.bn_out = nn.BatchNorm1d(nclass)

    def forward(self, x, adj):
        # 输入Dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 多头注意力计算
        head_outputs = [att(x, adj) for att in self.attentions]
        x = torch.cat(head_outputs, dim=-1)
        
        # 输出层
        x = self.out_att(x, adj)
        
        # 批归一化 (高效维度处理)
        x = x.transpose(1, 2)  # (batch, features, nodes)
        x = self.bn_out(x)     # 批归一化在特征维度
        return x.transpose(1, 2)  # 恢复维度

class ChemGCN(nn.Module):
   
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.2,
        n_heads: int = 4
    ):
        super().__init__()
        
        # 保存参数 (兼容接口)
        self.node_vec_len = node_vec_len
        self.node_fea_len = node_fea_len
        self.hidden_fea_len = hidden_fea_len
        self.n_conv = n_conv
        self.n_hidden = n_hidden
        self.p_dropout = p_dropout
        self.n_heads = n_heads

        # 初始变换 (带批归一化)
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, node_fea_len),
            nn.BatchNorm1d(node_fea_len),
            nn.LeakyReLU(0.2)
        )
        
        # 图卷积层 (带残差连接)
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv):
            in_dim = node_fea_len if i == 0 else hidden_fea_len
            self.conv_layers.append(
                MultiHeadGAT(
                    nfeat=in_dim,
                    nhid=hidden_fea_len // n_heads,  # 高效维度分配
                    nclass=hidden_fea_len,
                    dropout=p_dropout,
                    alpha=0.2,
                    nheads=n_heads
                )
            )
            # 为每层添加批归一化
            self.add_module(f'conv_bn_{i}', nn.BatchNorm1d(hidden_fea_len))
        
        # 图池化
        self.pooling = lambda x: x.mean(dim=1)  # 函数式定义，更轻量
        self.pooling_activation = nn.LeakyReLU(0.2)
        
        # 隐藏层 (带批归一化和残差)
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for i in range(n_hidden):
            in_dim = hidden_fea_len if i == 0 else hidden_fea_len
            self.hidden_layers.append(nn.Linear(in_dim, hidden_fea_len))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_fea_len))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_fea_len, n_outputs)
        )

    def forward(self, node_mat, adj_mat, mask_mat):
        # 初始变换
        batch_size, n_atoms, _ = node_mat.shape
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).view(batch_size, n_atoms, -1)
        
        # 初始掩码过滤
        node_fea = filter_node_matrix(node_fea, mask_mat)
        
        # 图卷积层 (带残差连接)
        for i, conv in enumerate(self.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            
            # 残差连接
            if residual.size() == node_fea.size():
                node_fea += residual
                
            # 批归一化 + 激活
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)
            
        # 中间层掩码过滤
        node_fea = filter_node_matrix(node_fea, mask_mat)
        
        # 图池化
        pooled = self.pooling_activation(self.pooling(node_fea))
        
        # 隐藏层处理
        for i in range(self.n_hidden):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, self.p_dropout, training=self.training)
        
        # 最终输出
        return self.output_layer(pooled)