"""
Method 4: 双通道消息传递 (共轭 vs 诱导)

思路:
    重新定义边类型, 将化学键分为两类:
        共轭通道 (conjugate): aromatic + double 键 → 传递共轭/π 电子效应
        诱导通道 (inductive): single + triple 键   → 传递 σ 诱导效应
    两个独立的消息传递通道分别处理各自的邻接矩阵, 通道输出通过可学习权重融合。

核心组件:
    - build_dual_adjacency(mol, max_atoms): 为单个分子生成 conj_adj / ind_adj
    - add_dual_adjacency(data): 数据预处理, 为整个数据集生成双通道邻接矩阵
    - DualChannelGNN: 双通道 GNN 模型 (复用 ConvolutionLayer)
    - build_model: 返回模型实例
"""
import os
import sys


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

CODE_END_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
for _p in (CODE_END_ROOT, ORIG_MODELS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdDistGeom as molDG

from unified_models.gnn.model import ConvolutionLayer


# ============== 数据预处理 ==============
def build_dual_adjacency(mol, max_atoms=75):
    """为单个分子生成双通道邻接矩阵 (共轭 / 诱导)

    边类型划分:
        共轭 (conj): aromatic, double
        诱导 (ind):  single, triple
    每个通道的邻接矩阵采用与原始 GNN 一致的距离加权, 并加上自环 (I)。

    Args:
        mol: RDKit Mol (未加氢或加氢均可, 内部统一 AddHs)
        max_atoms: 最大原子数 (矩阵尺寸)

    Returns:
        conj_adj: (max_atoms, max_atoms) 共轭通道邻接矩阵
        ind_adj:  (max_atoms, max_atoms) 诱导通道邻接矩阵
    """
    if mol is None:
        conj_adj = np.eye(max_atoms, dtype=np.float32)
        ind_adj = np.eye(max_atoms, dtype=np.float32)
        return conj_adj, ind_adj

    mol_h = Chem.AddHs(mol)
    n = mol_h.GetNumAtoms()

    try:
        dist_mat = molDG.GetMoleculeBoundsMatrix(mol_h)
        dist_mat[dist_mat == 0.0] = 1.0
    except Exception:
        dist_mat = np.ones((n, n), dtype=np.float32)

    conj_adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    ind_adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)

    for bond in mol_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i >= max_atoms or j >= max_atoms:
            continue
        w = 1.0 / float(dist_mat[i][j])
        bt = bond.GetBondType()
        if bt in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            conj_adj[i, j] = w
            conj_adj[j, i] = w
        elif bt in (Chem.BondType.SINGLE, Chem.BondType.TRIPLE):
            ind_adj[i, j] = w
            ind_adj[j, i] = w
        # 其他键类型 (如 1.5 等) 忽略, 避免重复计数

    # 各通道加自环 (与原始 GNN 的 adj + I 一致)
    conj_adj += np.eye(max_atoms, dtype=np.float32)
    ind_adj += np.eye(max_atoms, dtype=np.float32)
    return conj_adj, ind_adj


def add_dual_adjacency(data, max_atoms=75, device='cpu'):
    """为整个数据集生成双通道邻接矩阵

    Args:
        data: load_adj_format 返回的字典, 需包含 'smiles'
        max_atoms: 最大原子数 (默认 75)
        device: 输出张量所在设备

    Returns:
        更新后的 data 字典, 新增字段:
            - 'conj_adjs': torch.Tensor (N, max_atoms, max_atoms)
            - 'ind_adjs':  torch.Tensor (N, max_atoms, max_atoms)
    """
    smiles = data['smiles']
    n = len(smiles)
    conj_adjs = np.zeros((n, max_atoms, max_atoms), dtype=np.float32)
    ind_adjs = np.zeros((n, max_atoms, max_atoms), dtype=np.float32)

    for i in range(n):
        mol = Chem.MolFromSmiles(smiles[i])
        c_adj, i_adj = build_dual_adjacency(mol, max_atoms)
        conj_adjs[i] = c_adj
        ind_adjs[i] = i_adj

    new_data = dict(data)
    new_data['conj_adjs'] = torch.tensor(conj_adjs, dtype=torch.float32, device=device)
    new_data['ind_adjs'] = torch.tensor(ind_adjs, dtype=torch.float32, device=device)
    return new_data


# ============== 模型 ==============
class DualChannelGNN(nn.Module):
    """双通道消息传递 GNN

    两个独立的消息传递通道:
        - 共轭通道: 使用 conj_adj (aromatic + double)
        - 诱导通道: 使用 ind_adj  (single + triple)
    各通道独立完成 init_transform + 多层 ConvolutionLayer, 然后通过可学习门控融合:
        gate = sigmoid(W_g * cat[conj_fea, ind_fea] + b_g)
        fused = gate * conj_fea + (1 - gate) * ind_fea
    最后经 hidden_layers + output_layer 输出预测。

    forward 接口:
        forward(node_mat, conj_adj=None, ind_adj=None, adj_mat=None,
                mask_mat=None, ring_indices=None)
        若未提供 conj_adj/ind_adj 但提供了 adj_mat, 则两个通道均使用 adj_mat
        (便于与原始数据加载兼容)。
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                 p_dropout=0.2, mode='label'):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        self.mode = mode

        # 共享的初始特征变换 (两通道共享, 减少参数量; 也可各自独立)
        self.init_transform = nn.Sequential(
            nn.Linear(node_vec_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # mask 模式下的投影
        if mode == 'mask':
            self.mask_proj = nn.Linear(node_vec_len, hidden_dim, bias=False)
            with torch.no_grad():
                nn.init.ones_(self.mask_proj.weight)

        # 两个独立的卷积通道
        self.conj_convs = nn.ModuleList([
            ConvolutionLayer(hidden_dim, hidden_dim) for _ in range(n_conv)
        ])
        self.ind_convs = nn.ModuleList([
            ConvolutionLayer(hidden_dim, hidden_dim) for _ in range(n_conv)
        ])
        for i in range(n_conv):
            self.add_module(f'conj_bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.add_module(f'ind_bn_{i}', nn.BatchNorm1d(hidden_dim))

        # 可学习门控融合 (逐节点门控)
        self.fuse_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        # 全局通道权重 (额外提供整体通道权衡能力)
        self.channel_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # 池化
        self.pooling_activation = nn.LeakyReLU(0.2)

        # hidden + output (与 GNNModel 一致)
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_layer = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def _run_channel(self, node_fea, adj, convs, bn_prefix):
        """运行单个通道的消息传递"""
        for i, conv in enumerate(convs):
            residual = node_fea
            node_fea = conv(node_fea, adj)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(self, f'{bn_prefix}_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)
        return node_fea

    def forward(self, node_mat, conj_adj=None, ind_adj=None, adj_mat=None,
                mask_mat=None, ring_indices=None):
        batch_size, n_atoms, _ = node_mat.shape

        # 兼容性: 若未提供双通道邻接, 用 adj_mat 兜底 (两通道相同)
        if conj_adj is None:
            conj_adj = adj_mat
        if ind_adj is None:
            ind_adj = adj_mat
        if conj_adj is None and ind_adj is None:
            raise ValueError("DualChannelGNN requires conj_adj/ind_adj (or adj_mat as fallback)")

        # 初始变换
        node_fea = self.init_transform(
            node_mat.reshape(-1, self.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        if self.mode == 'mask' and mask_mat is not None:
            if mask_mat.shape[-1] != node_fea.shape[-1]:
                mask_mat = self.mask_proj(mask_mat.float())
                mask_mat = (mask_mat > 0).float()
            node_fea = node_fea * mask_mat

        # 两通道独立消息传递
        conj_fea = self._run_channel(node_fea, conj_adj, self.conj_convs, 'conj_bn')
        ind_fea = self._run_channel(node_fea, ind_adj, self.ind_convs, 'ind_bn')

        # 逐节点门控融合
        gate = torch.sigmoid(self.fuse_gate(torch.cat([conj_fea, ind_fea], dim=-1)))
        # 全局通道权重 (softmax 归一化)
        cw = F.softmax(self.channel_weight, dim=0)
        conj_w, ind_w = cw[0], cw[1]
        fused = gate * conj_fea * (1.0 + conj_w) + (1.0 - gate) * ind_fea * (1.0 + ind_w)
        # 上式等价于: 门控 + 全局通道加权 (用 (1+w) 形式保持尺度, 避免梯度消失)

        # 池化 (复用 GNNModel 的 pool 模式逻辑)
        if self.mode == 'pool' and ring_indices is not None:
            mask = (ring_indices >= 0).float().unsqueeze(-1)
            ring_fea = (fused * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            global_fea = fused.mean(dim=1)
            pooled = self.pooling_activation(0.7 * ring_fea + 0.3 * global_fea)
        else:
            pooled = self.pooling_activation(fused.mean(dim=1))

        for i in range(len(self.hidden_layers)):
            pooled = self.hidden_layers[i](pooled)
            pooled = self.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return self.output_layer(pooled)


# ============== 构建模型 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=2, n_outputs=1, p_dropout=0.2, mode='label', **kwargs):
    """返回 Method 4 模型实例

    Args:
        base: 目前支持 'gnn' (基于 ConvolutionLayer 的双通道 GNN)
        node_vec_len: 节点特征长度 (默认 60)
        其余参数同 GNNModel
    """
    base = base.lower()
    if base == 'gnn':
        return DualChannelGNN(node_vec_len, hidden_dim, n_conv, n_hidden,
                              n_outputs, p_dropout, mode)
    else:
        raise ValueError(f"Method 4 currently supports base='gnn' only, got: {base}")


if __name__ == '__main__':
    # 自测: 双通道邻接构建 + 模型前向传播
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_atoms, nvl = 75, 60

    # 测试 build_dual_adjacency
    mol = Chem.MolFromSmiles('c1ccc(O)cc1')
    conj_adj, ind_adj = build_dual_adjacency(mol, n_atoms)
    assert conj_adj.shape == (n_atoms, n_atoms) and ind_adj.shape == (n_atoms, n_atoms)
    print(f"[m4] conj_adj sum={conj_adj.sum():.2f}, ind_adj sum={ind_adj.sum():.2f}")

    # 测试模型
    model = build_model(node_vec_len=nvl, hidden_dim=64, n_conv=2, n_hidden=1,
                        n_outputs=1, mode='label').to(dev)
    node_mat = torch.randn(4, n_atoms, nvl, device=dev)
    conj = torch.eye(n_atoms, device=dev).unsqueeze(0).expand(4, -1, -1).clone()
    ind = torch.eye(n_atoms, device=dev).unsqueeze(0).expand(4, -1, -1).clone()
    # 双通道输入
    out = model(node_mat, conj_adj=conj, ind_adj=ind)
    print(f"[m4] DualChannelGNN output shape (dual-channel): {out.shape}")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    # adj_mat 兜底输入
    out2 = model(node_mat, adj_mat=conj)
    assert out2.shape == (4, 1)
    print("[m4] OK")
