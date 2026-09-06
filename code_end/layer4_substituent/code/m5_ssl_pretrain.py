"""
Method 5: 取代基替换预测自监督预训练

思路:
    预训练任务: 给定分子, 随机遮蔽/替换一个取代基, 预测替换前后的性质差异。
    步骤1: 从训练集中生成"伪配对" — 同一骨架不同取代基的分子对
    步骤2: 预训练模型预测 ΔP = P_modified - P_original
    步骤3: 微调到下游任务

核心组件:
    - generate_substituent_pairs(smiles_list, targets): 生成取代基替换对
    - build_pair_features(pairs, ...): 为配对构建图特征
    - PretrainDataset: 返回 (original_features, modified_features, delta_target)
    - SSLPretrainModel: 双输入共享编码器 GNN, 预测性质差值
    - pretrain_model(model, pairs_data, ...): 预训练循环
    - finetune_model(pretrained_model, task_data, ...): 微调循环

简化实现: 用同骨架 (Murcko scaffold) 分子配对; 找不到同骨架则随机配对。
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

# 统一导入路径 (与任务约定一致)
CODE_END_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
for _p in (CODE_END_ROOT, ORIG_MODELS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem.Scaffolds import MurckoScaffold

from unified_models.gnn.model import GNNModel


# ============== 配对生成 ==============
def _get_scaffold(smiles):
    """获取 Murcko 骨架 SMILES (失败返回空串)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ''


def generate_substituent_pairs(smiles_list, targets, max_pairs_per_scaffold=10,
                               seed=42):
    """生成取代基替换对 (同骨架优先, 否则随机配对)

    Args:
        smiles_list: SMILES 列表
        targets:     对应性质值列表
        max_pairs_per_scaffold: 每个骨架最多生成的配对数
        seed: 随机种子

    Returns:
        list of (orig_smi, mod_smi, delta_target) 其中 delta = target_orig - target_mod
    """
    rng = np.random.default_rng(seed)
    items = list(zip(smiles_list, targets))
    n = len(items)

    # 按骨架分组
    scaffold_map = {}
    for smi, tgt in items:
        scaf = _get_scaffold(smi)
        scaffold_map.setdefault(scaf, []).append((smi, float(tgt)))

    pairs = []
    singletons = []
    for scaf, group in scaffold_map.items():
        if len(group) >= 2:
            # 同骨架内随机配对
            idxs = list(range(len(group)))
            rng.shuffle(idxs)
            count = 0
            for i in range(0, len(idxs) - 1, 2):
                if count >= max_pairs_per_scaffold:
                    break
                a = group[idxs[i]]
                b = group[idxs[i + 1]]
                pairs.append((a[0], b[0], a[1] - b[1]))
                count += 1
        else:
            singletons.extend(group)

    # 单骨架分子随机配对 (跨骨架), 保证覆盖率
    if len(singletons) >= 2:
        idxs = list(range(len(singletons)))
        rng.shuffle(idxs)
        for i in range(0, len(idxs) - 1, 2):
            a = singletons[idxs[i]]
            b = singletons[idxs[i + 1]]
            pairs.append((a[0], b[0], a[1] - b[1]))

    # 兜底: 若配对数过少, 全局随机配对
    if len(pairs) < max(1, n // 4):
        idxs = list(range(n))
        rng.shuffle(idxs)
        for i in range(0, n - 1, 2):
            a = items[idxs[i]]
            b = items[idxs[i + 1]]
            pairs.append((a[0], b[0], float(a[1]) - float(b[1])))

    return pairs


# ============== 图特征构建 ==============
def _build_simple_graph(smiles, node_vec_len=60, max_atoms=75):
    """从 SMILES 构建简化的节点特征矩阵和邻接矩阵 (不依赖 atom_on_ring)

    节点特征复用原始 Graph 的核心字段 (原子序数、Gasteiger 电荷、芳香性、
    邻居数、H 数); 邻接矩阵采用与原始 GNN 一致的距离加权 + 自环。

    Returns:
        node_mat: (max_atoms, node_vec_len) float32
        adj_mat:  (max_atoms, max_atoms) float32
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (np.zeros((max_atoms, node_vec_len), dtype=np.float32),
                np.eye(max_atoms, dtype=np.float32))
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.ComputeGasteigerCharges(mol_h)
    except Exception:
        pass

    node_mat = np.zeros((max_atoms, node_vec_len), dtype=np.float32)
    for atom in mol_h.GetAtoms():
        i = atom.GetIdx()
        if i >= max_atoms:
            continue
        node_mat[i, atom.GetAtomicNum()] = 10.0
        try:
            node_mat[i, node_vec_len - 1] = float(atom.GetProp('_GasteigerCharge'))
        except KeyError:
            pass
        node_mat[i, node_vec_len - 5] = int(atom.GetIsAromatic())
        node_mat[i, node_vec_len - 6] = len(atom.GetNeighbors())
        node_mat[i, node_vec_len - 7] = atom.GetTotalNumHs()

    adj_mat = rdmolops.GetAdjacencyMatrix(mol_h).astype(np.float32)
    try:
        dist_mat = molDG.GetMoleculeBoundsMatrix(mol_h)
        dist_mat[dist_mat == 0.0] = 1.0
        adj_mat = adj_mat * (1.0 / dist_mat)
    except Exception:
        pass
    pad = max_atoms - adj_mat.shape[0]
    if pad > 0:
        adj_mat = np.pad(adj_mat, ((0, pad), (0, pad)), mode='constant')
    adj_mat = adj_mat + np.eye(max_atoms, dtype=np.float32)
    return node_mat, adj_mat


def build_pair_features(pairs, node_vec_len=60, max_atoms=75):
    """为配对列表构建图特征张量

    Args:
        pairs: generate_substituent_pairs 返回的列表
        node_vec_len: 节点特征长度 (默认 60)
        max_atoms: 最大原子数 (默认 75)

    Returns:
        dict:
            'orig_node_mats': (M, max_atoms, node_vec_len)
            'orig_adj_mats':  (M, max_atoms, max_atoms)
            'mod_node_mats':  (M, max_atoms, node_vec_len)
            'mod_adj_mats':   (M, max_atoms, max_atoms)
            'delta_targets':  (M, 1)
    """
    M = len(pairs)
    orig_nodes = np.zeros((M, max_atoms, node_vec_len), dtype=np.float32)
    orig_adjs = np.zeros((M, max_atoms, max_atoms), dtype=np.float32)
    mod_nodes = np.zeros((M, max_atoms, node_vec_len), dtype=np.float32)
    mod_adjs = np.zeros((M, max_atoms, max_atoms), dtype=np.float32)
    deltas = np.zeros((M, 1), dtype=np.float32)

    for i, (orig_smi, mod_smi, delta) in enumerate(pairs):
        on, oa = _build_simple_graph(orig_smi, node_vec_len, max_atoms)
        mn, ma = _build_simple_graph(mod_smi, node_vec_len, max_atoms)
        orig_nodes[i] = on
        orig_adjs[i] = oa
        mod_nodes[i] = mn
        mod_adjs[i] = ma
        deltas[i, 0] = delta

    return {
        'orig_node_mats': torch.tensor(orig_nodes, dtype=torch.float32),
        'orig_adj_mats': torch.tensor(orig_adjs, dtype=torch.float32),
        'mod_node_mats': torch.tensor(mod_nodes, dtype=torch.float32),
        'mod_adj_mats': torch.tensor(mod_adjs, dtype=torch.float32),
        'delta_targets': torch.tensor(deltas, dtype=torch.float32),
    }


# ============== 数据集 ==============
class PretrainDataset(Dataset):
    """取代基替换预训练数据集

    __getitem__ 返回:
        ((orig_node, orig_adj, mod_node, mod_adj), delta_target)
    """

    def __init__(self, pair_features):
        self.f = pair_features

    def __len__(self):
        return self.f['orig_node_mats'].shape[0]

    def __getitem__(self, i):
        return (
            self.f['orig_node_mats'][i],
            self.f['orig_adj_mats'][i],
            self.f['mod_node_mats'][i],
            self.f['mod_adj_mats'][i],
        ), self.f['delta_targets'][i]


# ============== 模型 ==============
class SSLPretrainModel(nn.Module):
    """取代基替换预测自监督预训练模型 (双输入共享编码器)

    共享 GNN 编码器分别编码 original / modified 分子, 拼接
    [orig_repr, mod_repr, orig_repr - mod_repr] 后预测 ΔP。

    forward(orig_node, orig_adj, mod_node, mod_adj) → (B, 1)
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden=1,
                 p_dropout=0.2, mode='label'):
        super().__init__()
        self.node_vec_len = node_vec_len
        self.hidden_dim = hidden_dim
        # 共享 GNN 编码器: n_hidden=0, n_outputs=hidden_dim → 输出 (B, hidden_dim)
        self.encoder = GNNModel(
            node_vec_len=node_vec_len,
            hidden_dim=hidden_dim,
            n_conv=n_conv,
            n_hidden=0,
            n_outputs=hidden_dim,
            p_dropout=p_dropout,
            mode=mode,
        )
        # ΔP 预测头: 拼接 [orig, mod, diff]
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, node_mat, adj_mat):
        return self.encoder(node_mat, adj_mat)  # (B, hidden_dim)

    def forward(self, orig_node, orig_adj, mod_node, mod_adj):
        orig_repr = self.encode(orig_node, orig_adj)
        mod_repr = self.encode(mod_node, mod_adj)
        diff = orig_repr - mod_repr
        combined = torch.cat([orig_repr, mod_repr, diff], dim=-1)
        return self.delta_head(combined)  # (B, 1)


class SSLFinetuneModel(nn.Module):
    """基于预训练编码器的下游微调模型

    复用预训练 encoder, 接新的回归头。
    forward(node_mat, adj_mat) → (B, 1), 接口与 GNNModel 一致。
    """

    def __init__(self, pretrained, hidden_dim, p_dropout=0.2):
        super().__init__()
        self.encoder = pretrained.encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_mat, adj_mat, mask_mat=None, ring_indices=None):
        repr_ = self.encoder(node_mat, adj_mat)  # (B, hidden_dim)
        return self.head(repr_)  # (B, 1)


# ============== 训练循环 ==============
def pretrain_model(model, pairs_data, epochs=50, lr=1e-3, device='cpu',
                   batch_size=32, verbose=True):
    """预训练循环: 预测 ΔP = P_modified - P_original

    Args:
        model: SSLPretrainModel
        pairs_data: build_pair_features 返回的字典
        epochs, lr, batch_size, device: 训练超参
        verbose: 是否打印进度

    Returns:
        训练好的 model (含已更新的共享 encoder)
    """
    model = model.to(device)
    dataset = PretrainDataset(pairs_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for (on, oa, mn, ma), delta in loader:
            on = on.to(device); oa = oa.to(device)
            mn = mn.to(device); ma = ma.to(device)
            delta = delta.to(device)

            pred = model(on, oa, mn, ma)
            loss = criterion(pred, delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[m5 pretrain] epoch {epoch+1}/{epochs}  loss={total_loss/max(1,n_batches):.4f}")

    return model


def finetune_model(pretrained_model, task_data, epochs=50, lr=1e-4,
                   device='cpu', batch_size=32, hidden_dim=128, verbose=True):
    """微调循环: 在下游任务上微调预训练编码器

    Args:
        pretrained_model: SSLPretrainModel (已预训练)
        task_data: dict, 需包含 'node_mats', 'adj_mats', 'outputs'
            - node_mats: (N, max_atoms, node_vec_len)
            - adj_mats:  (N, max_atoms, max_atoms)
            - outputs:   (N,) 或 (N, 1)
        epochs, lr, batch_size, device: 训练超参
        hidden_dim: 预训练编码器的 hidden_dim (用于构建微调头)
        verbose: 是否打印进度

    Returns:
        SSLFinetuneModel, 已在 task_data 上微调
    """
    ft_model = SSLFinetuneModel(pretrained_model, hidden_dim).to(device)

    node_mats = task_data['node_mats']
    adj_mats = task_data['adj_mats']
    outputs = task_data['outputs']
    if not torch.is_tensor(node_mats):
        node_mats = torch.as_tensor(node_mats, dtype=torch.float32)
    if not torch.is_tensor(adj_mats):
        adj_mats = torch.as_tensor(adj_mats, dtype=torch.float32)
    if not torch.is_tensor(outputs):
        outputs = torch.as_tensor(outputs, dtype=torch.float32)
    outputs = outputs.reshape(-1, 1).float()

    N = node_mats.shape[0]
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ft_model.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            nm = node_mats[idx].to(device)
            am = adj_mats[idx].to(device)
            tgt = outputs[idx].to(device)

            pred = ft_model(nm, am)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[m5 finetune] epoch {epoch+1}/{epochs}  loss={total_loss/max(1,n_batches):.4f}")

    return ft_model


# ============== 构建模型 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=1, n_outputs=1, p_dropout=0.2, mode='label', **kwargs):
    """返回 Method 5 预训练模型实例 (SSLPretrainModel)

    Args:
        base: 目前支持 'gnn' (基于 GNNModel 编码器)
        node_vec_len: 节点特征长度 (默认 60)
        n_hidden: 预测头隐藏层数 (默认 1)
        其余参数同 GNNModel

    注意: n_outputs 参数仅用于接口一致, 预训练头固定输出 1 维 (ΔP)。
    """
    base = base.lower()
    if base == 'gnn':
        return SSLPretrainModel(node_vec_len, hidden_dim, n_conv, n_hidden,
                                p_dropout, mode)
    else:
        raise ValueError(f"Method 5 currently supports base='gnn' only, got: {base}")


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nvl, max_atoms = 60, 75

    # 测试配对生成
    smiles = ['c1ccc(O)cc1', 'c1ccc(OC)cc1', 'c1ccc([N+](=O)[O-])cc1',
              'c1ccccc1C', 'c1ccc(Cl)cc1', 'c1ccc(C)cc1']
    targets = [1.0, 1.2, 0.3, 0.9, 0.5, 0.85]
    pairs = generate_substituent_pairs(smiles, targets, seed=0)
    print(f"[m5] generated {len(pairs)} pairs")
    assert len(pairs) >= 1

    # 构建特征 + 预训练模型前向
    feats = build_pair_features(pairs, nvl, max_atoms)
    model = build_model(node_vec_len=nvl, hidden_dim=32, n_conv=2, n_hidden=1,
                        mode='label').to(dev)
    on = feats['orig_node_mats'].to(dev)
    oa = feats['orig_adj_mats'].to(dev)
    mn = feats['mod_node_mats'].to(dev)
    ma = feats['mod_adj_mats'].to(dev)
    out = model(on, oa, mn, ma)
    print(f"[m5] SSLPretrainModel output shape: {out.shape} (expected: [{len(pairs)}, 1])")
    assert out.shape == (len(pairs), 1), f"Unexpected: {out.shape}"

    # 测试微调模型前向 (接口与 GNNModel 一致)
    ft = SSLFinetuneModel(model, hidden_dim=32).to(dev)
    nm = torch.randn(4, max_atoms, nvl, device=dev)
    am = torch.eye(max_atoms, device=dev).unsqueeze(0).expand(4, -1, -1).clone()
    ft_out = ft(nm, am)
    assert ft_out.shape == (4, 1), f"Unexpected: {ft_out.shape}"
    print("[m5] OK")
