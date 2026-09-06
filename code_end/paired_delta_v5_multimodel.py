
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# -*- coding: utf-8 -*-
"""
配对 Δ 监督 v5 多模型版  (v5 完整版 + 6 个 GNN 模型)
=====================================================================
v5 修改位置索引 (全文搜索 "★" 可逐条跳转):
  ★1 常量区: LAMBDA_DELTA 0.5 -> 2.0
  ★2 常量区: WARMUP_EPOCHS 50 -> 15
  ★3 常量区: DELTA_MAX 修正 (原值比真实Δ还小, 会罚正确答案)
  ★4 常量区: 新增 TAU (噪声级Δ不罚方向)
  ★5 scaffold_group_split 整体替换: 只有小组能进验证集
  ★6 新增 build_partner_map (伙伴映射)
  ★7 新增 partner_delta_loss (伙伴配对Δ损失), 删除旧 paired_delta_loss
  ★8 train_model: 删除 GroupBatchSampler, 训练循环改为
       "随机batch + 给每个分子随机叫一个同环伙伴" 的统一结构
  ★9 train_model: warmup 结束时同时重置学习率
  ★10 mode_label 加 _v5 后缀, 与 v4 结果区分

多模型扩展 (v5 新增):
  支持 6 个 GNN 模型, 分两种数据格式:
    - adj 格式: MPNN, GNN, GAT, GraphSAGE, GIN  (统一接口)
    - PyG 格式: AttentiveFP  (DataLoader 接口)
  通过 model_factory + format-aware forward_fn 统一训练循环。

v4 -> v5 的核心变化 (为什么):
  v4 的 GroupBatchSampler 让每个 batch 全是同一骨架的分子,
  模型在不同骨架的水准之间来回拉扯, λ 稍大就欠拟合(只能用到 0.5)。
  v5 让 batch 恢复随机(全局水准稳), 配对信号通过"伙伴调用"获得,
  因此 λ 可以开到 2-5 而不崩。
=====================================================================
"""
import os, sys, time, csv, random, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
os.chdir(PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)
from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)

from common.tasks import TASKS, DEFAULT_SEED, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format, load_pyg_format
from unified_models.mpnn.model import MPNNModel
from unified_models.gnn.model import GNNModel
from unified_models.gat.model import GATModel
from unified_models.graphsage.model import GraphSAGEModel
from unified_models.gin.model import GINModel
from baseline_gnn.code.pyg_models import build_pyg_model
from generalization_test.code.splits import prepare_external_test_csv

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results', 'paired_delta_v5_multimodel')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NVL, MAX_ATOMS, EXT_MAX_ATOMS = 60, 75, 85
RING_FLAG_VALUE = 10
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 32,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 50,
}

# ★1 修改点 (v5): LAMBDA_DELTA 0.5 -> 2.0 (可经 CLI --lambda 覆盖)
#    原因: v5 用随机 batch + 伙伴调用, 全局水准稳定, λ 可以开大
#    注意: 实测 λ=2.0 在 HOMA 上过拟合 (Train R²=0.996), 建议 0.5-1.0 起步
LAMBDA_DELTA = 2.0
# ★2 修改点 (v5): WARMUP_EPOCHS 50 -> 15 (可经 CLI --warmup 覆盖)
#    原因: 15 轮足够学好核心预测, 留更多轮次给配对Δ训练
WARMUP_EPOCHS = 15
# ★3 修改点 (v5): DELTA_MAX 修正 (原值比真实Δ还小, 会罚正确答案)
#    HOMA 真实Δ最大 ~0.32, MBCO ~0.12, NICS ~11
DELTA_MAX = {'HOMA': 0.35, 'NICS_1zz': 12.0, 'MBCO': 0.13}
# ★4 修改点 (v5): 新增 TAU 噪声阈值
#    |Δ| 低于 TAU 时不罚方向 (属于噪声水平, 罚方向无意义)
TAU = {'HOMA': 0.008, 'NICS_1zz': 0.3, 'MBCO': 0.003}

# 多模型注册表
ADJ_MODELS = {
    'MPNN': MPNNModel,
    'GNN': GNNModel,
    'GAT': GATModel,
    'GraphSAGE': GraphSAGEModel,
    'GIN': GINModel,
}
PYG_MODELS = ['AttentiveFP']
ALL_MODELS = list(ADJ_MODELS.keys()) + PYG_MODELS


def set_full_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============ 1. Murcko 骨架计算 ============
def compute_scaffolds(smiles_list):
    """计算每个分子的 Murcko 骨架, 返回 scaffold_key 列表"""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    keys = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            k = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ''
        except Exception:
            k = ''
        keys.append(k)
    return keys


# ★5 修改点 (v5): scaffold_group_split 整体替换 (只有小组能进验证集)
# ============ 2. 骨架分组验证集划分 (v5 修正版) ============
def scaffold_group_split(scaffold_keys, val_ratio=0.125, seed=42, min_train_groups=1):
    """按骨架分组划分训练/验证集 (v5修正版)
    规则: 只有"小组"(<= 目标验证量的1/4)才允许进验证集, 大组永远留训练集。
    原版缺陷: 先抽到的大组(可能几百个分子的苯环)会整体进验证集,
    训练直接丢掉最富有的配对资源。
    """
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for idx, k in enumerate(scaffold_keys):
        groups[k].append(idx)
    n_total = len(scaffold_keys)
    n_val_target = int(val_ratio * n_total)
    max_val_group = max(5, n_val_target // 4)
    small_keys = [k for k in groups if len(groups[k]) <= max_val_group]
    rng.shuffle(small_keys)
    val_idx, val_size = [], 0
    for k in small_keys:
        if val_size >= n_val_target:
            break
        val_idx.extend(groups[k])
        val_size += len(groups[k])
    if val_size < n_val_target * 0.5:
        print(f"  警告: 小组不足 (验证集仅 {val_size} 条), 回退到随机划分")
        perm = rng.permutation(n_total)
        n_val = int(val_ratio * n_total)
        return perm[n_val:].copy(), perm[:n_val].copy()
    val_set = set(val_idx)
    train_idx = np.array([i for i in range(n_total) if i not in val_set])
    print(f"  骨架划分: {len(groups)} 个骨架组, 验证集 {val_size} 条 "
          f"(单组上限 {max_val_group}), 大组全部保留在训练集")
    return train_idx, np.array(val_idx)


# ★6 修改点 (v5): 新增 build_partner_map
# ============ 3. 伙伴映射 (v5 新增) ============
def build_partner_map(scaffold_keys, train_idx):
    """每个训练分子 -> 同骨架的其他训练分子数组 (只含训练集, 防止泄漏)"""
    from collections import defaultdict
    groups = defaultdict(list)
    for i in train_idx:
        groups[scaffold_keys[int(i)]].append(int(i))
    return {i: np.array([j for j in g if j != i], dtype=np.int64)
            for g in groups.values() for i in g if len(g) >= 2}


# ============ 3b. GroupBatchSampler (v4 策略, 保留供 --strategy group_batch 使用) ============
class GroupBatchSampler:
    """每个 batch 来自同一骨架组, 保证 batch 内分子可配对 (v4 策略)
    实测: GroupBatchSampler 提供了重要的正则化效果 (within-scaffold BatchNorm),
    比 v5 的随机 batch + partner_map 更有效。
    """
    def __init__(self, scaffold_keys, train_idx, batch_size, seed=42, drop_last=False):
        from collections import defaultdict
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        groups = defaultdict(list)
        for idx in train_idx:
            groups[scaffold_keys[idx]].append(idx)
        self.groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 1}
        self.pairable_groups = {k: v for k, v in self.groups.items() if len(v) >= 2}
        self.single_groups = {k: v for k, v in self.groups.items() if len(v) == 1}
        n_pairable = sum(len(v) for v in self.pairable_groups.values())
        n_single = sum(len(v) for v in self.single_groups.values())
        print(f"  GroupBatchSampler: {len(self.pairable_groups)} 个可配对组 "
              f"({n_pairable} 分子), {len(self.single_groups)} 个单分子组 ({n_single})", flush=True)

    def __iter__(self):
        batches = []
        for k, indices in self.pairable_groups.items():
            n = len(indices)
            shuffled = self.rng.permutation(n)
            # 收集该组的所有 batch, 若末尾出现 size==1 的残余则并入前一个 batch
            group_batches = []
            for i in range(0, n, self.batch_size):
                batch_idx = indices[shuffled[i:i + self.batch_size]]
                group_batches.append(batch_idx)
            if len(group_batches) >= 2 and len(group_batches[-1]) == 1:
                # 把孤立的 1 个分子合并到前一个 batch
                group_batches[-2] = np.concatenate([group_batches[-2], group_batches[-1]])
                group_batches.pop()
            batches.extend(group_batches)
        single_indices = np.concatenate(list(self.single_groups.values())) if self.single_groups else np.array([], dtype=int)
        if len(single_indices) > 0:
            shuffled = self.rng.permutation(len(single_indices))
            for i in range(0, len(single_indices), self.batch_size):
                b = single_indices[shuffled[i:i + self.batch_size]]
                if len(b) >= 2:  # 跳过孤立单分子 (BatchNorm1d 需 >=2)
                    batches.append(b)
        self.rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        n_batches = 0
        for indices in self.pairable_groups.values():
            n_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        if self.single_groups:
            n_single = sum(len(v) for v in self.single_groups.values())
            n_batches += (n_single + self.batch_size - 1) // self.batch_size
        return n_batches


# ============ 3c. batch 内配对 Δ 损失 (v4 策略, 支持 TAU) ============
def batch_paired_delta_loss(pred, true, batch_indices, scaffold_keys,
                            max_pairs_per_batch=32, delta_max=None, tau=0.0, sign_weight=0.3):
    """batch 内同骨架配对 Δ 损失 (v4 策略, 加入 v5 的 TAU 噪声阈值)
    在 batch 内找同骨架的分子对, 计算 (pred_i - pred_j) vs (true_i - true_j)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, idx in enumerate(batch_indices):
        groups[scaffold_keys[idx]].append(i)
    if not groups:
        return torch.tensor(0.0, device=pred.device)
    dp_list, dt_list = [], []
    for k, local_indices in groups.items():
        if len(local_indices) < 2:
            continue
        idx_arr = np.array(local_indices)
        if len(idx_arr) > max_pairs_per_batch:
            idx_arr = np.random.choice(idx_arr, max_pairs_per_batch, replace=False)
        for i, j in itertools.combinations(idx_arr, 2):
            dp_list.append(pred[i] - pred[j])
            dt_list.append(true[i] - true[j])
    if not dp_list:
        return torch.tensor(0.0, device=pred.device)
    dp = torch.stack(dp_list)
    dt = torch.stack(dt_list).to(pred.device)
    mse_term = F.mse_loss(dp, dt)
    if tau > 0:
        m = dt.abs() > tau
        if m.any():
            sign_term = F.softplus(-dp[m] * dt[m]).mean()
        else:
            sign_term = torch.tensor(0.0, device=pred.device)
    else:
        sign_term = F.softplus(-dp * dt).mean()
    loss = mse_term + sign_weight * sign_term
    if delta_max is not None:
        excess = F.relu(dp.abs() - delta_max)
        loss = loss + 0.5 * (excess ** 2).mean()
    return loss


# ============ 4. 幅度封顶 (正则项, 非输出变换) ============
class CappedOutputHead(nn.Module):
    """标识类: 启用 Δ 幅度正则 (不直接封顶输出)
    正确的封顶需要在 delta 分支上, 而非整个输出。
    此处仅作为标识, 实际封顶通过 partner_delta_loss 中的幅度正则实现。
    """
    def __init__(self, delta_max):
        super().__init__()
        self.delta_max = delta_max
        self.base = nn.Parameter(torch.tensor(0.0))

    def forward(self, model_output):
        return model_output  # 不变换输出


# ★7 修改点 (v5): 新增 partner_delta_loss (替换旧 paired_delta_loss)
# ============ 5. 伙伴配对 Δ 损失 (v5 新增) ============
def partner_delta_loss(forward_fn, outputs, partner_map, bi, rng, device,
                       tau=0.0, delta_max=None, sign_weight=0.3):
    """伙伴配对 Δ 损失: 给 batch 里每个分子随机抽一个同骨架伙伴(来自训练集),
    计算 (pred_mol - pred_partner) vs (true_mol - true_partner)。
    batch 本身保持随机, 全局水准稳定; 配对项只负责取代基差异。"""
    mol, par = [], []
    for ii in bi.tolist():
        ps = partner_map.get(int(ii))
        if ps is not None and len(ps) > 0:
            mol.append(int(ii))
            par.append(int(ps[rng.randint(len(ps))]))
    if not mol:
        return None
    both = torch.tensor(mol + par, device=device, dtype=torch.long)
    pb = forward_fn(both)
    k = len(mol)
    dp = pb[:k] - pb[k:]
    dt = outputs[both[:k]] - outputs[both[k:]]
    loss = F.mse_loss(dp, dt)
    if tau > 0:
        m = dt.abs() > tau
        if m.any():
            loss = loss + sign_weight * F.softplus(-dp[m] * dt[m]).mean()
    else:
        loss = loss + sign_weight * F.softplus(-dp * dt).mean()
    if delta_max is not None:
        loss = loss + 0.5 * (F.relu(dp.abs() - delta_max) ** 2).mean()
    return loss


# ============ 6. 模型工厂 ============
def build_model(model_name, nvl, params, device):
    """构建模型, 返回 (model, model_type)
    model_type: 'adj' 或 'pyg'
    """
    if model_name in ADJ_MODELS:
        cls = ADJ_MODELS[model_name]
        kwargs = dict(node_vec_len=nvl, hidden_dim=params['hidden_dim'],
                      n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
                      n_outputs=1, p_dropout=params['p_dropout'], mode='label')
        if model_name == 'GAT':
            kwargs['n_heads'] = 4
        model = cls(**kwargs).to(device)
        return model, 'adj'
    elif model_name in PYG_MODELS:
        # PyG 模型需要 in_channels, 延迟到数据加载后构建
        return None, 'pyg'
    else:
        raise ValueError(f"未知模型: {model_name}")


# ============ 7. 数据加载 (统一接口) ============
def load_data_for_model(model_name, task, device):
    """加载数据, 返回统一的字典:
    {
        'format': 'adj' | 'pyg',
        'outputs': tensor(n,) on device,
        'smiles': list,
        'scaffold_keys': list (调用方填充),
        'n': int,
        'nvl': int,
        # adj 格式额外:
        'node_mats': tensor, 'adj_mats': tensor,
        # pyg 格式额外:
        'data_list': list[Data],
        'in_channels': int, 'edge_dim': int,
    }
    """
    if model_name in ADJ_MODELS:
        data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)
        return {
            'format': 'adj',
            'outputs': data['outputs'],
            'smiles': data['smiles'],
            'n': data['n'],
            'nvl': data['node_vec_len'],
            'node_mats': data['node_mats'],
            'adj_mats': data['adj_mats'],
        }
    elif model_name in PYG_MODELS:
        data_list, nvl = load_pyg_format(task['dataset_path'], task['target_col'],
                                          NVL, MAX_ATOMS,
                                          ring_flag_value=RING_FLAG_VALUE, device='cpu')
        # 提取 outputs 张量 (从 data.y)
        outputs = torch.stack([d.y for d in data_list]).squeeze().to(device)
        smiles = [d.smiles for d in data_list] if hasattr(data_list[0], 'smiles') else None
        in_channels = data_list[0].x.size(-1)
        edge_dim = data_list[0].edge_attr.size(-1) if data_list[0].edge_attr is not None else 4
        return {
            'format': 'pyg',
            'outputs': outputs,
            'smiles': smiles,
            'n': len(data_list),
            'nvl': nvl,
            'data_list': data_list,
            'in_channels': in_channels,
            'edge_dim': edge_dim,
        }
    else:
        raise ValueError(f"未知模型: {model_name}")


def load_external_test_for_model(model_name, task, device, train_nvl, train_in_channels=None):
    """加载 lunci6 测试集, 返回 (test_data_dict, test_idx_tensor)"""
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source='lunci6')
    if model_name in ADJ_MODELS:
        data = load_adj_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)
        return {
            'format': 'adj',
            'outputs': data['outputs'],
            'n': data['n'],
            'nvl': data['node_vec_len'],
            'node_mats': data['node_mats'],
            'adj_mats': data['adj_mats'],
        }, torch.arange(data['n'], device=device)
    elif model_name in PYG_MODELS:
        data_list, nvl = load_pyg_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                                         ring_flag_value=RING_FLAG_VALUE, device='cpu')
        outputs = torch.stack([d.y for d in data_list]).squeeze().to(device)
        return {
            'format': 'pyg',
            'outputs': outputs,
            'n': len(data_list),
            'nvl': nvl,
            'data_list': data_list,
            'in_channels': data_list[0].x.size(-1),
            'edge_dim': data_list[0].edge_attr.size(-1) if data_list[0].edge_attr is not None else 4,
        }, torch.arange(len(data_list), device=device)


# ============ 8. 训练函数 ============
def train_model(model_name, task, device, mode='original', seed=42,
                use_scaffold_val=True, use_capped_head=False, strategy='group_batch'):
    """
    mode: 'original' (逐点MSE) | 'paired_delta' (MSE+λΔ) | 'pure_delta' (仅Δ)
    strategy: 'group_batch' (v4, 推荐) | 'partner_map' (v5)
    use_scaffold_val: True=骨架分组验证, False=随机验证
    use_capped_head: True=加tanh封顶
    """
    set_full_seed(seed)
    task_name = task['name']

    # 加载主数据集
    print(f"  加载主数据集 (model={model_name})...", flush=True)
    train_data = load_data_for_model(model_name, task, device)
    n = train_data['n']
    fmt = train_data['format']
    outputs = train_data['outputs']
    smiles_list = train_data['smiles']
    nvl = train_data['nvl']

    # 计算骨架
    if smiles_list is None:
        # PyG 数据可能没有 smiles, 从原始 CSV 读取
        df = pd.read_csv(task['dataset_path'])
        smiles_list = df['smiles'].tolist()[:n]
    print(f"  计算 Murcko 骨架...", flush=True)
    scaffold_keys = compute_scaffolds(smiles_list)
    n_unique = len(set(scaffold_keys))
    print(f"  {n} 分子, {n_unique} 唯一骨架 (format={fmt})", flush=True)

    # 验证集划分
    if use_scaffold_val:
        train_idx, val_idx = scaffold_group_split(scaffold_keys, val_ratio=0.125, seed=seed)
        val_split_type = 'scaffold_group'
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = int(0.125 * n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_split_type = 'random'
    train_idx_t = torch.tensor(train_idx, device=device)
    val_idx_t = torch.tensor(val_idx, device=device)
    print(f"  验证集划分: {val_split_type}, train={len(train_idx)}, val={len(val_idx)}", flush=True)

    # lunci6 测试集
    print(f"  加载 lunci6 测试集...", flush=True)
    test_data, test_idx = load_external_test_for_model(model_name, task, device, nvl)

    # 构建模型
    if fmt == 'adj':
        model, _ = build_model(model_name, nvl, PARAMS, device)
    else:  # pyg
        in_channels = train_data['in_channels']
        edge_dim = train_data['edge_dim']
        model = build_pyg_model(model_name, in_channels, PARAMS['hidden_dim'],
                                PARAMS['n_conv_layers'], PARAMS['p_dropout'], edge_dim).to(device)
    print(f"  模型: {model_name} (params={sum(p.numel() for p in model.parameters())})", flush=True)

    # 可选: 加封顶头
    if use_capped_head:
        delta_max = DELTA_MAX.get(task_name, 1.0)
        capped_head = CappedOutputHead(delta_max).to(device)
        with torch.no_grad():
            capped_head.base.fill_(float(outputs[train_idx_t].mean()))
        params = list(model.parameters()) + list(capped_head.parameters())
        print(f"  封顶头: Δmax={delta_max}, base init={float(capped_head.base):.4f}", flush=True)
    else:
        capped_head = None
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=PARAMS['learning_rate'],
                                 weight_decay=PARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PARAMS['patience'] // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    # 格式感知的 forward 函数
    if fmt == 'adj':
        node_mats = train_data['node_mats']
        adj_mats = train_data['adj_mats']

        def forward_model(idx):
            out = model(node_mats[idx], adj_mats[idx]).squeeze(-1)
            if capped_head is not None:
                out = capped_head(out)
            return out
    else:  # pyg
        from torch_geometric.data import Batch
        data_list = train_data['data_list']
        in_channels = train_data['in_channels']

        def forward_model(idx):
            if isinstance(idx, (list, np.ndarray)):
                idx = torch.tensor(idx, device=device, dtype=torch.long)
            elif not isinstance(idx, torch.Tensor):
                idx = torch.tensor([int(idx)], device=device, dtype=torch.long)
            samples = [data_list[int(i)] for i in idx.tolist()]
            batch = Batch.from_data_list(samples).to(device)
            # pad/truncate x to in_channels
            x = batch.x
            if x.size(-1) != in_channels:
                if x.size(-1) < in_channels:
                    pad = torch.zeros(x.size(0), in_channels - x.size(-1), device=device)
                    x = torch.cat([x, pad], dim=-1)
                else:
                    x = x[..., :in_channels]
            out = model(x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1)
            if capped_head is not None:
                out = capped_head(out)
            return out

    # 配对策略初始化
    use_group_batch = (strategy == 'group_batch' and mode in ('paired_delta', 'pure_delta'))
    use_partner_map = (strategy == 'partner_map' and mode in ('paired_delta', 'pure_delta'))

    if use_group_batch:
        sampler = GroupBatchSampler(scaffold_keys, train_idx, PARAMS['batch_size'], seed=seed)
        print(f"  策略: group_batch (v4), LAMBDA_DELTA={LAMBDA_DELTA}, WARMUP_EPOCHS={WARMUP_EPOCHS}, "
              f"TAU={TAU.get(task_name, 0)}, DELTA_MAX={DELTA_MAX.get(task_name)}", flush=True)
    elif use_partner_map:
        partner_map = build_partner_map(scaffold_keys, train_idx)
        rng_partner = np.random.RandomState(seed)
        n_pairable = len(partner_map)
        print(f"  策略: partner_map (v5), 伙伴配对: {n_pairable}/{len(train_idx)}", flush=True)
        print(f"  v5 配置: LAMBDA_DELTA={LAMBDA_DELTA}, WARMUP_EPOCHS={WARMUP_EPOCHS}, "
              f"TAU={TAU.get(task_name, 0)}, DELTA_MAX={DELTA_MAX.get(task_name)}", flush=True)
    else:
        partner_map = {}

    # 训练循环
    best_val, best_state, pcount = float('inf'), None, 0
    warmup_ended = False
    t0 = time.time()
    n_train = len(train_idx)
    bs = PARAMS['batch_size']

    for epoch in range(1, PARAMS['n_epochs'] + 1):
        # Warmup 结束时重置 early stopping (+ LR, 仅 partner_map 策略)
        if mode in ('paired_delta', 'pure_delta') and epoch == WARMUP_EPOCHS + 1 and not warmup_ended:
            warmup_ended = True
            best_val = float('inf')
            pcount = 0
            if use_partner_map:  # v5: LR也重置
                for g in optimizer.param_groups:
                    g['lr'] = PARAMS['learning_rate']
            print(f"    [Warmup结束] 重置 early stopping{' + LR' if use_partner_map else ''}, "
                  f"切换到配对Δ训练", flush=True)
        model.train()
        if capped_head is not None:
            capped_head.train()

        in_warmup = (epoch <= WARMUP_EPOCHS) and (mode in ('paired_delta', 'pure_delta'))

        # ---- 策略分支: group_batch (v4) vs partner_map (v5) vs random (warmup/original) ----
        if use_group_batch and not in_warmup:
            # v4 策略: GroupBatchSampler + batch 内配对
            for batch_idx_np in sampler:
                if len(batch_idx_np) < 2:  # BatchNorm1d 需 >=2 样本
                    continue
                bi = torch.tensor(batch_idx_np, device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                preds = forward_model(bi)
                true_batch = outputs[bi]
                loss_point = loss_fn(preds, true_batch)
                if mode == 'paired_delta':
                    loss_delta = batch_paired_delta_loss(
                        preds, true_batch, batch_idx_np, scaffold_keys,
                        delta_max=DELTA_MAX.get(task_name) if use_capped_head else None,
                        tau=TAU.get(task_name, 0.0))
                    loss = loss_point + LAMBDA_DELTA * loss_delta
                else:  # pure_delta
                    loss_delta = batch_paired_delta_loss(
                        preds, true_batch, batch_idx_np, scaffold_keys,
                        delta_max=DELTA_MAX.get(task_name) if use_capped_head else None,
                        tau=TAU.get(task_name, 0.0))
                    loss = 0.01 * loss_point + loss_delta
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # v5 策略 (partner_map) 或 warmup/original: 随机 batch
            perm = train_idx_t[torch.randperm(n_train, device=device)]
            for i0 in range(0, n_train, bs):
                bi = perm[i0:i0 + bs]
                optimizer.zero_grad(set_to_none=True)
                preds = forward_model(bi)
                loss_point = loss_fn(preds, outputs[bi])
                loss = loss_point
                if use_partner_map and not in_warmup:
                    ld = partner_delta_loss(forward_model, outputs, partner_map, bi,
                                            rng_partner, device,
                                            tau=TAU.get(task_name, 0.0),
                                            delta_max=DELTA_MAX.get(task_name) if use_capped_head else None)
                    if ld is not None:
                        if mode == 'paired_delta':
                            loss = loss_point + LAMBDA_DELTA * ld
                        else:  # pure_delta
                            loss = 0.01 * loss_point + ld
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 验证
        model.eval()
        if capped_head is not None:
            capped_head.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, len(val_idx_t), bs):
                bi = val_idx_t[i:i + bs]
                vp.append(forward_model(bi))
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss_point = loss_fn(vp, vt).item()

            # warmup 后, 用组合 val_loss (逐点 + Δ) 做 early stopping
            if mode in ('paired_delta', 'pure_delta') and not in_warmup:
                val_scaffold = [scaffold_keys[i] for i in val_idx]
                from collections import defaultdict
                val_groups = defaultdict(list)
                for i, k in enumerate(val_scaffold):
                    val_groups[k].append(i)
                dp_list, dt_list = [], []
                for k, gi in val_groups.items():
                    if len(gi) < 2:
                        continue
                    gi = np.array(gi)
                    if len(gi) > 20:
                        gi = np.random.choice(gi, 20, replace=False)
                    for i, j in itertools.combinations(gi, 2):
                        dp_list.append(vp[i] - vp[j])
                        dt_list.append(vt[i] - vt[j])
                if dp_list:
                    dp = torch.stack(dp_list)
                    dt = torch.stack(dt_list)
                    vloss_delta = F.mse_loss(dp, dt).item()
                    vloss = vloss_point + LAMBDA_DELTA * vloss_delta
                else:
                    vloss = vloss_point
            else:
                vloss = vloss_point
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if capped_head is not None:
                best_state['capped_head.base'] = capped_head.base.clone()
            pcount = 0
        else:
            pcount += 1
        if pcount >= PARAMS['patience']:
            break
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: val_loss={vloss:.6f} (best={best_val:.6f})", flush=True)

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items() if not k.startswith('capped_head')})
        if capped_head is not None and 'capped_head.base' in best_state:
            capped_head.base.data = best_state['capped_head.base']

    # ============ 评估 lunci6 ============
    model.eval()
    if capped_head is not None:
        capped_head.eval()
    te_outputs = test_data['outputs']

    if fmt == 'adj':
        te_node_mats = test_data['node_mats']
        te_adj_mats = test_data['adj_mats']
        preds, trues = [], []
        with torch.no_grad():
            for i in range(0, len(test_idx), bs):
                bi = test_idx[i:i + bs]
                te_nvl = te_node_mats[bi].shape[-1]
                batch_nm = te_node_mats[bi]
                if te_nvl != nvl:
                    if te_nvl < nvl:
                        pad = torch.zeros(*batch_nm.shape[:-1], nvl - te_nvl, device=device)
                        batch_nm = torch.cat([batch_nm, pad], dim=-1)
                    else:
                        batch_nm = batch_nm[..., :nvl]
                p = model(batch_nm, te_adj_mats[bi]).squeeze(-1)
                if capped_head is not None:
                    p = capped_head(p)
                preds.append(p)
                trues.append(te_outputs[bi])
    else:  # pyg
        from torch_geometric.data import Batch
        te_data_list = test_data['data_list']
        in_channels = train_data['in_channels']
        preds, trues = [], []
        with torch.no_grad():
            for i in range(0, len(test_idx), bs):
                bi = test_idx[i:i + bs]
                samples = [te_data_list[int(j)] for j in bi.tolist()]
                batch = Batch.from_data_list(samples).to(device)
                x = batch.x
                if x.size(-1) != in_channels:
                    if x.size(-1) < in_channels:
                        pad = torch.zeros(x.size(0), in_channels - x.size(-1), device=device)
                        x = torch.cat([x, pad], dim=-1)
                    else:
                        x = x[..., :in_channels]
                p = model(x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1)
                if capped_head is not None:
                    p = capped_head(p)
                preds.append(p)
                trues.append(te_outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2, mae, rmse = compute_metrics(true, pred)

    # 同时评估训练集 (检查是否过拟合)
    with torch.no_grad():
        tr_preds, tr_trues = [], []
        for i in range(0, len(train_idx_t), bs):
            bi = train_idx_t[i:i + bs]
            p = forward_model(bi)
            tr_preds.append(p)
            tr_trues.append(outputs[bi])
    tr_pred = torch.cat(tr_preds).cpu().numpy()
    tr_true = torch.cat(tr_trues).cpu().numpy()
    tr_r2, tr_mae, tr_rmse = compute_metrics(tr_true, tr_pred)

    # ★10 修改点 (v5): mode_label 加版本后缀 + λ 值 + 策略
    strat_tag = f"_{strategy}" if mode in ('paired_delta', 'pure_delta') else ""
    lambda_tag = f"_lam{LAMBDA_DELTA}" if mode in ('paired_delta', 'pure_delta') else ""
    warmup_tag = f"_wu{WARMUP_EPOCHS}" if mode in ('paired_delta', 'pure_delta') else ""
    mode_label = f"{model_name}_{mode}{'_capped' if use_capped_head else ''}{'_scaffoldval' if use_scaffold_val else '_randval'}{strat_tag}{lambda_tag}{warmup_tag}_v5"
    out_dir = os.path.join(OUTPUT_DIR, task_name, mode_label)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'true': true, 'pred': pred}).to_csv(
        os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

    print(f"  [{mode_label}/{task_name}] lunci6: R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Train: R²={tr_r2:.4f} MAE={tr_mae:.4f} | Time={train_time:.0f}s", flush=True)

    del model, optimizer, scheduler
    if capped_head is not None:
        del capped_head
    torch.cuda.empty_cache()

    return {
        'task': task_name, 'model': model_name, 'mode': mode_label,
        'lunci6_r2': r2, 'lunci6_mae': mae, 'lunci6_rmse': rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae,
        'train_time_sec': train_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA')
    parser.add_argument('--models', type=str, default='MPNN',
                        help=f'逗号分隔模型名, 可选: {",".join(ALL_MODELS)}')
    parser.add_argument('--modes', type=str, default='paired_delta',
                        help='逗号分隔: original,paired_delta,pure_delta,capped_delta 或 all')
    parser.add_argument('--lambda_delta', type=float, default=None,
                        help='覆盖 LAMBDA_DELTA (默认 2.0, 建议 0.5-1.0 起步)')
    parser.add_argument('--warmup', type=int, default=None,
                        help='覆盖 WARMUP_EPOCHS (默认 15)')
    parser.add_argument('--strategy', type=str, default='group_batch',
                        choices=['group_batch', 'partner_map'],
                        help='配对策略: group_batch (v4, 推荐) 或 partner_map (v5)')
    args = parser.parse_args()

    # CLI 覆盖全局常量
    global LAMBDA_DELTA, WARMUP_EPOCHS
    if args.lambda_delta is not None:
        LAMBDA_DELTA = args.lambda_delta
    if args.warmup is not None:
        WARMUP_EPOCHS = args.warmup

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"v5 配置: LAMBDA_DELTA={LAMBDA_DELTA}, WARMUP_EPOCHS={WARMUP_EPOCHS}, "
          f"DELTA_MAX={DELTA_MAX}, TAU={TAU}", flush=True)

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]
    model_list = args.models.split(',')

    # 解析模式
    if args.modes == 'all':
        mode_configs = [
            ('original', False, False),
            ('paired_delta', True, False),
            ('paired_delta', True, True),  # capped
            ('pure_delta', True, False),
        ]
    else:
        mode_configs = []
        for m in args.modes.split(','):
            if m == 'original':
                mode_configs.append(('original', False, False))
            elif m == 'paired_delta':
                mode_configs.append(('paired_delta', True, False))
            elif m == 'capped_delta':
                mode_configs.append(('paired_delta', True, True))
            elif m == 'pure_delta':
                mode_configs.append(('pure_delta', True, False))

    all_results = []
    for task in task_list:
        print(f"\n{'#' * 60}\n# 任务: {task['name']}\n{'#' * 60}", flush=True)
        for model_name in model_list:
            for mode, use_sv, use_cap in mode_configs:
                label = f"{model_name}/{mode}{'_capped' if use_cap else ''}/{args.strategy}"
                print(f"\n--- {label} ---", flush=True)
                try:
                    res = train_model(model_name, task, device, mode=mode, seed=42,
                                      use_scaffold_val=use_sv, use_capped_head=use_cap,
                                      strategy=args.strategy)
                    res['config'] = label
                    res['strategy'] = args.strategy
                    all_results.append(res)
                    # 增量保存 (防止中途崩溃丢失结果)
                    pd.DataFrame(all_results).to_csv(
                        os.path.join(OUTPUT_DIR, 'v5_multimodel_summary.csv'), index=False)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{label}/{task['name']}] 失败: {e}", flush=True)

    # 汇总
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'v5_multimodel_summary.csv'), index=False)
        print(f"\n{'=' * 60}\n汇总\n{'=' * 60}")
        cols = ['task', 'model', 'config', 'mode', 'lunci6_r2', 'lunci6_mae', 'lunci6_rmse',
                'train_r2', 'train_mae', 'train_time_sec']
        print(df[cols].to_string(index=False))
        print(f"\n已保存: {OUTPUT_DIR}/v5_multimodel_summary.csv", flush=True)


if __name__ == '__main__':
    main()
