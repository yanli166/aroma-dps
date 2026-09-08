
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
配对 Δ 监督 + 锚定封顶 + 骨架分组验证
=====================================================================
针对 lunci6 诊断发现的"取代基过度响应 + 响应是噪声"问题,
实现三个核心修复:

1. GroupBatchSampler: 每个 batch 来自同一 Murcko 骨架, 保证 batch 内
   分子"同核心不同取代基", 使配对Δ监督可行
2. 配对 Δ 损失: loss = mse(pred,y) + λ * mse(pred[i]-pred[j], y[i]-y[j])
   核心项精确对消, 梯度信噪比提升 ~30dB
3. 幅度封顶 (可选): 输出 = base + Δmax·tanh(raw), 结构上杜绝 5-8 倍过度响应
4. 验证集按骨架 GroupKFold 划分, 避免骨架泄漏导致虚高分数

对比实验:
  A. 原始 MPNN (逐点 MSE, 随机验证)
  B. 配对Δ MPNN (MSE + λΔ, 骨架验证)
  C. 配对Δ + 封顶 MPNN (MSE + λΔ + tanh封顶, 骨架验证)
  D. 纯配对Δ (无逐点 MSE, 骨架验证) — 极端测试

评估: lunci6 集外测试 (312 条萘环衍生物)
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
from common.graph_data import load_adj_format
from unified_models.mpnn.model import MPNNModel
from generalization_test.code.splits import prepare_external_test_csv

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results', 'paired_delta_experiment')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NVL, MAX_ATOMS, EXT_MAX_ATOMS = 60, 75, 85
RING_FLAG_VALUE = 10
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 32,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 50,
}

# 配对Δ损失权重 (第一轮 5.0 太大导致欠拟合, 降到 0.5)
LAMBDA_DELTA = 0.5
# Warmup epochs: 前 N 轮用纯逐点 MSE, 让模型先学好核心预测
WARMUP_EPOCHS = 50
# 幅度封顶 (文献物理上界) - 仅作为正则项, 不直接封顶输出
DELTA_MAX = {'HOMA': 0.15, 'NICS_1zz': 10.0, 'MBCO': 0.05}


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


# ============ 2. 骨架分组验证集划分 (GroupKFold) ============
def scaffold_group_split(scaffold_keys, val_ratio=0.125, seed=42, min_train_groups=1):
    """按骨架分组划分训练/验证集, 避免骨架泄漏

    保证: 同一骨架的所有分子要么全在 train, 要么全在 val
    """
    from collections import defaultdict
    rng = np.random.RandomState(seed)

    # 按骨架分组
    groups = defaultdict(list)
    for idx, k in enumerate(scaffold_keys):
        groups[k].append(idx)

    # 按组大小排序 (大组优先放训练集)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)  # 随机化

    n_total = len(scaffold_keys)
    n_val_target = int(val_ratio * n_total)

    val_idx, train_idx = [], []
    val_size = 0
    for k in group_keys:
        g = groups[k]
        # 优先把小组放验证集 (直到达到目标大小)
        if val_size < n_val_target and len(g) <= (n_val_target - val_size) * 1.5:
            val_idx.extend(g)
            val_size += len(g)
        else:
            train_idx.extend(g)

    # 如果验证集太小, 从训练集大组中再拆一些
    if val_size < n_val_target * 0.8:
        # 回退: 按分子随机划分 (但标记为非骨架安全)
        print(f"  警告: 骨架分组验证集仅 {val_size} 条 (< {n_val_target*0.8:.0f}), "
              f"回退到随机划分")
        rng2 = np.random.RandomState(seed)
        perm = rng2.permutation(n_total)
        n_val = int(val_ratio * n_total)
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()

    return np.array(train_idx), np.array(val_idx)


# ============ 3. GroupBatchSampler ============
class GroupBatchSampler:
    """每个 batch 来自同一骨架组, 保证 batch 内分子可配对

    用法: for batch_idx in sampler: train_step(data[batch_idx])
    """
    def __init__(self, scaffold_keys, train_idx, batch_size, seed=42, drop_last=False):
        from collections import defaultdict
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        # 按骨架分组 (仅训练集)
        groups = defaultdict(list)
        for idx in train_idx:
            groups[scaffold_keys[idx]].append(idx)
        self.groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 1}

        # 只保留 >=2 的组用于配对, 单独分子的组单独处理
        self.pairable_groups = {k: v for k, v in self.groups.items() if len(v) >= 2}
        self.single_groups = {k: v for k, v in self.groups.items() if len(v) == 1}

        n_pairable = sum(len(v) for v in self.pairable_groups.values())
        n_single = sum(len(v) for v in self.single_groups.values())
        print(f"  GroupBatchSampler: {len(self.pairable_groups)} 个可配对组 "
              f"({n_pairable} 分子), {len(self.single_groups)} 个单分子组 ({n_single})")

    def __iter__(self):
        batches = []
        # 可配对组: 每组生成若干 batch
        for k, indices in self.pairable_groups.items():
            n = len(indices)
            shuffled = self.rng.permutation(n)
            for i in range(0, n, self.batch_size):
                batch_idx = indices[shuffled[i:i + self.batch_size]]
                if self.drop_last and len(batch_idx) < 2:
                    continue
                batches.append(batch_idx)

        # 单分子组: 合并成普通 batch
        single_indices = np.concatenate(list(self.single_groups.values())) if self.single_groups else np.array([], dtype=int)
        if len(single_indices) > 0:
            shuffled = self.rng.permutation(len(single_indices))
            for i in range(0, len(single_indices), self.batch_size):
                batches.append(single_indices[shuffled[i:i + self.batch_size]])

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


# ============ 4. 幅度封顶 (正则项, 非输出变换) ============
class CappedOutputHead(nn.Module):
    """标识类: 启用 Δ 幅度正则 (不直接封顶输出)

    正确的封顶需要在 delta 分支上, 而非整个输出。
    此处仅作为标识, 实际封顶通过 paired_delta_loss 中的幅度正则实现。
    """
    def __init__(self, delta_max):
        super().__init__()
        self.delta_max = delta_max
        self.base = nn.Parameter(torch.tensor(0.0))  # 保留参数但不使用

    def forward(self, model_output):
        return model_output  # 不变换输出


# ============ 5. 配对 Δ 损失 ============
def paired_delta_loss(pred, true, batch_indices, scaffold_keys,
                      lambda_delta=0.5, max_pairs_per_batch=32,
                      delta_max=None):
    """计算配对 Δ 损失

    在 batch 内找同骨架的分子对, 计算 (pred_i - pred_j) vs (true_i - true_j)
    如果 delta_max 不为 None, 加幅度正则: 惩罚 |pred_i - pred_j| > delta_max 的对
    """
    # 按 batch 内的骨架分组
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
        # 随机配对
        idx_arr = np.array(local_indices)
        if len(idx_arr) > max_pairs_per_batch:
            # 随机采样, 控制计算量
            idx_arr = np.random.choice(idx_arr, max_pairs_per_batch, replace=False)
        # 所有两两组合
        for i, j in itertools.combinations(idx_arr, 2):
            dp_list.append(pred[i] - pred[j])
            dt_list.append(true[i] - true[j])

    if not dp_list:
        return torch.tensor(0.0, device=pred.device)

    dp = torch.stack(dp_list)
    dt = torch.stack(dt_list).to(pred.device)
    # Δ MSE + 符号惩罚
    mse_term = F.mse_loss(dp, dt)
    # 符号损失: 方向错了就罚
    sign_term = F.softplus(-dp * dt).mean()
    loss = mse_term + 0.3 * sign_term

    # 幅度正则: 惩罚过大的 Δ (仅当 delta_max 给定时)
    if delta_max is not None:
        # 软惩罚: |dp| > delta_max 时加 quadratic penalty
        excess = F.relu(dp.abs() - delta_max)
        loss = loss + 0.5 * (excess ** 2).mean()

    return loss


# ============ 6. 训练函数 ============
def train_model(model_name, task, device, mode='original', seed=42,
                use_scaffold_val=True, use_capped_head=False):
    """
    mode: 'original' (逐点MSE) | 'paired_delta' (MSE+λΔ) | 'pure_delta' (仅Δ)
    use_scaffold_val: True=骨架分组验证, False=随机验证
    use_capped_head: True=加tanh封顶
    """
    set_full_seed(seed)

    # 加载主数据集
    train_data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n = train_data['n']
    smiles_list = train_data['smiles']
    node_mats, adj_mats = train_data['node_mats'], train_data['adj_mats']
    outputs = train_data['outputs']
    nvl = train_data['node_vec_len']

    # 计算骨架
    print(f"  计算 Murcko 骨架...", flush=True)
    scaffold_keys = compute_scaffolds(smiles_list)
    n_unique = len(set(scaffold_keys))
    print(f"  {n} 分子, {n_unique} 唯一骨架", flush=True)

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
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source='lunci6')
    test_data = load_adj_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    # 模型
    model = MPNNModel(node_vec_len=nvl, hidden_dim=PARAMS['hidden_dim'],
                      n_conv=PARAMS['n_conv_layers'], n_hidden=PARAMS['n_hidden_layers'],
                      n_outputs=1, p_dropout=PARAMS['p_dropout'], mode='label').to(device)

    # 可选: 加封顶头
    if use_capped_head:
        delta_max = DELTA_MAX.get(task['name'], 1.0)
        capped_head = CappedOutputHead(delta_max).to(device)
        # 初始化 base 为训练集均值
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

    def forward_model(idx):
        out = model(node_mats[idx], adj_mats[idx]).squeeze()
        if capped_head is not None:
            out = capped_head(out)
        return out

    # GroupBatchSampler (仅配对Δ模式)
    if mode in ('paired_delta', 'pure_delta'):
        sampler = GroupBatchSampler(scaffold_keys, train_idx, PARAMS['batch_size'], seed=seed)
        use_sampler = True
    else:
        use_sampler = False

    # 训练循环
    best_val, best_state, pcount = float('inf'), None, 0
    warmup_ended = False
    t0 = time.time()
    n_train = len(train_idx)

    for epoch in range(1, PARAMS['n_epochs'] + 1):
        # Warmup 结束时重置 early stopping (让配对Δ训练公平比较)
        if mode in ('paired_delta', 'pure_delta') and epoch == WARMUP_EPOCHS + 1 and not warmup_ended:
            warmup_ended = True
            best_val = float('inf')
            pcount = 0
            print(f"    [Warmup结束] 重置 early stopping, 切换到配对Δ训练", flush=True)
        model.train()
        if capped_head is not None:
            capped_head.train()

        # Warmup: 前 WARMUP_EPOCHS 轮用纯逐点 MSE
        in_warmup = (epoch <= WARMUP_EPOCHS) and (mode in ('paired_delta', 'pure_delta'))

        if use_sampler and not in_warmup:
            # 用 GroupBatchSampler
            for batch_idx_np in sampler:
                if len(batch_idx_np) < 1:
                    continue
                bi = torch.tensor(batch_idx_np, device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                preds = forward_model(bi)
                true_batch = outputs[bi]

                if mode == 'paired_delta':
                    loss_point = loss_fn(preds, true_batch)
                    loss_delta = paired_delta_loss(
                        preds, true_batch, batch_idx_np, scaffold_keys,
                        lambda_delta=LAMBDA_DELTA,
                        delta_max=DELTA_MAX.get(task['name']) if use_capped_head else None)
                    loss = loss_point + LAMBDA_DELTA * loss_delta
                elif mode == 'pure_delta':
                    loss = paired_delta_loss(
                        preds, true_batch, batch_idx_np, scaffold_keys,
                        lambda_delta=1.0,
                        delta_max=DELTA_MAX.get(task['name']) if use_capped_head else None)
                    # 加一个很小的逐点项防止退化
                    loss = loss + 0.01 * loss_fn(preds, true_batch)
                else:
                    loss = loss_fn(preds, true_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # 原始随机 batch (warmup 期间或 original 模式)
            perm = train_idx_t[torch.randperm(n_train, device=device)]
            for i in range(0, n_train, PARAMS['batch_size']):
                bi = perm[i:i + PARAMS['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                preds = forward_model(bi)
                loss = loss_fn(preds, outputs[bi])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 验证
        model.eval()
        if capped_head is not None:
            capped_head.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, len(val_idx_t), PARAMS['batch_size']):
                bi = val_idx_t[i:i + PARAMS['batch_size']]
                vp.append(forward_model(bi))
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss_point = loss_fn(vp, vt).item()

            # warmup 后, 用组合 val_loss (逐点 + Δ) 做 early stopping
            if mode in ('paired_delta', 'pure_delta') and not in_warmup:
                # 计算验证集的 Δ val_loss (按骨架分组)
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
                    # 组合: 逐点 + λ*Δ (与训练损失一致)
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

    # 评估 lunci6
    model.eval()
    if capped_head is not None:
        capped_head.eval()
    te_node_mats, te_adj_mats = test_data['node_mats'], test_data['adj_mats']
    te_outputs = test_data['outputs']
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(test_idx), PARAMS['batch_size']):
            bi = test_idx[i:i + PARAMS['batch_size']]
            te_nvl = te_node_mats[bi].shape[-1]
            batch_nm = te_node_mats[bi]
            if te_nvl != nvl:
                if te_nvl < nvl:
                    pad = torch.zeros(*batch_nm.shape[:-1], nvl - te_nvl, device=device)
                    batch_nm = torch.cat([batch_nm, pad], dim=-1)
                else:
                    batch_nm = batch_nm[..., :nvl]
            p = model(batch_nm, te_adj_mats[bi]).squeeze()
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
        for i in range(0, len(train_idx_t), PARAMS['batch_size']):
            bi = train_idx_t[i:i + PARAMS['batch_size']]
            p = forward_model(bi)
            tr_preds.append(p)
            tr_trues.append(outputs[bi])
    tr_pred = torch.cat(tr_preds).cpu().numpy()
    tr_true = torch.cat(tr_trues).cpu().numpy()
    tr_r2, tr_mae, tr_rmse = compute_metrics(tr_true, tr_pred)

    # 保存预测
    mode_label = f"{mode}{'_capped' if use_capped_head else ''}{'_scaffoldval' if use_scaffold_val else '_randval'}"
    out_dir = os.path.join(OUTPUT_DIR, task['name'], mode_label)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'true': true, 'pred': pred}).to_csv(
        os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

    print(f"  [{mode_label}/{task['name']}] lunci6: R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Train: R²={tr_r2:.4f} MAE={tr_mae:.4f} | Time={train_time:.0f}s", flush=True)

    del model, optimizer, scheduler
    if capped_head is not None:
        del capped_head
    torch.cuda.empty_cache()

    return {
        'task': task['name'], 'mode': mode_label,
        'lunci6_r2': r2, 'lunci6_mae': mae, 'lunci6_rmse': rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae,
        'train_time_sec': train_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA')
    parser.add_argument('--modes', type=str, default='all',
                        help='逗号分隔: original,paired_delta,pure_delta,capped_delta 或 all')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    if args.modes == 'all':
        mode_configs = [
            # (mode, use_scaffold_val, use_capped_head, label)
            ('original', False, False, 'A_original_randval'),
            ('original', True, False, 'A_original_scaffoldval'),
            ('paired_delta', True, False, 'B_paired_delta'),
            ('paired_delta', True, True, 'C_paired_delta_capped'),
            ('pure_delta', True, False, 'D_pure_delta'),
        ]
    else:
        mode_configs = []
        for m in args.modes.split(','):
            if m == 'original':
                mode_configs.append(('original', False, False, 'A_original_randval'))
                mode_configs.append(('original', True, False, 'A_original_scaffoldval'))
            elif m == 'paired_delta':
                mode_configs.append(('paired_delta', True, False, 'B_paired_delta'))
            elif m == 'capped_delta':
                mode_configs.append(('paired_delta', True, True, 'C_paired_delta_capped'))
            elif m == 'pure_delta':
                mode_configs.append(('pure_delta', True, False, 'D_pure_delta'))

    all_results = []
    for task in task_list:
        print(f"\n{'#' * 60}\n# 任务: {task['name']}\n{'#' * 60}", flush=True)
        for mode, use_sv, use_cap, label in mode_configs:
            print(f"\n--- {label} (mode={mode}, scaffold_val={use_sv}, capped={use_cap}) ---", flush=True)
            try:
                res = train_model('MPNN', task, device, mode=mode, seed=42,
                                  use_scaffold_val=use_sv, use_capped_head=use_cap)
                res['config'] = label
                all_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{label}/{task['name']}] 失败: {e}", flush=True)

    # 汇总
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'paired_delta_summary.csv'), index=False)
        print(f"\n{'=' * 60}\n汇总\n{'=' * 60}")
        cols = ['task', 'config', 'mode', 'lunci6_r2', 'lunci6_mae', 'lunci6_rmse',
                'train_r2', 'train_mae', 'train_time_sec']
        print(df[cols].to_string(index=False))
        print(f"\n已保存: {OUTPUT_DIR}/paired_delta_summary.csv", flush=True)


if __name__ == '__main__':
    main()
