"""
共享训练/评估工具 (四阶段实验共用)

提供:
  - set_full_seed:      完整随机种子设置
  - train_one:          统一训练循环 (支持自定义/PyG模型)
  - evaluate:           评估并返回指标 + 预测值
  - save_results:       保存 summary.csv / cv_results.csv / test_predictions.csv / parity_plot
  - run_experiment:     完整单次实验 (5折CV + final模型 + 测试)
  - compute_estar:      E* = median(best_epochs) 工具
  - verify_split_invariants: 实验后自动验证 split 不变性 + group 无泄漏
"""
import os
import csv
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aroma_dps.data.tasks import compute_metrics, DEFAULT_SEED


def set_full_seed(seed):
    """设置完整随机种子 (torch + numpy + random + cudnn)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def compute_estar(best_epochs, max_epoch):
    """E* = median(best_epochs)。
    若 best_epochs 为空或全 >= max_epoch, 警告可能撞上限。
    """
    arr = np.asarray(best_epochs, dtype=int)
    if len(arr) == 0:
        raise ValueError("compute_estar: best_epochs 为空")
    estar = int(np.median(arr))
    ceiling_hits = int((arr >= max_epoch - 1).sum())  # 容忍 ±1 抖动
    return estar, {'estar': estar,
                   'best_epochs': arr.tolist(),
                   'max_epoch': int(max_epoch),
                   'n_ceiling_hits': ceiling_hits,
                   'n_folds': len(arr)}


# ============== 自定义模型 (adj 格式) 训练 ==============
def train_custom_model(model, params, data, train_idx, val_idx, device,
                       n_epochs=200, patience=30, seed=42):
    """训练单个自定义 GNN (adj 格式, 支持 ring_indices)"""
    set_full_seed(seed)
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    ring_indices = data['ring_indices']
    outputs = data['outputs']
    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    def forward_batch(bi):
        # 修复 batch=1 时 squeeze -> 0-dim 问题 (docs P0 #4)
        return model(node_mats[bi], adj_mats[bi], ring_indices=ring_indices[bi]).reshape(-1)

    best_val, best_state, pcount, best_epoch = float('inf'), None, 0, 1
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = forward_batch(bi)
            loss = loss_fn(preds, outputs[bi].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(forward_batch(bi)); vt.append(outputs[bi].reshape(-1))
            vp = torch.cat(vp, 0); vt = torch.cat(vt, 0)
        # [P1-11] 选模型按验证 MAE (反向传播仍用 MSE)
        vmae = (vp - vt).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_val:
            best_val = vmae; best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}; pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    del optimizer, scheduler
    return model


@torch.no_grad()
def eval_custom_model(model, data, idx, batch_size, device):
    model.eval()
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    ring_indices = data['ring_indices']
    outputs = data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        p = model(node_mats[bi], adj_mats[bi], ring_indices=ring_indices[bi]).reshape(-1)
        preds.append(p); trues.append(outputs[bi].reshape(-1))
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return compute_metrics(true, pred), pred, true


# ============== PyG 模型训练 ==============
def train_pyg_model(model, params, data_list, train_idx, val_idx, device,
                    n_epochs=200, patience=30, seed=42):
    from torch_geometric.loader import DataLoader
    set_full_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    train_subset = [data_list[i] for i in train_idx]
    val_subset = [data_list[i] for i in val_idx]
    train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)

    best_val, best_state, pcount, best_epoch = float('inf'), None, 0, 1
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ring_mask = getattr(batch, 'ring_mask', None)
            preds = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr,
                          ring_mask=ring_mask).reshape(-1)
            loss = loss_fn(preds, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                ring_mask = getattr(batch, 'ring_mask', None)
                vp.append(model(batch.x, batch.edge_index, batch.batch, batch.edge_attr,
                                ring_mask=ring_mask).reshape(-1))
                vt.append(batch.y.reshape(-1))
            vp = torch.cat(vp, 0); vt = torch.cat(vt, 0)
        vmae = (vp - vt).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_val:
            best_val = vmae; best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}; pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    del optimizer, scheduler
    return model


@torch.no_grad()
def eval_pyg_model(model, data_list, idx, batch_size, device):
    from torch_geometric.loader import DataLoader
    model.eval()
    loader = DataLoader([data_list[i] for i in idx], batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        ring_mask = getattr(batch, 'ring_mask', None)
        preds.append(model(batch.x, batch.edge_index, batch.batch, batch.edge_attr,
                           ring_mask=ring_mask).reshape(-1))
        trues.append(batch.y.reshape(-1))
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return compute_metrics(true, pred), pred, true


# ============== 结果保存 ==============
def save_results(out_dir, model_name, config_name, task_name, n_total,
                 cv_rows, train_metrics, test_metrics, te_pred, te_true,
                 train_time_sec, extra=None):
    """保存 summary.csv / cv_results.csv / test_predictions.csv / parity_plot.png"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    tr_r2, tr_mae, tr_rmse = train_metrics
    te_r2, te_mae, te_rmse = test_metrics

    cv_r2 = np.mean([r['r2'] for r in cv_rows])
    cv_r2_std = np.std([r['r2'] for r in cv_rows], ddof=1)
    cv_mae = np.mean([r['mae'] for r in cv_rows])
    cv_rmse = np.mean([r['rmse'] for r in cv_rows])

    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['config', config_name])
        w.writerow(['task', task_name]); w.writerow(['n', n_total])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', tr_r2]); w.writerow(['train_mae', tr_mae]); w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2]); w.writerow(['test_mae', te_mae]); w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time_sec])
        if extra:
            for k, v in extra.items():
                w.writerow([k, v])

    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(te_true, te_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(te_true.min(), te_pred.min()), max(te_true.max(), te_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{task_name} - {model_name}/{config_name}\nTest R2={te_r2:.4f}, MAE={te_mae:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {
        'model': model_name, 'config': config_name, 'task': task_name, 'n': n_total,
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae, 'train_rmse': tr_rmse,
        'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'train_time_sec': train_time_sec,
    }


# ============== [0831 重构] 实验后自动验证 ==============
def verify_split_invariants(splits, groups, label='experiment'):
    """统一验证项 (smoke test 也调用):
       1. train_idx / test_idx 不重叠
       2. 每个 fold 的 tr/va 不重叠
       3. 每对分组的 group(分子) 不重叠 (canonical SMILES)
       4. 整体覆盖 train_idx + test_idx == len(groups)
    """
    train_idx = np.asarray(splits['train_idx'])
    test_idx = np.asarray(splits['test_idx'])
    if len(np.intersect1d(train_idx, test_idx)):
        raise AssertionError(f"[{label}] train/test 索引重叠")

    groups = np.asarray(groups)
    a = set(groups[train_idx].tolist())
    b = set(groups[test_idx].tolist())
    if a & b:
        raise AssertionError(f"[{label}] train/test 分子泄漏: {len(a & b)}")

    for k, (tr, va) in enumerate(splits['folds']):
        if len(np.intersect1d(tr, va)):
            raise AssertionError(f"[{label}] fold{k+1} tr/va 索引重叠")
        ga = set(groups[tr].tolist()); gb = set(groups[va].tolist())
        if ga & gb:
            raise AssertionError(f"[{label}] fold{k+1} 分子泄漏: {len(ga & gb)}")

    if len(np.union1d(train_idx, test_idx)) != len(groups):
        raise AssertionError(f"[{label}] train+test ≠ 总样本数")

    return True
