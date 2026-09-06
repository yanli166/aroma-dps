"""
泛化验证: 训练 + 评估模块 (适配 0716 新数据)

复用 code_end 的三层模型训练逻辑, 支持四种测试场景:
  - scaffold: 骨架拆分 (DeepChem)
  - ring_type: 芳环类型拆分
  - lunci6: lunci6 集外测试
  - lunci78: lunci78 集外测试
"""
import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))
from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)

from common.tasks import compute_metrics, DEFAULT_SEED
from common.graph_data import load_adj_format
from unified_models.gat.model import GATModel
from unified_models.gin.model import GINModel
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from unified_models.graphsage.model import GraphSAGEModel

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}
CUSTOM_MODELS = {
    'GNN': GNNModel, 'GIN': GINModel, 'GAT': GATModel,
    'MPNN': MPNNModel, 'GraphSAGE': GraphSAGEModel,
}


def set_full_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model_name, params, data, train_idx, val_idx, device,
                n_epochs=200, patience=30, seed=42):
    """训练单个 GNN (label 编码, 与 Layer 2 一致)"""
    set_full_seed(seed)
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    outputs = data['outputs']
    nvl = data['node_vec_len']
    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    kwargs = dict(node_vec_len=nvl, hidden_dim=params['hidden_dim'],
                  n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
                  n_outputs=1, p_dropout=params['p_dropout'], mode='label')
    if model_name == 'GAT':
        kwargs['n_heads'] = 4
    model = CUSTOM_MODELS[model_name](**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            if len(bi) < 2:
                continue  # 跳过 batch=1 避免 BatchNorm 崩溃
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze(-1)
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                if len(bi) < 2:
                    bi = val_idx[i:i + batch_size]  # eval 模式下 BatchNorm 用 running stats, 安全
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze(-1))
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model


@torch.no_grad()
def eval_model(model, data, idx, batch_size, device):
    """评估模型 (支持与训练数据不同的 data 字典, 用于集外测试)"""
    model.eval()
    node_mats, adj_mats, outputs = data['node_mats'], data['adj_mats'], data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        preds.append(model(node_mats[bi], adj_mats[bi]).squeeze(-1))
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2, mae, rmse = compute_metrics(true, pred)
    return r2, mae, rmse, pred, true


def run_single_test(model_name, task, params, device, test_type, split_info,
                    output_root, n_epochs=200, patience=30, seed=42):
    """执行单次泛化测试"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name = task['name']
    holdout = split_info.get('holdout', 'default')
    if test_type == 'ring_type':
        safe_holdout = holdout.replace('/', '_').replace(' ', '_')[:50]
        out_dir = os.path.join(output_root, name, test_type, safe_holdout, model_name)
    else:
        out_dir = os.path.join(output_root, name, test_type, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"任务: {name} | 模型: {model_name} | 测试: {test_type}")
    if 'holdout' in split_info:
        print(f"  Hold-out: {split_info['holdout']}")
    print(f"{'='*60}")

    if test_type in ('lunci6', 'lunci78', 'lunci10'):
        train_data = split_info['train_data']
        test_data = split_info['test_data']
        train_idx = split_info['train_idx']
        val_idx = split_info['val_idx']
        test_idx = split_info['test_idx']

        t0 = time.time()
        model = train_model(model_name, params, train_data, train_idx, val_idx,
                            device, n_epochs, patience, seed)
        train_time = time.time() - t0

        tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, train_data, train_idx,
                                                   params['batch_size'], device)
        te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, test_data, test_idx,
                                                               params['batch_size'], device)
        n_test = len(test_idx)
        n_train = len(train_idx)
        # 保存预测结果
        pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
            os.path.join(out_dir, 'test_predictions.csv'), index=False)
        del model; torch.cuda.empty_cache()
    else:
        data = split_info['data']
        train_idx = torch.tensor(split_info['train_idx'], device=device)
        val_idx = torch.tensor(split_info['val_idx'], device=device)
        test_idx = torch.tensor(split_info['test_idx'], device=device)

        t0 = time.time()
        model = train_model(model_name, params, data, train_idx, val_idx,
                            device, n_epochs, patience, seed)
        train_time = time.time() - t0

        tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, data, train_idx,
                                                   params['batch_size'], device)
        te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, data, test_idx,
                                                               params['batch_size'], device)
        n_train = len(train_idx)
        n_test = len(test_idx)
        pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
            os.path.join(out_dir, 'test_predictions.csv'), index=False)
        del model; torch.cuda.empty_cache()

    print(f"  Train R2={tr_r2:.4f} | Test R2={te_r2:.4f} MAE={te_mae:.4f} RMSE={te_rmse:.4f} "
          f"({train_time:.0f}s, n_train={n_train}, n_test={n_test})")

    # 散点图
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(te_true, te_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(te_true.min(), te_pred.min()), max(te_true.max(), te_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{name} - {model_name} ({test_type})\nTest R2={te_r2:.4f}, MAE={te_mae:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', name])
        w.writerow(['test_type', test_type]); w.writerow(['holdout', holdout])
        w.writerow(['n_train', n_train]); w.writerow(['n_test', n_test])
        w.writerow(['train_r2', tr_r2]); w.writerow(['train_mae', tr_mae]); w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2]); w.writerow(['test_mae', te_mae]); w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time])

    return {
        'task': name, 'model': model_name, 'test_type': test_type,
        'holdout': holdout, 'n_train': n_train, 'n_test': n_test,
        'train_r2': tr_r2, 'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'train_time_sec': train_time,
    }
