"""
第二层: 通用图神经网络基线实验 (label编码) — 修复版

修复项:
  - M5: 保存 test_predictions.csv (true/pred 两列)
  - M6: summary.csv 增加 train_time_sec 字段
  - m1: 设置完整随机种子 (torch + numpy + random)
  - m2: 设置 cudnn.deterministic
  - m4: 路径配置化 (从 constants.py 读取)
  - m7: CV 标准差用 ddof=1 (样本标准差)
"""
import os
import sys
import csv
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))
from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)

from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED
from common.graph_data import load_adj_format, load_pyg_format
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
PYG_MODELS = ['AttentiveFP', 'DMPNN']


def set_full_seed(seed):
    """m1: 设置完整随机种子 (torch + numpy + random)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== 自定义模型 (adj 格式) 训练 ==============
def train_custom_model(model_cls, params, data, train_idx, val_idx, device,
                       mode='label', n_epochs=200, patience=30, seed=42):
    """训练单个自定义 GNN (GPU 张量批索引)"""
    # m1: 完整随机种子
    set_full_seed(seed)
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    mask_mats, ring_indices = data['mask_mats'], data['ring_indices']
    outputs = data['outputs']
    nvl, max_atoms = data['node_vec_len'], data['max_atoms']
    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    kwargs = dict(node_vec_len=nvl, hidden_dim=params['hidden_dim'],
                  n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
                  n_outputs=1, p_dropout=params['p_dropout'], mode=mode)
    if model_cls is GATModel:
        kwargs['n_heads'] = 4
    model = model_cls(**kwargs).to(device)

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
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze()
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model


@torch.no_grad()
def eval_custom_model(model, data, idx, batch_size, device):
    model.eval()
    node_mats, adj_mats, outputs = data['node_mats'], data['adj_mats'], data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        preds.append(model(node_mats[bi], adj_mats[bi]).squeeze())
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return compute_metrics(true, pred), pred, true  # M5: 返回 pred 和 true


# ============== PyG 模型 训练 ==============
def train_pyg_model(model_name, params, data_list, train_idx, val_idx, device,
                    n_epochs=200, patience=30, seed=42):
    from torch_geometric.loader import DataLoader
    from baseline_gnn.code.pyg_models import build_pyg_model
    # m1: 完整随机种子
    set_full_seed(seed)

    in_channels = data_list[0].x.size(-1)
    edge_dim = data_list[0].edge_attr.size(-1) if data_list[0].edge_attr is not None else 4
    model = build_pyg_model(model_name, in_channels, params['hidden_dim'],
                            params['n_conv_layers'], params['p_dropout'], edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    train_subset = [data_list[i] for i in train_idx]
    val_subset = [data_list[i] for i in val_idx]
    train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1)
            loss = loss_fn(preds, batch.y.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                vp.append(model(batch.x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1))
                vt.append(batch.y.squeeze(-1))
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model, val_loader


@torch.no_grad()
def eval_pyg_model(model, data_list, idx, batch_size, device):
    from torch_geometric.loader import DataLoader
    model.eval()
    loader = DataLoader([data_list[i] for i in idx], batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        preds.append(model(batch.x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1))
        trues.append(batch.y.squeeze(-1))
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return compute_metrics(true, pred), pred, true  # M5: 返回 pred 和 true


# ============== 统一运行入口 ==============
def run_model_on_task(model_name, task, params, device, output_root,
                      n_epochs=200, patience=30, seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    name = task['name']
    out_dir = os.path.join(output_root, name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n任务: {name} | 模型: {model_name}\n{'='*60}")

    nvl, max_atoms = 60, 75
    t_total_start = time.time()  # M6: 记录总训练时间

    if model_name in CUSTOM_MODELS:
        data = load_adj_format(task['dataset_path'], task['target_col'], nvl, max_atoms,
                               ring_flag_value=10, device=device)
        n = data['n']
        splits = canonical_splits(n, seed=seed)
        test_idx, cv_folds, final_tr, final_va = splits
        test_idx = torch.tensor(test_idx, device=device)
        final_tr = torch.tensor(final_tr, device=device)
        final_va = torch.tensor(final_va, device=device)

        cv_rows = []
        for fold, (tr, va) in enumerate(cv_folds):
            tr_t = torch.tensor(tr, device=device); va_t = torch.tensor(va, device=device)
            t0 = time.time()
            model = train_custom_model(CUSTOM_MODELS[model_name], params, data, tr_t, va_t,
                                       device, mode='label', n_epochs=n_epochs, patience=patience, seed=seed+fold)
            (r2, mae, rmse), _, _ = eval_custom_model(model, data, va_t, params['batch_size'], device)
            print(f"  Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} ({time.time()-t0:.0f}s)")
            cv_rows.append({'fold': fold+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
            del model; torch.cuda.empty_cache()

        final_model = train_custom_model(CUSTOM_MODELS[model_name], params, data, final_tr, final_va,
                                         device, mode='label', n_epochs=n_epochs, patience=patience, seed=seed)
        (tr_r2, tr_mae, tr_rmse), _, _ = eval_custom_model(final_model, data, final_tr, params['batch_size'], device)
        (te_r2, te_mae, te_rmse), te_pred, te_true = eval_custom_model(final_model, data, test_idx, params['batch_size'], device)
        torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        del final_model; torch.cuda.empty_cache()

    elif model_name in PYG_MODELS:
        data_list, _ = load_pyg_format(task['dataset_path'], task['target_col'], nvl, max_atoms,
                                       ring_flag_value=10, device='cpu')
        n = len(data_list)
        splits = canonical_splits(n, seed=seed)
        test_idx, cv_folds, final_tr, final_va = splits

        cv_rows = []
        for fold, (tr, va) in enumerate(cv_folds):
            t0 = time.time()
            model, _ = train_pyg_model(model_name, params, data_list, tr, va, device,
                                       n_epochs=n_epochs, patience=patience, seed=seed+fold)
            (r2, mae, rmse), _, _ = eval_pyg_model(model, data_list, va, params['batch_size'], device)
            print(f"  Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} ({time.time()-t0:.0f}s)")
            cv_rows.append({'fold': fold+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
            del model; torch.cuda.empty_cache()

        final_model, _ = train_pyg_model(model_name, params, data_list, final_tr, final_va, device,
                                         n_epochs=n_epochs, patience=patience, seed=seed)
        (tr_r2, tr_mae, tr_rmse), _, _ = eval_pyg_model(final_model, data_list, final_tr, params['batch_size'], device)
        (te_r2, te_mae, te_rmse), te_pred, te_true = eval_pyg_model(final_model, data_list, test_idx, params['batch_size'], device)
        torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        del final_model; torch.cuda.empty_cache()
    else:
        raise ValueError(model_name)

    train_time_sec = time.time() - t_total_start  # M6: 总训练时间
    # m7: ddof=1
    cv_r2 = np.mean([r['r2'] for r in cv_rows]); cv_r2_std = np.std([r['r2'] for r in cv_rows], ddof=1)
    cv_mae = np.mean([r['mae'] for r in cv_rows]); cv_rmse = np.mean([r['rmse'] for r in cv_rows])
    print(f"  [{model_name}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | Test: R2={te_r2:.4f} MAE={te_mae:.4f} RMSE={te_rmse:.4f} | Time: {train_time_sec:.0f}s")

    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    # M5: 保存 test_predictions.csv
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', name]); w.writerow(['n', n])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', tr_r2]); w.writerow(['train_mae', tr_mae]); w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2]); w.writerow(['test_mae', te_mae]); w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time_sec])  # M6

    # 散点图
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(te_true, te_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(te_true.min(), te_pred.min()), max(te_true.max(), te_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{name} - {model_name}\nTest R2={te_r2:.4f}, MAE={te_mae:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {'model': model_name, 'task': name, 'n': n,
            'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'train_r2': tr_r2, 'train_mae': tr_mae, 'train_rmse': tr_rmse,
            'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
            'train_time_sec': train_time_sec}


def main():
    parser = argparse.ArgumentParser(description='第二层:通用GNN基线 (修复版)')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/results/layer2_gnn')
    parser.add_argument('--models', type=str, default='all',
                        help='逗号分隔: GNN,GIN,GAT,MPNN,GraphSAGE,AttentiveFP,DMPNN 或 all')
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()

    # m2: cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS); params['n_epochs'] = args.n_epochs; params['patience'] = args.patience

    all_models = list(CUSTOM_MODELS.keys()) + PYG_MODELS
    model_list = all_models if args.models == 'all' else args.models.split(',')
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        for m in model_list:
            try:
                res = run_model_on_task(m, task, params, device, args.output_dir,
                                        args.n_epochs, args.patience, args.seed)
                all_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[{m}|{task['name']}] 失败: {e}")

    cols = ['task', 'model', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
            'test_r2', 'test_mae', 'test_rmse', 'train_time_sec']
    df_out = pd.DataFrame(all_results)[cols]
    print(f"\n{'='*60}\n第二层GNN基线汇总\n{'='*60}")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(args.output_dir, 'all_gnn_summary.csv'), index=False)


if __name__ == '__main__':
    main()
