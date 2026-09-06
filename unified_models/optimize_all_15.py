"""
15种模型统一超参优化 (5模型 × 3环策略)
模型: GNN, GAT, GIN, GraphSAGE, MPNN
策略: label, mask, pool

两阶段优化:
  Phase 1 (粗扫): 15种组合各做少量trials, 快速找到有希望的区域
  Phase 2 (细化): 对Top-N组合做更多trials, 充分挖掘潜力

GPU加速: 全数据常驻GPU, 消除传输开销
方法论: 8/2划分 → 80%做5折CV → 最终在20%测试集上评估
"""
import os
import sys
import csv
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import Trial
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib

# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from unified_models.common.graphs import process_and_save_data
from unified_models.gnn.model import GNNModel
from unified_models.gat.model import GATModel
from unified_models.gin.model import GINModel
from unified_models.graphsage.model import GraphSAGEModel
from unified_models.mpnn.model import MPNNModel


MODELS = {
    'gnn': GNNModel,
    'gat': GATModel,
    'gin': GINModel,
    'graphsage': GraphSAGEModel,
    'mpnn': MPNNModel,
}

STRATEGIES = ['label', 'mask', 'pool']


# ============== 数据加载 (全GPU) ==============

def load_data_to_gpu(dataset_path, node_vec_len, max_atoms, target_col, device,
                     seed=42, train_size=0.8):
    """加载图数据并全部放到 GPU 上"""
    data = process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col)
    n = len(data['node_mats'])

    node_mats = torch.tensor(np.array(data['node_mats']), dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    mask_mats = torch.tensor(np.array(data['mask_mats']), dtype=torch.float32, device=device)
    ring_indices = torch.tensor(np.array(data['ring_indices']), dtype=torch.long, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int((1 - train_size) * n)
    test_idx_np = idx[:n_test]
    trainval_idx_np = idx[n_test:]

    test_idx = torch.tensor(test_idx_np, device=device)
    trainval_idx = torch.tensor(trainval_idx_np, device=device)

    return {
        'node_mats': node_mats, 'adj_mats': adj_mats,
        'mask_mats': mask_mats, 'ring_indices': ring_indices,
        'outputs': outputs, 'n': n,
        'test_idx': test_idx, 'trainval_idx': trainval_idx,
        'trainval_idx_np': trainval_idx_np, 'test_idx_np': test_idx_np,
    }


# ============== 模型构建 ==============

def build_model(model_name, mode, node_vec_len, params, device):
    """统一的模型构建工厂"""
    kwargs = dict(
        node_vec_len=node_vec_len,
        hidden_dim=params['hidden_dim'],
        n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'],
        n_outputs=1,
        p_dropout=params['p_dropout'],
        mode=mode,
    )
    if model_name == 'gat':
        kwargs['n_heads'] = params.get('n_heads', 4)
    model = MODELS[model_name](**kwargs).to(device)
    return model


def forward_model(model, mode, node_b, adj_b, mask_b, ring_b):
    """统一的forward调用 - 根据mode传入不同参数"""
    if mode == 'mask':
        return model(node_b, adj_b, mask_mat=mask_b).squeeze(-1)
    elif mode == 'pool':
        return model(node_b, adj_b, ring_indices=ring_b).squeeze(-1)
    else:  # label
        return model(node_b, adj_b).squeeze(-1)


# ============== 训练 & 评估 ==============

def train_eval_fast(model_name, mode, params, data, train_idx, val_idx,
                    node_vec_len, device, n_epochs, patience=20, seed=42):
    """全GPU训练 - 返回验证集指标"""
    torch.manual_seed(seed)
    batch_size = params['batch_size']

    model = build_model(model_name, mode, node_vec_len, params, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    mask_mats = data['mask_mats']
    ring_indices = data['ring_indices']
    outputs = data['outputs']

    n_train = len(train_idx)
    n_val = len(val_idx)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = forward_model(model, mode,
                                 node_mats[bi], adj_mats[bi],
                                 mask_mats[bi], ring_indices[bi])
            loss = loss_fn(pred, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                pred = forward_model(model, mode,
                                     node_mats[bi], adj_mats[bi],
                                     mask_mats[bi], ring_indices[bi])
                vp.append(pred)
                vt.append(outputs[bi])
            val_pred = torch.cat(vp)
            val_true = torch.cat(vt)
            val_loss = loss_fn(val_pred, val_true).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终验证评估
    model.eval()
    with torch.no_grad():
        vp, vt = [], []
        for i in range(0, n_val, batch_size):
            bi = val_idx[i:i + batch_size]
            pred = forward_model(model, mode,
                                 node_mats[bi], adj_mats[bi],
                                 mask_mats[bi], ring_indices[bi])
            vp.append(pred)
            vt.append(outputs[bi])
        val_pred = torch.cat(vp).cpu().numpy()
        val_true = torch.cat(vt).cpu().numpy()

    val_r2 = r2_score(val_true, val_pred)
    val_mae = float(np.abs(val_true - val_pred).mean())
    val_rmse = float(np.sqrt(((val_true - val_pred) ** 2).mean()))

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_val_loss, val_mae, val_r2, val_rmse


@torch.no_grad()
def evaluate_on_set(model, mode, data, idx, batch_size, device):
    """在指定索引子集上评估"""
    model.eval()
    n = len(idx)
    preds, trues = [], []
    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    mask_mats = data['mask_mats']
    ring_indices = data['ring_indices']
    outputs = data['outputs']

    for i in range(0, n, batch_size):
        bi = idx[i:i + batch_size]
        pred = forward_model(model, mode,
                             node_mats[bi], adj_mats[bi],
                             mask_mats[bi], ring_indices[bi])
        preds.append(pred)
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    return r2, mae, rmse, true, pred


# ============== Optuna 目标函数 ==============

def make_objective(model_name, mode, data, train_idx, val_idx,
                   node_vec_len, device, n_epochs, patience):
    """创建Optuna目标函数"""
    def objective(trial):
        params = {
            'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 5),
            'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'p_dropout': trial.suggest_float('p_dropout', 0.1, 0.4),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
        if model_name == 'gat':
            params['n_heads'] = trial.suggest_categorical('n_heads', [2, 4, 8])

        t0 = time.time()
        val_loss, val_mae, val_r2, val_rmse = train_eval_fast(
            model_name, mode, params, data, train_idx, val_idx,
            node_vec_len, device, n_epochs, patience)
        elapsed = time.time() - t0
        print(f"  Trial {trial.number}: val_loss={val_loss:.4f}, R²={val_r2:.4f}, "
              f"MAE={val_mae:.4f}  ({elapsed:.0f}s)")
        return val_loss
    return objective


# ============== 最终训练 (带5折CV + 测试集) ==============

def train_final_with_cv(model_name, mode, params, data, node_vec_len, device,
                        n_epochs, patience, seed=42):
    """用最优超参做完整评估: 5折CV + 最终模型 + 测试集"""
    print(f"\n  --- 5折CV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = []
    trainval_idx_np = data['trainval_idx_np']
    trainval_idx = data['trainval_idx']

    for fold, (tr_idx, va_idx) in enumerate(kf.split(trainval_idx_np)):
        fold_train = torch.tensor(trainval_idx_np[tr_idx], device=device)
        fold_val = torch.tensor(trainval_idx_np[va_idx], device=device)
        t0 = time.time()
        vloss, vmae, vr2, vrmse = train_eval_fast(
            model_name, mode, params, data, fold_train, fold_val,
            node_vec_len, device, n_epochs, patience, seed + fold)
        print(f"    Fold {fold+1}: R²={vr2:.4f}, MAE={vmae:.4f}, RMSE={vrmse:.4f}  ({time.time()-t0:.0f}s)")
        cv_results.append({'fold': fold+1, 'val_r2': vr2, 'val_mae': vmae, 'val_rmse': vrmse})
        torch.cuda.empty_cache()

    cv_r2 = np.mean([r['val_r2'] for r in cv_results])
    cv_r2_std = np.std([r['val_r2'] for r in cv_results])
    cv_mae = np.mean([r['val_mae'] for r in cv_results])
    cv_rmse = np.mean([r['val_rmse'] for r in cv_results])
    print(f"  5折CV平均: R²={cv_r2:.4f}±{cv_r2_std:.4f}, MAE={cv_mae:.4f}, RMSE={cv_rmse:.4f}")

    # 最终模型: 87.5% train + 12.5% val from trainval
    print(f"  --- 最终模型训练 ---")
    n_tv = len(trainval_idx_np)
    rng2 = np.random.RandomState(seed)
    tv_perm = trainval_idx_np.copy()
    rng2.shuffle(tv_perm)
    n_tv_train = int(0.875 * n_tv)
    final_train = torch.tensor(tv_perm[:n_tv_train], device=device)
    final_val = torch.tensor(tv_perm[n_tv_train:], device=device)

    t0 = time.time()
    # 训练最终模型并获取模型实例
    torch.manual_seed(seed)
    batch_size = params['batch_size']
    model = build_model(model_name, mode, node_vec_len, params, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    mask_mats = data['mask_mats']
    ring_indices = data['ring_indices']
    outputs = data['outputs']
    n_train = len(final_train)
    n_val = len(final_val)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = final_train[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = forward_model(model, mode,
                                 node_mats[bi], adj_mats[bi],
                                 mask_mats[bi], ring_indices[bi])
            loss = loss_fn(pred, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = final_val[i:i + batch_size]
                pred = forward_model(model, mode,
                                     node_mats[bi], adj_mats[bi],
                                     mask_mats[bi], ring_indices[bi])
                vp.append(pred)
                vt.append(outputs[bi])
            val_pred = torch.cat(vp)
            val_true = torch.cat(vt)
            val_loss = loss_fn(val_pred, val_true).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  训练完成 ({time.time()-t0:.0f}s)")

    # 测试集评估
    test_r2, test_mae, test_rmse, test_true, test_pred = evaluate_on_set(
        model, mode, data, data['test_idx'], batch_size, device)
    train_r2, train_mae, train_rmse, train_true, train_pred = evaluate_on_set(
        model, mode, data, final_train, batch_size, device)
    print(f"  >>> 测试集: R²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
    print(f"  >>> 训练集: R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std,
        'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
        'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
        'cv_results': cv_results,
        'test_true': test_true, 'test_pred': test_pred,
        'train_true': train_true, 'train_pred': train_pred,
    }


# ============== 可视化 ==============

def save_parity_plot(out_dir, true, pred, title, r2, mae, rmse):
    """保存密度散点图"""
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    # 密度着色
    xy = np.vstack([true, pred])
    try:
        kde = gaussian_kde(xy)(xy)
        idx = kde.argsort()
        true_s, pred_s, kde_s = true[idx], pred[idx], kde[idx]
        sc = ax.scatter(true_s, pred_s, c=kde_s, s=10, alpha=0.5, cmap='viridis')
        plt.colorbar(sc, label='Density', shrink=0.8)
    except Exception:
        ax.scatter(true, pred, alpha=0.4, s=10, c='blue')
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    ax.set_xlabel('True Values'); ax.set_ylabel('Predicted Values')
    ax.set_title(f'{title}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_parity_plot.png'), bbox_inches='tight')
    plt.close()


# ============== 主流程 ==============

def run_phase1(data, node_vec_len, device, output_root, n_trials, n_epochs, patience,
               models=None, strategies=None, seed=42):
    """Phase 1: 快速粗扫所有15种组合"""
    print(f"\n{'='*80}")
    print(f"Phase 1: 快速粗扫 (trials={n_trials}, epochs={n_epochs})")
    print(f"{'='*80}")

    models = models or list(MODELS.keys())
    strategies = strategies or STRATEGIES

    # 用单一train/val划分做快速评估
    trainval_idx_np = data['trainval_idx_np']
    rng = np.random.RandomState(seed)
    perm = trainval_idx_np.copy()
    rng.shuffle(perm)
    n_tv = len(perm)
    n_tv_train = int(0.875 * n_tv)
    quick_train = torch.tensor(perm[:n_tv_train], device=device)
    quick_val = torch.tensor(perm[n_tv_train:], device=device)

    all_results = []
    for model_name in models:
        for mode in strategies:
            combo_name = f"{model_name}_{mode}"
            print(f"\n>>> 粗扫: {combo_name}")

            out_dir = os.path.join(output_root, 'phase1', combo_name)
            os.makedirs(out_dir, exist_ok=True)

            study = optuna.create_study(
                direction='minimize',
                study_name=f'phase1_{combo_name}',
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(
                make_objective(model_name, mode, data, quick_train, quick_val,
                              node_vec_len, device, n_epochs, patience),
                n_trials=n_trials, show_progress_bar=False
            )

            best_params = study.best_trial.params.copy()
            if model_name == 'gat':
                best_params['n_heads'] = study.best_trial.params.get('n_heads', 4)

            print(f"  最佳 val_loss: {study.best_trial.value:.6f}")
            print(f"  最佳参数: {best_params}")

            # 保存
            study.trials_dataframe().to_csv(
                os.path.join(out_dir, 'all_trials.csv'), index=False)
            pd.DataFrame([best_params]).to_csv(
                os.path.join(out_dir, 'best_params.csv'), index=False)
            with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)

            all_results.append({
                'combo': combo_name,
                'model': model_name,
                'mode': mode,
                'best_val_loss': study.best_trial.value,
                'best_params': best_params,
            })
            torch.cuda.empty_cache()

    # 排序汇总
    all_results.sort(key=lambda x: x['best_val_loss'])
    print(f"\n{'='*80}")
    print("Phase 1 粗扫结果排名 (按val_loss排序)")
    print(f"{'='*80}")
    print(f"{'排名':<5} {'组合':<20} {'val_loss':>12}")
    print("-" * 40)
    for i, r in enumerate(all_results):
        print(f"{i+1:<5} {r['combo']:<20} {r['best_val_loss']:>12.6f}")

    # 保存汇总
    summary_path = os.path.join(output_root, 'phase1', 'phase1_summary.csv')
    pd.DataFrame([{
        'rank': i+1, 'combo': r['combo'], 'model': r['model'], 'mode': r['mode'],
        'best_val_loss': r['best_val_loss'],
    } for i, r in enumerate(all_results)]).to_csv(summary_path, index=False)

    # 保存所有best_params
    all_params_path = os.path.join(output_root, 'phase1', 'all_best_params.json')
    with open(all_params_path, 'w') as f:
        json.dump({r['combo']: r['best_params'] for r in all_results}, f, indent=2)

    return all_results


def run_phase2(data, node_vec_len, device, output_root, phase1_results,
               top_n, n_trials, n_epochs, patience, seed=42):
    """Phase 2: 对Top-N组合做细化优化"""
    top_combos = phase1_results[:top_n]
    print(f"\n{'='*80}")
    print(f"Phase 2: 细化优化 Top-{top_n} (trials={n_trials}, epochs={n_epochs})")
    print(f"{'='*80}")

    trainval_idx_np = data['trainval_idx_np']
    rng = np.random.RandomState(seed)
    perm = trainval_idx_np.copy()
    rng.shuffle(perm)
    n_tv = len(perm)
    n_tv_train = int(0.875 * n_tv)
    quick_train = torch.tensor(perm[:n_tv_train], device=device)
    quick_val = torch.tensor(perm[n_tv_train:], device=device)

    refined_results = []
    for r in top_combos:
        combo_name = r['combo']
        model_name = r['model']
        mode = r['mode']
        print(f"\n>>> 细化: {combo_name} (Phase1 val_loss={r['best_val_loss']:.6f})")

        out_dir = os.path.join(output_root, 'phase2', combo_name)
        os.makedirs(out_dir, exist_ok=True)

        study = optuna.create_study(
            direction='minimize',
            study_name=f'phase2_{combo_name}',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(
            make_objective(model_name, mode, data, quick_train, quick_val,
                          node_vec_len, device, n_epochs, patience),
            n_trials=n_trials, show_progress_bar=False
        )

        best_params = study.best_trial.params.copy()
        if model_name == 'gat':
            best_params['n_heads'] = study.best_trial.params.get('n_heads', 4)

        print(f"  细化后最佳 val_loss: {study.best_trial.value:.6f}")
        print(f"  最佳参数: {best_params}")

        study.trials_dataframe().to_csv(
            os.path.join(out_dir, 'all_trials.csv'), index=False)
        pd.DataFrame([best_params]).to_csv(
            os.path.join(out_dir, 'best_params.csv'), index=False)
        with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)

        refined_results.append({
            'combo': combo_name,
            'model': model_name,
            'mode': mode,
            'phase1_val_loss': r['best_val_loss'],
            'phase2_val_loss': study.best_trial.value,
            'best_params': best_params,
        })
        torch.cuda.empty_cache()

    return refined_results


def run_final_eval(data, node_vec_len, device, output_root,
                   phase1_results, phase2_results, n_epochs, patience, seed=42):
    """对所有15种组合用各自最优超参做最终评估 (5折CV + 测试集)"""
    print(f"\n{'='*80}")
    print(f"最终评估: 所有15种组合 (5折CV + 测试集, epochs={n_epochs})")
    print(f"{'='*80}")

    # 合并phase1和phase2的最优参数
    phase2_map = {r['combo']: r for r in phase2_results}
    all_best = []
    for r in phase1_results:
        if r['combo'] in phase2_map:
            p2 = phase2_map[r['combo']]
            # 取phase1和phase2中更好的
            if p2['phase2_val_loss'] < r['best_val_loss']:
                best_params = p2['best_params']
                best_val_loss = p2['phase2_val_loss']
                source = 'phase2'
            else:
                best_params = r['best_params']
                best_val_loss = r['best_val_loss']
                source = 'phase1'
        else:
            best_params = r['best_params']
            best_val_loss = r['best_val_loss']
            source = 'phase1'
        all_best.append({
            'combo': r['combo'], 'model': r['model'], 'mode': r['mode'],
            'best_params': best_params, 'best_val_loss': best_val_loss,
            'source': source,
        })

    final_results = []
    for r in all_best:
        combo_name = r['combo']
        model_name = r['model']
        mode = r['mode']
        params = r['best_params']
        print(f"\n>>> 最终评估: {combo_name} (参数来源: {r['source']})")

        out_dir = os.path.join(output_root, 'final', combo_name)
        os.makedirs(out_dir, exist_ok=True)

        # 保存参数
        with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
            json.dump(params, f, indent=2)

        result = train_final_with_cv(
            model_name, mode, params, data, node_vec_len, device,
            n_epochs, patience, seed)

        # 保存结果
        pd.DataFrame(result['cv_results']).to_csv(
            os.path.join(out_dir, 'cv_results.csv'), index=False)
        pd.DataFrame({'true': result['test_true'], 'pred': result['test_pred']}).to_csv(
            os.path.join(out_dir, 'test_set_predictions.csv'), index=False)
        pd.DataFrame({'true': result['train_true'], 'pred': result['train_pred']}).to_csv(
            os.path.join(out_dir, 'train_set_predictions.csv'), index=False)

        with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['metric', 'value'])
            w.writerow(['combo', combo_name])
            w.writerow(['model', model_name])
            w.writerow(['mode', mode])
            w.writerow(['param_source', r['source']])
            w.writerow(['cv_r2', result['cv_r2']])
            w.writerow(['cv_r2_std', result['cv_r2_std']])
            w.writerow(['cv_mae', result['cv_mae']])
            w.writerow(['cv_rmse', result['cv_rmse']])
            w.writerow(['train_r2', result['train_r2']])
            w.writerow(['train_mae', result['train_mae']])
            w.writerow(['train_rmse', result['train_rmse']])
            w.writerow(['test_r2', result['test_r2']])
            w.writerow(['test_mae', result['test_mae']])
            w.writerow(['test_rmse', result['test_rmse']])

        # 散点图
        save_parity_plot(out_dir, result['test_true'], result['test_pred'],
                         f'{combo_name} (Test Set)', result['test_r2'],
                         result['test_mae'], result['test_rmse'])

        final_results.append({
            'combo': combo_name, 'model': model_name, 'mode': mode,
            'cv_r2': result['cv_r2'], 'cv_r2_std': result['cv_r2_std'],
            'cv_mae': result['cv_mae'], 'cv_rmse': result['cv_rmse'],
            'train_r2': result['train_r2'], 'train_mae': result['train_mae'],
            'train_rmse': result['train_rmse'],
            'test_r2': result['test_r2'], 'test_mae': result['test_mae'],
            'test_rmse': result['test_rmse'],
            'best_params': params, 'param_source': r['source'],
        })
        torch.cuda.empty_cache()

    # 汇总表
    final_results.sort(key=lambda x: -x['test_r2'])  # 按test_r2降序
    summary_path = os.path.join(output_root, 'final', 'final_summary.csv')
    pd.DataFrame([{
        'rank': i+1, 'combo': r['combo'], 'model': r['model'], 'mode': r['mode'],
        'cv_r2': r['cv_r2'], 'cv_r2_std': r['cv_r2_std'],
        'test_r2': r['test_r2'], 'test_mae': r['test_mae'], 'test_rmse': r['test_rmse'],
        'train_r2': r['train_r2'], 'param_source': r['param_source'],
    } for i, r in enumerate(final_results)]).to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print("最终评估结果排名 (按Test R²降序)")
    print(f"{'='*80}")
    print(f"{'排名':<5} {'组合':<20} {'CV R²':>14} {'Test R²':>10} {'Test MAE':>10} {'Test RMSE':>10}")
    print("-" * 75)
    for i, r in enumerate(final_results):
        print(f"{i+1:<5} {r['combo']:<20} {r['cv_r2']:.4f}±{r['cv_r2_std']:.4f} "
              f"{r['test_r2']:>10.4f} {r['test_mae']:>10.4f} {r['test_rmse']:>10.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='15种模型统一超参优化')
    parser.add_argument('--dataset_path', type=str,
                        default='_PROJ_ROOT/nics-nics1zz-out-no3.csv')
    parser.add_argument('--target_col', type=str, default='homa_value')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT/unified_optuna_15models')
    parser.add_argument('--node_vec_len', type=int, default=60)
    parser.add_argument('--max_atoms', type=int, default=75)
    # Phase 1
    parser.add_argument('--phase1_trials', type=int, default=12,
                        help='Phase1 每个组合的trials数')
    parser.add_argument('--phase1_epochs', type=int, default=50,
                        help='Phase1 每个trial的epochs数')
    parser.add_argument('--phase1_patience', type=int, default=12)
    # Phase 2
    parser.add_argument('--top_n', type=int, default=5,
                        help='Phase2 细化的Top-N组合数')
    parser.add_argument('--phase2_trials', type=int, default=25,
                        help='Phase2 每个组合的trials数')
    parser.add_argument('--phase2_epochs', type=int, default=80,
                        help='Phase2 每个trial的epochs数')
    parser.add_argument('--phase2_patience', type=int, default=20)
    # Final
    parser.add_argument('--final_epochs', type=int, default=200,
                        help='最终评估的epochs数')
    parser.add_argument('--final_patience', type=int, default=30)
    # 过滤
    parser.add_argument('--models', nargs='+', default=None,
                        help='指定模型 (默认全部)')
    parser.add_argument('--modes', nargs='+', default=None,
                        help='指定策略 (默认全部)')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='跳过Phase1, 直接用已有参数做最终评估')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='跳过Phase2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.set_num_threads(96)

    # 1. 加载数据
    print(f"\n========== 加载数据: {args.dataset_path} ==========")
    t0 = time.time()
    data = load_data_to_gpu(args.dataset_path, args.node_vec_len,
                            args.max_atoms, args.target_col, device, seed=args.seed)
    print(f"数据集大小: {data['n']} (train_val={len(data['trainval_idx'])}, "
          f"test={len(data['test_idx'])})  ({time.time()-t0:.1f}s)")
    print(f"GPU显存: node_mats={data['node_mats'].element_size()*data['node_mats'].nelement()/1e6:.1f}MB, "
          f"adj_mats={data['adj_mats'].element_size()*data['adj_mats'].nelement()/1e6:.1f}MB")

    models = args.models or list(MODELS.keys())
    modes = args.modes or STRATEGIES

    # 2. Phase 1: 粗扫
    if not args.skip_phase1:
        phase1_results = run_phase1(
            data, args.node_vec_len, device, args.output_dir,
            args.phase1_trials, args.phase1_epochs, args.phase1_patience,
            models=models, strategies=modes, seed=args.seed)
    else:
        # 加载已有结果
        params_path = os.path.join(args.output_dir, 'phase1', 'all_best_params.json')
        if os.path.exists(params_path):
            with open(params_path) as f:
                all_params = json.load(f)
            phase1_results = []
            for combo_name, params in all_params.items():
                parts = combo_name.rsplit('_', 1)
                phase1_results.append({
                    'combo': combo_name, 'model': parts[0], 'mode': parts[1],
                    'best_val_loss': 0, 'best_params': params,
                })
            print(f"已加载Phase1参数: {len(phase1_results)} 个组合")
        else:
            print(f"错误: 未找到Phase1结果 {params_path}, 请先运行Phase1")
            return

    # 3. Phase 2: 细化
    phase2_results = []
    if not args.skip_phase2:
        phase2_results = run_phase2(
            data, args.node_vec_len, device, args.output_dir,
            phase1_results, args.top_n,
            args.phase2_trials, args.phase2_epochs, args.phase2_patience, seed=args.seed)

    # 4. 最终评估
    final_results = run_final_eval(
        data, args.node_vec_len, device, args.output_dir,
        phase1_results, phase2_results,
        args.final_epochs, args.final_patience, seed=args.seed)

    print(f"\n{'='*80}")
    print(f"全部完成! 结果保存到: {args.output_dir}")
    print(f"  Phase1 粗扫: {args.output_dir}/phase1/")
    print(f"  Phase2 细化: {args.output_dir}/phase2/")
    print(f"  最终评估:   {args.output_dir}/final/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
