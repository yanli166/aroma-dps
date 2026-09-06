"""
Optuna 超参数优化: 对 Layer 3 top3 模型进行贝叶斯优化

在 Layer 3 修复初始化 bug 后, 对 top3 模型 (基于 test R²) 进行超参数搜索。
搜索空间: hidden_dim, n_conv, n_hidden, lr, dropout, batch_size, weight_decay
优化目标: 5折CV 平均 R²
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = os.path.join(PROJ_ROOT, 'code_end')
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, '..', 'data_90', 'alldata_in_3090', 'model1'))
sys.path.insert(0, '_PROJ_ROOT + "/unified_models"')

from common.tasks import TASKS, canonical_splits, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from unified_models.graphsage.model import GraphSAGEModel

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: optuna 未安装, 请运行 pip install optuna")


CUSTOM_MODELS = {'GNN': GNNModel, 'MPNN': MPNNModel, 'GraphSAGE': GraphSAGEModel}

# Layer 3 top3 模型 (基于修复后的结果, 使用 label 编码)
TOP3_MODELS = {
    'HOMA':     ['MPNN', 'GraphSAGE', 'GNN'],
    'NICS_1zz': ['MPNN', 'GNN', 'GraphSAGE'],
    'MBCO':     ['GraphSAGE', 'GNN', 'MPNN'],
}


def set_full_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial, model_name, task, data, device, seed=42):
    """Optuna 目标函数: 返回 5折CV 平均 R²"""
    # 搜索空间
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 4),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'p_dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    n_epochs = 150  # 缩短以加速搜索
    patience = 20

    n = data['n']
    splits = canonical_splits(n, seed=seed)
    _, cv_folds, _, _ = splits

    cv_r2s = []
    for fold, (tr, va) in enumerate(cv_folds):
        tr_t = torch.tensor(tr, device=device)
        va_t = torch.tensor(va, device=device)

        set_full_seed(seed + fold)
        kwargs = dict(
            node_vec_len=data['node_vec_len'],
            hidden_dim=params['hidden_dim'],
            n_conv=params['n_conv_layers'],
            n_hidden=params['n_hidden_layers'],
            n_outputs=1, p_dropout=params['p_dropout'], mode='label'
        )
        if model_name == 'GAT':
            kwargs['n_heads'] = 4
        model = CUSTOM_MODELS[model_name](**kwargs).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params['learning_rate'],
                                     weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
        loss_fn = nn.MSELoss()

        best_val, best_state, pcount = float('inf'), None, 0
        node_mats, adj_mats = data['node_mats'], data['adj_mats']
        outputs = data['outputs']
        n_train, n_val = len(tr_t), len(va_t)
        batch_size = params['batch_size']

        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = tr_t[torch.randperm(n_train, device=device)]
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
                    bi = va_t[i:i + batch_size]
                    vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
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
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = va_t[i:i + batch_size]
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
                vt.append(outputs[bi])
            vp = torch.cat(vp).cpu().numpy()
            vt = torch.cat(vt).cpu().numpy()
        r2, _, _ = compute_metrics(vt, vp)
        cv_r2s.append(r2)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    return np.mean(cv_r2s)


def run_optuna_search(task_name, model_name, device, n_trials=30, seed=42, output_dir=None):
    """对单个任务×模型运行 Optuna 搜索"""
    task = next(t for t in TASKS if t['name'] == task_name)
    data = load_adj_format(task['dataset_path'], task['target_col'],
                           60, 75, ring_flag_value=10, device=device)

    print(f"\n{'='*60}")
    print(f"Optuna 搜索: {task_name} / {model_name}")
    print(f"{'='*60}")

    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(seed=seed),
                                study_name=f'{task_name}_{model_name}')
    study.optimize(lambda trial: objective(trial, model_name, task, data, device, seed),
                   n_trials=n_trials, show_progress_bar=False)

    print(f"\n最佳 R²: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    # 用最佳参数训练最终模型并评估
    best_params = study.best_params
    params = {
        'hidden_dim': best_params['hidden_dim'],
        'n_conv_layers': best_params['n_conv_layers'],
        'n_hidden_layers': best_params['n_hidden_layers'],
        'learning_rate': best_params['lr'],
        'p_dropout': best_params['dropout'],
        'batch_size': best_params['batch_size'],
        'weight_decay': best_params['weight_decay'],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 保存搜索结果
        df = study.trials_dataframe()
        df.to_csv(os.path.join(output_dir, f'optuna_trials_{task_name}_{model_name}.csv'), index=False)
        # 保存最佳参数
        import json
        with open(os.path.join(output_dir, f'best_params_{task_name}_{model_name}.json'), 'w') as f:
            json.dump({'params': best_params, 'cv_r2': study.best_value}, f, indent=2)

    return study.best_value, best_params


def main():
    parser = argparse.ArgumentParser(description='Optuna 超参数优化')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--models', type=str, default='top3')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/optuna_search/results')
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("请先安装 optuna: pip install optuna")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    task_list = ['HOMA', 'NICS_1zz', 'MBCO'] if args.tasks == 'all' else args.tasks.split(',')

    all_results = []
    for task_name in task_list:
        if args.models == 'top3':
            model_list = TOP3_MODELS[task_name]
        else:
            model_list = args.models.split(',')

        for model_name in model_list:
            try:
                best_r2, best_params = run_optuna_search(
                    task_name, model_name, device, args.n_trials,
                    output_dir=args.output_dir)
                all_results.append({
                    'task': task_name, 'model': model_name,
                    'best_cv_r2': best_r2, **best_params
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{task_name}/{model_name}] 失败: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.output_dir, 'optuna_summary.csv'), index=False)
        print(f"\n{'='*60}\nOptuna 搜索汇总\n{'='*60}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
