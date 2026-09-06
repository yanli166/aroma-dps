"""
三任务独立评估: HOMA(original), HOMA(collet), MBCO(lunci2)
方法论: 8/2划分测试集 -> 80%做5折CV -> 最终在20%测试集上评估
使用GNN-label模型，默认超参（不做超参优化）
"""
import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from unified_models.common.graphs import process_and_save_data
from unified_models.gnn.model import GNNModel


# 默认超参（与0427实验一致）
DEFAULT_PARAMS = {
    'n_conv_layers': 3,
    'n_hidden_layers': 2,
    'hidden_dim': 128,
    'learning_rate': 0.001,
    'p_dropout': 0.2,
    'batch_size': 64,
    'weight_decay': 1e-5,
}


def load_data_to_tensor(dataset_path, node_vec_len, max_atoms, target_col, device,
                        ring_flag_value=10):
    """加载数据为GPU张量

    ring_flag_value: 目标环原子标记值 (默认 10, 与原 GNN-label 一致; 设为 1 做 ablation)
    """
    data = process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col,
                                 ring_flag_value=ring_flag_value)
    n = len(data['node_mats'])
    node_mats = torch.tensor(np.array(data['node_mats']), dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()
    return node_mats, adj_mats, outputs, n


def train_model(params, node_mats, adj_mats, outputs, train_idx, val_idx,
                device, n_epochs=200, patience=30, seed=42):
    """训练单个模型，返回模型和验证集结果"""
    torch.manual_seed(seed)
    batch_size = params['batch_size']
    n_train = len(train_idx)
    n_val = len(val_idx)

    model = GNNModel(
        node_vec_len=node_mats.shape[-1],
        hidden_dim=params['hidden_dim'],
        n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'],
        n_outputs=1,
        p_dropout=params['p_dropout'],
        mode='label',
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(node_mats[bi], adj_mats[bi]).squeeze()
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
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
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
            vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
            vt.append(outputs[bi])
        val_pred = torch.cat(vp).cpu().numpy()
        val_true = torch.cat(vt).cpu().numpy()

    val_r2 = r2_score(val_true, val_pred)
    val_mae = float(np.abs(val_true - val_pred).mean())
    val_rmse = float(np.sqrt(((val_true - val_pred) ** 2).mean()))

    del optimizer, scheduler
    return model, best_val_loss, val_r2, val_mae, val_rmse


@torch.no_grad()
def evaluate_on_set(model, node_mats, adj_mats, outputs, idx, batch_size, device):
    """在指定索引子集上评估"""
    model.eval()
    n = len(idx)
    preds, trues = [], []
    for i in range(0, n, batch_size):
        bi = idx[i:i + batch_size]
        pred = model(node_mats[bi], adj_mats[bi]).squeeze()
        preds.append(pred)
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    return r2, mae, rmse, true, pred


def run_task(task_name, dataset_path, params, node_vec_len, max_atoms, device,
             output_root, n_epochs=200, patience=30, seed=42, ring_flag_value=10):
    """运行单个任务的完整评估: 8/2分 + 5折CV + 测试集

    ring_flag_value: 目标环原子标记值 (默认 10, 与原 GNN-label 一致; 设为 1 做 ablation)
    """
    print(f"\n{'='*70}")
    print(f"任务: {task_name}")
    print(f"数据: {dataset_path}")
    print(f"ring_flag_value: {ring_flag_value}")
    print(f"{'='*70}")

    out_dir = os.path.join(output_root, task_name)
    os.makedirs(out_dir, exist_ok=True)

    # 加载数据
    t0 = time.time()
    node_mats, adj_mats, outputs, n = load_data_to_tensor(
        dataset_path, node_vec_len, max_atoms, 'homa_value', device,
        ring_flag_value=ring_flag_value)
    print(f"数据量: {n}  ({time.time()-t0:.1f}s)")
    print(f"目标值范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}], "
          f"均值={outputs.mean().item():.4f}, 标准差={outputs.std().item():.4f}")

    # 1. 8/2划分: 80% train_val, 20% test
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_test = int(0.2 * n)
    test_idx_np = all_idx[:n_test]
    trainval_idx_np = all_idx[n_test:]

    test_idx = torch.tensor(test_idx_np, device=device)
    trainval_idx = torch.tensor(trainval_idx_np, device=device)
    print(f"划分: train_val={len(trainval_idx)}, test={len(test_idx)}")

    # 2. 5折CV on train_val
    print(f"\n--- 5折交叉验证 (在 {len(trainval_idx)} 条 train_val 上) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(trainval_idx_np)):
        fold_train = torch.tensor(trainval_idx_np[tr_idx], device=device)
        fold_val = torch.tensor(trainval_idx_np[va_idx], device=device)
        t0 = time.time()
        model, vloss, vr2, vmae, vrmse = train_model(
            params, node_mats, adj_mats, outputs, fold_train, fold_val,
            device, n_epochs, patience, seed + fold)
        print(f"  Fold {fold+1}: val_loss={vloss:.4f}, R²={vr2:.4f}, "
              f"MAE={vmae:.4f}, RMSE={vrmse:.4f}  ({time.time()-t0:.0f}s)")
        cv_results.append({'fold': fold+1, 'val_loss': vloss, 'val_r2': vr2,
                           'val_mae': vmae, 'val_rmse': vrmse})
        del model
        torch.cuda.empty_cache()

    cv_r2 = np.mean([r['val_r2'] for r in cv_results])
    cv_mae = np.mean([r['val_mae'] for r in cv_results])
    cv_rmse = np.mean([r['val_rmse'] for r in cv_results])
    cv_r2_std = np.std([r['val_r2'] for r in cv_results])
    print(f"\n  5折CV平均: R²={cv_r2:.4f}±{cv_r2_std:.4f}, "
          f"MAE={cv_mae:.4f}, RMSE={cv_rmse:.4f}")

    # 3. 最终模型: 用全部train_val训练，在test上评估
    print(f"\n--- 最终模型训练 (全部 {len(trainval_idx)} 条 train_val) ---")
    # 这里用80%训练, 20%的train_val做early stopping验证
    n_tv = len(trainval_idx_np)
    rng2 = np.random.RandomState(seed)
    tv_perm = trainval_idx_np.copy()
    rng2.shuffle(tv_perm)
    n_tv_train = int(0.875 * n_tv)  # 0.875*0.8=0.7 of total
    final_train = torch.tensor(tv_perm[:n_tv_train], device=device)
    final_val = torch.tensor(tv_perm[n_tv_train:], device=device)

    t0 = time.time()
    final_model, fvloss, fvr2, fvmae, fvrmse = train_model(
        params, node_mats, adj_mats, outputs, final_train, final_val,
        device, n_epochs, patience, seed)
    print(f"  训练完成 ({time.time()-t0:.0f}s), best_val_loss={fvloss:.4f}")

    # 测试集评估
    r2_test, mae_test, rmse_test, true_test, pred_test = evaluate_on_set(
        final_model, node_mats, adj_mats, outputs, test_idx,
        params['batch_size'], device)
    print(f"\n  >>> 测试集结果: R²={r2_test:.4f}, MAE={mae_test:.4f}, RMSE={rmse_test:.4f}")

    # 训练集评估
    r2_train, mae_train, rmse_train, true_train, pred_train = evaluate_on_set(
        final_model, node_mats, adj_mats, outputs, final_train,
        params['batch_size'], device)
    print(f"  >>> 训练集结果: R²={r2_train:.4f}, MAE={mae_train:.4f}, RMSE={rmse_train:.4f}")

    # 保存结果
    # CV结果
    pd.DataFrame(cv_results).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)

    # summary
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['task', task_name])
        w.writerow(['n_total', n])
        w.writerow(['n_train_val', len(trainval_idx)])
        w.writerow(['n_test', len(test_idx)])
        w.writerow(['cv_mean_r2', cv_r2]); w.writerow(['cv_std_r2', cv_r2_std])
        w.writerow(['cv_mean_mae', cv_mae]); w.writerow(['cv_mean_rmse', cv_rmse])
        w.writerow(['final_train_r2', r2_train]); w.writerow(['final_train_mae', mae_train])
        w.writerow(['final_train_rmse', rmse_train])
        w.writerow(['final_val_r2', fvr2]); w.writerow(['final_val_mae', fvmae])
        w.writerow(['final_val_rmse', fvrmse])
        w.writerow(['test_r2', r2_test]); w.writerow(['test_mae', mae_test])
        w.writerow(['test_rmse', rmse_test])

    # 保存预测数据
    pd.DataFrame({'true': true_test, 'pred': pred_test}).to_csv(
        os.path.join(out_dir, 'test_set_predictions.csv'), index=False)
    pd.DataFrame({'true': true_train, 'pred': pred_train}).to_csv(
        os.path.join(out_dir, 'train_set_predictions.csv'), index=False)

    # 散点图 - 测试集
    plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(true_test, pred_test, alpha=0.4, s=10, c='blue')
    lims = [min(true_test.min(), pred_test.min()), max(true_test.max(), pred_test.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True Values'); plt.ylabel('Predicted Values')
    plt.title(f'{task_name} - Test Set Parity Plot\n'
              f'R²={r2_test:.4f}, MAE={mae_test:.4f}, RMSE={rmse_test:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_parity_plot.png'), bbox_inches='tight'); plt.close()

    # 散点图 - 训练集
    plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(true_train, pred_train, alpha=0.2, s=5, c='green')
    lims = [min(true_train.min(), pred_train.min()), max(true_train.max(), pred_train.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True Values'); plt.ylabel('Predicted Values')
    plt.title(f'{task_name} - Train Set Parity Plot\n'
              f'R²={r2_train:.4f}, MAE={mae_train:.4f}, RMSE={rmse_train:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_parity_plot.png'), bbox_inches='tight'); plt.close()

    # 保存模型
    torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    del final_model
    torch.cuda.empty_cache()

    return {
        'task': task_name,
        'n': n,
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std,
        'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'test_r2': r2_test, 'test_mae': mae_test, 'test_rmse': rmse_test,
        'train_r2': r2_train, 'train_mae': mae_train, 'train_rmse': rmse_train,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='三任务独立评估')
    parser.add_argument('--node_vec_len', type=int, default=60)
    parser.add_argument('--max_atoms', type=int, default=75)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT/three_task_results')
    parser.add_argument('--ring_flag_value', type=int, default=10,
                        help='目标环原子标记值 (默认 10, 设为 1 做 ablation)')
    args = parser.parse_args()

    torch.set_num_threads(96)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")
    print(f"ring_flag_value: {args.ring_flag_value}")

    tasks = [
        ('homa_nics_original', '_PROJ_ROOT/nics-nics1zz-out-no3.csv'),
        ('homa_collet_0702', '_PROJ_ROOT/collet_homa_0702.csv'),
        ('mbco_lunci2', '_PROJ_ROOT/outcsv/lunci2-mbcout.csv'),
    ]

    all_results = []
    for task_name, path in tasks:
        if not os.path.exists(path):
            print(f"跳过 {task_name}: 文件不存在 {path}")
            continue
        result = run_task(task_name, path, DEFAULT_PARAMS,
                          args.node_vec_len, args.max_atoms, device,
                          args.output_dir, args.n_epochs, args.patience,
                          ring_flag_value=args.ring_flag_value)
        all_results.append(result)

    # 汇总表
    print(f"\n{'='*70}")
    print("三任务评估汇总")
    print(f"{'='*70}")
    print(f"{'任务':<25} {'CV R²':>12} {'Test R²':>10} {'Test MAE':>10} {'Test RMSE':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['task']:<25} {r['cv_r2']:.4f}±{r['cv_r2_std']:.4f} "
              f"{r['test_r2']:>10.4f} {r['test_mae']:>10.4f} {r['test_rmse']:>10.4f}")

    # 保存汇总
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'all_tasks_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'n', 'cv_r2_mean', 'cv_r2_std', 'cv_mae', 'cv_rmse',
                     'train_r2', 'train_mae', 'train_rmse',
                     'test_r2', 'test_mae', 'test_rmse'])
        for r in all_results:
            w.writerow([r['task'], r['n'], f"{r['cv_r2']:.6f}", f"{r['cv_r2_std']:.6f}",
                        f"{r['cv_mae']:.6f}", f"{r['cv_rmse']:.6f}",
                        f"{r['train_r2']:.6f}", f"{r['train_mae']:.6f}", f"{r['train_rmse']:.6f}",
                        f"{r['test_r2']:.6f}", f"{r['test_mae']:.6f}", f"{r['test_rmse']:.6f}"])

    print(f"\n结果保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
