"""
GNN-label 高效原位 Optuna 超参优化 + 最终训练 + 外部数据集评估
- 数据只加载一次，所有 trial 共享内存数据
- 支持 GPU
- 优化完成后用最优超参重新训练完整模型，生成散点图
- 在 collet_homa_0702.csv / lunci2-mbcout.csv 上做外部测试与独立训练评估
"""
import os
import sys
import csv
import copy
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import Trial
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import r2_score
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

from unified_models.common.graphs import process_and_save_data, GraphData, collate_graph_dataset
from unified_models.common.utils import Standardizer, train_one_epoch, evaluate
from unified_models.gnn.model import GNNModel


def make_loaders(data, node_vec_len, max_atoms, batch_size, seed=42, train_size=0.8):
    """构造 train/val DataLoader (random split)"""
    dataset = GraphData(data, node_vec_len, max_atoms)
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    split = int(np.floor(train_size * n))
    train_idx, val_idx = idx[:split], idx[split:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_graph_dataset, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_graph_dataset, num_workers=0)
    return train_loader, val_loader, train_set, val_set


def train_eval_once(params, data, node_vec_len, max_atoms, device, n_epochs,
                    seed=42, patience=25):
    """用给定超参训练并返回最佳 val_loss（用于 Optuna objective）"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, _, _ = make_loaders(
        data, node_vec_len, max_atoms, params['batch_size'], seed)

    outputs_tensor = torch.Tensor(data['outputs']).squeeze()
    std = Standardizer(outputs_tensor)

    model = GNNModel(
        node_vec_len=node_vec_len,
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
        train_loss, _, _, _, _ = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, 'label',
            max_atoms, node_vec_len, scheduler)
        val_loss, _, _, _, _, _ = evaluate(
            model, val_loader, loss_fn, device, 'label', max_atoms, node_vec_len)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 用最佳状态评估详细指标
    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_mae, val_r2, val_rmse, val_true, val_pred = evaluate(
        model, val_loader, loss_fn, device, 'label', max_atoms, node_vec_len)

    del model, optimizer, scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_val_loss, val_mae, val_r2, val_rmse, val_true, val_pred


def objective(trial: Trial, data, node_vec_len, max_atoms, device, n_epochs):
    """Optuna 目标函数"""
    params = {
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 6),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 4),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
        'p_dropout': trial.suggest_float('p_dropout', 0.1, 0.4),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    t0 = time.time()
    val_loss, val_mae, val_r2, val_rmse, _, _ = train_eval_once(
        params, data, node_vec_len, max_atoms, device, n_epochs)
    elapsed = time.time() - t0
    print(f"  Trial {trial.number}: val_loss={val_loss:.4f}, R²={val_r2:.4f}, "
          f"MAE={val_mae:.4f}, RMSE={val_rmse:.4f}  ({elapsed:.0f}s)")
    return val_loss


def train_final(params, data, node_vec_len, max_atoms, device, n_epochs,
                output_dir, seed=42, patience=30, plot=True):
    """用最优超参训练完整模型并保存所有结果和图表"""
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, train_set, val_set = make_loaders(
        data, node_vec_len, max_atoms, params['batch_size'], seed)

    outputs_tensor = torch.Tensor(data['outputs']).squeeze()
    std = Standardizer(outputs_tensor)

    model = GNNModel(
        node_vec_len=node_vec_len,
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
    best_epoch = 0
    patience_counter = 0
    epochs_list, train_losses, val_losses = [], [], []
    epoch_records = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_mae, train_r2, _, _ = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, 'label',
            max_atoms, node_vec_len, scheduler)
        val_loss, val_mae, val_r2, val_rmse, _, _ = evaluate(
            model, val_loader, loss_fn, device, 'label', max_atoms, node_vec_len)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_records.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_mae': train_mae,
            'train_r2': train_r2, 'val_loss': val_loss, 'val_mae': val_mae,
            'val_r2': val_r2, 'val_rmse': val_rmse
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{n_epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_R²={val_r2:.4f}, val_MAE={val_mae:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # 最终评估
    train_loss, train_mae, train_r2, train_rmse, train_true, train_pred = evaluate(
        model, train_loader, loss_fn, device, 'label', max_atoms, node_vec_len)
    val_loss, val_mae, val_r2, val_rmse, val_true, val_pred = evaluate(
        model, val_loader, loss_fn, device, 'label', max_atoms, node_vec_len)

    print(f"\n  最终结果:")
    print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}, RMSE={train_rmse:.4f}")
    print(f"  Val:   Loss={val_loss:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}, RMSE={val_rmse:.4f}")

    # 保存配置
    with open(os.path.join(output_dir, 'config.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'value'])
        for k, v in params.items():
            w.writerow([k, v])
        w.writerow(['n_epochs', n_epochs])
        w.writerow(['best_epoch', best_epoch])
        w.writerow(['best_val_loss', best_val_loss])

    # 保存 summary
    with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['best_val_loss', best_val_loss])
        w.writerow(['best_epoch', best_epoch])
        w.writerow(['final_train_loss', train_loss])
        w.writerow(['final_train_mae', train_mae])
        w.writerow(['final_train_r2', train_r2])
        w.writerow(['final_train_rmse', train_rmse])
        w.writerow(['final_val_loss', val_loss])
        w.writerow(['final_val_mae', val_mae])
        w.writerow(['final_val_r2', val_r2])
        w.writerow(['final_val_rmse', val_rmse])

    # 保存 epoch 记录
    with open(os.path.join(output_dir, 'epoch_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_mae', 'train_r2',
                                                'val_loss', 'val_mae', 'val_r2', 'val_rmse'])
        writer.writeheader()
        writer.writerows(epoch_records)

    # 保存预测数据
    pd.DataFrame({'true': train_true.numpy(), 'pred': train_pred.numpy()}).to_csv(
        os.path.join(output_dir, 'train_set_data.csv'), index=False)
    pd.DataFrame({'true': val_true.numpy(), 'pred': val_pred.numpy()}).to_csv(
        os.path.join(output_dir, 'val_set_data.csv'), index=False)

    if plot:
        # 损失曲线
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=1.5)
        plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=1.5)
        plt.scatter([best_epoch], [best_val_loss], color='gold', s=100, zorder=5,
                    label=f'Best Val (Epoch {best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GNN-label (Optimized) Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), bbox_inches='tight')
        plt.close()

        # 验证集散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(val_true.numpy(), val_pred.numpy(), alpha=0.4, s=8, c='blue')
        lims = [min(val_true.min(), val_pred.min()),
                max(val_true.max(), val_pred.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)')
        plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'GNN-label (Optimized) Validation Parity Plot\n'
                  f'R²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_parity_plot.png'), bbox_inches='tight')
        plt.close()

        # 训练集散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(train_true.numpy(), train_pred.numpy(), alpha=0.2, s=5, c='green')
        lims = [min(train_true.min(), train_pred.min()),
                max(train_true.max(), train_pred.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)')
        plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'GNN-label (Optimized) Train Parity Plot\n'
                  f'R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_parity_plot.png'), bbox_inches='tight')
        plt.close()

    return model, std, (train_true, train_pred), (val_true, val_pred)


@torch.no_grad()
def predict_external(model, data, node_vec_len, max_atoms, device, batch_size=128):
    """在外部数据集上做预测并返回指标"""
    dataset = GraphData(data, node_vec_len, max_atoms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_graph_dataset, num_workers=0)
    loss_fn = nn.MSELoss()
    val_loss, val_mae, val_r2, val_rmse, true, pred = evaluate(
        model, loader, loss_fn, device, 'label', max_atoms, node_vec_len)
    return val_loss, val_mae, val_r2, val_rmse, true, pred


def main():
    parser = argparse.ArgumentParser(description='GNN-label Optuna优化 + 外部评估')
    parser.add_argument('--dataset_path', type=str,
                        default= os.path.join(_PROJ_ROOT, "nics-nics1zz-out-no3.csv"))
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='每个trial的epoch数')
    parser.add_argument('--final_epochs', type=int, default=300,
                        help='最终训练的epoch数')
    parser.add_argument('--node_vec_len', type=int, default=60)
    parser.add_argument('--max_atoms', type=int, default=75)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str,
                        default= os.path.join(_PROJ_ROOT, "gnn_label_optimized"))
    args = parser.parse_args()

    torch.set_num_threads(12)
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")
    print(f"Dataset: {args.dataset_path}")

    # ========== 1. 加载原始训练数据 ==========
    print("\n========== 加载数据 ==========")
    t0 = time.time()
    data = process_and_save_data(args.dataset_path, args.node_vec_len,
                                  args.max_atoms, 'homa_value')
    print(f"原始数据集大小: {len(data['node_mats'])}  ({time.time()-t0:.1f}s)")

    # ========== 2. Optuna 优化 ==========
    print(f"\n========== Optuna 优化 ({args.n_trials} trials, {args.n_epochs} epochs/trial) ==========")
    study = optuna.create_study(
        direction='minimize',
        study_name='gnn_label_opt',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=15)
    )
    study.optimize(
        lambda trial: objective(trial, data, args.node_vec_len, args.max_atoms,
                                device, args.n_epochs),
        n_trials=args.n_trials, show_progress_bar=False
    )

    print(f"\n{'='*60}")
    print(f"Optuna 完成! 试验数: {len(study.trials)}")
    print(f"最佳 val_loss: {study.best_trial.value:.6f}")
    print("最佳参数:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, 'all_trials.csv'), index=False)
    best_params = study.best_trial.params.copy()
    pd.DataFrame([best_params]).to_csv(
        os.path.join(args.output_dir, 'best_parameters.csv'), index=False)

    # ========== 3. 最终训练 ==========
    print(f"\n========== 用最优超参训练最终模型 ({args.final_epochs} epochs) ==========")
    model, std, (train_true, train_pred), (val_true, val_pred) = train_final(
        best_params, data, args.node_vec_len, args.max_atoms, device,
        args.final_epochs, args.output_dir, plot=True)

    # ========== 4. 外部数据集评估 ==========
    print(f"\n========== 外部数据集评估 ==========")
    external_datasets = {
        'collet_homa_0702': os.path.join(_PROJ_ROOT, "collet_homa_0702.csv"),
        'lunci2_mbcout': os.path.join(_PROJ_ROOT, "outcsv/lunci2-mbcout.csv"),
    }

    ext_results = {}
    for name, path in external_datasets.items():
        if not os.path.exists(path):
            print(f"  [{name}] 文件不存在: {path}")
            continue
        print(f"\n  [{name}] 加载: {path}")
        ext_data = process_and_save_data(path, args.node_vec_len,
                                          args.max_atoms, 'homa_value')
        print(f"  [{name}] 数据量: {len(ext_data['node_mats'])}")

        # 方式A: 用原始数据训练的模型直接预测（零样本迁移）
        t0 = time.time()
        loss_a, mae_a, r2_a, rmse_a, true_a, pred_a = predict_external(
            model, ext_data, args.node_vec_len, args.max_atoms, device)
        print(f"  [{name}] 零样本迁移: R²={r2_a:.4f}, MAE={mae_a:.4f}, "
              f"RMSE={rmse_a:.4f}  ({time.time()-t0:.1f}s)")

        # 方式B: 在该数据集上独立训练（用最优超参）
        t0 = time.time()
        ext_dir = os.path.join(args.output_dir, f'ext_{name}')
        model_b, _, _, (val_t, val_p) = train_final(
            best_params, ext_data, args.node_vec_len, args.max_atoms, device,
            args.final_epochs, ext_dir, plot=True)
        loss_b, mae_b, r2_b, rmse_b, _, _ = predict_external(
            model_b, ext_data, args.node_vec_len, args.max_atoms, device)
        print(f"  [{name}] 独立训练: R²={r2_b:.4f}, MAE={mae_b:.4f}, "
              f"RMSE={rmse_b:.4f}  ({time.time()-t0:.1f}s)")

        # 零样本迁移散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(true_a.numpy(), pred_a.numpy(), alpha=0.4, s=8, c='purple')
        lims = [min(true_a.min(), pred_a.min()),
                max(true_a.max(), pred_a.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)')
        plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'{name} (Zero-shot Transfer)\n'
                  f'R²={r2_a:.4f}, MAE={mae_a:.4f}, RMSE={rmse_a:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{name}_zeroshot_parity.png'),
                    bbox_inches='tight')
        plt.close()

        ext_results[name] = {
            'zeroshot': (r2_a, mae_a, rmse_a),
            'independent': (r2_b, mae_b, rmse_b),
        }
        del model_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 保存外部评估汇总
    with open(os.path.join(args.output_dir, 'external_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'mode', 'R2', 'MAE', 'RMSE'])
        for name, res in ext_results.items():
            r2, mae, rmse = res['zeroshot']
            w.writerow([name, 'zeroshot', f'{r2:.6f}', f'{mae:.6f}', f'{rmse:.6f}'])
            r2, mae, rmse = res['independent']
            w.writerow([name, 'independent', f'{r2:.6f}', f'{mae:.6f}', f'{rmse:.6f}'])

    print(f"\n{'='*60}")
    print("全部完成! 结果保存到:", args.output_dir)


if __name__ == '__main__':
    main()
