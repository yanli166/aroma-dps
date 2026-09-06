"""
GNN-label 高效GPU优化版 - 全数据常驻GPU，消除传输开销
适用于 GPU 环境，速度比原版快 5-10x
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import Trial
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

from unified_models.common.graphs import process_and_save_data
from unified_models.gnn.model import GNNModel


def load_data_to_gpu(dataset_path, node_vec_len, max_atoms, target_col, device, seed=42, train_size=0.8):
    """加载图数据并全部放到 GPU 上，返回 train/val 的预切分 batch 索引"""
    data = process_and_save_data(dataset_path, node_vec_len, max_atoms, target_col)
    n = len(data['node_mats'])

    # 一次性将所有数据转为 GPU 张量
    node_mats = torch.tensor(np.array(data['node_mats']), dtype=torch.float32, device=device)
    adj_mats = torch.tensor(np.array(data['adj_mats']), dtype=torch.float32, device=device)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32, device=device).squeeze()

    # 随机划分
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(np.floor(train_size * n))
    train_idx = torch.tensor(idx[:split], device=device)
    val_idx = torch.tensor(idx[split:], device=device)

    return node_mats, adj_mats, outputs, train_idx, val_idx, n, data


def train_eval_fast(params, node_mats, adj_mats, outputs, train_idx, val_idx,
                    node_vec_len, device, n_epochs, patience=25, seed=42):
    """全GPU训练 - 数据不离开GPU"""
    torch.manual_seed(seed)
    hidden_dim = params['hidden_dim']
    batch_size = params['batch_size']

    model = GNNModel(
        node_vec_len=node_vec_len,
        hidden_dim=hidden_dim,
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

    n_train = len(train_idx)
    n_val = len(val_idx)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        # === 训练 ===
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            node_b = node_mats[batch_idx]
            adj_b = adj_mats[batch_idx]
            out_b = outputs[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(node_b, adj_b).squeeze()
            loss = loss_fn(pred, out_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += torch.abs(pred - out_b).mean().item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # === 验证 ===
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for i in range(0, n_val, batch_size):
                batch_idx = val_idx[i:i + batch_size]
                node_b = node_mats[batch_idx]
                adj_b = adj_mats[batch_idx]
                out_b = outputs[batch_idx]
                pred = model(node_b, adj_b).squeeze()
                val_preds.append(pred)
                val_targets.append(out_b)

            val_pred = torch.cat(val_preds)
            val_true = torch.cat(val_targets)
            val_loss = loss_fn(val_pred, val_true).item()
            val_mae = torch.abs(val_pred - val_true).mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 用最佳状态评估
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for i in range(0, n_val, batch_size):
            batch_idx = val_idx[i:i + batch_size]
            pred = model(node_mats[batch_idx], adj_mats[batch_idx]).squeeze()
            val_preds.append(pred)
            val_targets.append(outputs[batch_idx])
        val_pred = torch.cat(val_preds)
        val_true = torch.cat(val_targets)

    val_loss = loss_fn(val_pred, val_true).item()
    val_mae = torch.abs(val_pred - val_true).mean().item()
    val_r2 = r2_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
    val_rmse = torch.sqrt(((val_pred - val_true) ** 2).mean()).item()

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_val_loss, val_mae, val_r2, val_rmse, val_true, val_pred


def objective(trial, node_mats, adj_mats, outputs, train_idx, val_idx,
              node_vec_len, device, n_epochs):
    params = {
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 6),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 4),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
        'p_dropout': trial.suggest_float('p_dropout', 0.1, 0.4),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    t0 = time.time()
    val_loss, val_mae, val_r2, val_rmse, _, _ = train_eval_fast(
        params, node_mats, adj_mats, outputs, train_idx, val_idx,
        node_vec_len, device, n_epochs)
    elapsed = time.time() - t0
    print(f"  Trial {trial.number}: val_loss={val_loss:.4f}, R²={val_r2:.4f}, "
          f"MAE={val_mae:.4f}, RMSE={val_rmse:.4f}  ({elapsed:.0f}s)")
    return val_loss


def train_final_fast(params, node_mats, adj_mats, outputs, train_idx, val_idx,
                     node_vec_len, device, n_epochs, output_dir, seed=42,
                     patience=30, plot=True):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    batch_size = params['batch_size']
    hidden_dim = params['hidden_dim']

    model = GNNModel(
        node_vec_len=node_vec_len,
        hidden_dim=hidden_dim,
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

    n_train = len(train_idx)
    n_val = len(val_idx)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epochs_list, train_losses, val_losses = [], [], []
    epoch_records = []

    for epoch in range(1, n_epochs + 1):
        # 训练
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(node_mats[batch_idx], adj_mats[batch_idx]).squeeze()
            loss = loss_fn(pred, outputs[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mae += torch.abs(pred - outputs[batch_idx]).mean().item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        avg_train_mae = epoch_mae / n_batches

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
            val_mae = torch.abs(val_pred - val_true).mean().item()
            val_r2 = r2_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
            val_rmse = torch.sqrt(((val_pred - val_true) ** 2).mean()).item()

        epochs_list.append(epoch)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        epoch_records.append({
            'epoch': epoch, 'train_loss': avg_train_loss, 'train_mae': avg_train_mae,
            'train_r2': 0, 'val_loss': val_loss, 'val_mae': val_mae,
            'val_r2': val_r2, 'val_rmse': val_rmse
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{n_epochs}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_R²={val_r2:.4f}, val_MAE={val_mae:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # 最终评估
    model.eval()
    with torch.no_grad():
        # Train
        tp, tt = [], []
        for i in range(0, n_train, batch_size):
            bi = train_idx[i:i + batch_size]
            tp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
            tt.append(outputs[bi])
        train_pred = torch.cat(tp).cpu().numpy()
        train_true = torch.cat(tt).cpu().numpy()
        # Val
        vp, vt = [], []
        for i in range(0, n_val, batch_size):
            bi = val_idx[i:i + batch_size]
            vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
            vt.append(outputs[bi])
        val_pred = torch.cat(vp).cpu().numpy()
        val_true = torch.cat(vt).cpu().numpy()

    train_r2 = r2_score(train_true, train_pred)
    train_mae = float(np.abs(train_true - train_pred).mean())
    train_rmse = float(np.sqrt(((train_true - train_pred) ** 2).mean()))
    val_r2 = r2_score(val_true, val_pred)
    val_mae = float(np.abs(val_true - val_pred).mean())
    val_rmse = float(np.sqrt(((val_true - val_pred) ** 2).mean()))

    print(f"\n  最终结果:")
    print(f"  Train: R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
    print(f"  Val:   R²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}")

    # 保存所有结果
    with open(os.path.join(output_dir, 'config.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['param', 'value'])
        for k, v in params.items(): w.writerow([k, v])
        w.writerow(['n_epochs', n_epochs]); w.writerow(['best_epoch', best_epoch])
        w.writerow(['best_val_loss', best_val_loss])

    with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['best_val_loss', best_val_loss]); w.writerow(['best_epoch', best_epoch])
        w.writerow(['final_train_r2', train_r2]); w.writerow(['final_train_mae', train_mae])
        w.writerow(['final_train_rmse', train_rmse]); w.writerow(['final_val_r2', val_r2])
        w.writerow(['final_val_mae', val_mae]); w.writerow(['final_val_rmse', val_rmse])

    with open(os.path.join(output_dir, 'epoch_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_mae', 'train_r2',
                                                'val_loss', 'val_mae', 'val_r2', 'val_rmse'])
        writer.writeheader(); writer.writerows(epoch_records)

    pd.DataFrame({'true': train_true, 'pred': train_pred}).to_csv(
        os.path.join(output_dir, 'train_set_data.csv'), index=False)
    pd.DataFrame({'true': val_true, 'pred': val_pred}).to_csv(
        os.path.join(output_dir, 'val_set_data.csv'), index=False)

    if plot:
        # 损失曲线
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=1.5)
        plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=1.5)
        plt.scatter([best_epoch], [best_val_loss], color='gold', s=100, zorder=5,
                    label=f'Best Val (Epoch {best_epoch})')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('GNN-label (Optimized) Loss Curve'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), bbox_inches='tight'); plt.close()

        # 验证集散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(val_true, val_pred, alpha=0.4, s=8, c='blue')
        lims = [min(val_true.min(), val_pred.min()), max(val_true.max(), val_pred.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)'); plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'GNN-label (Optimized) Validation Parity Plot\n'
                  f'R²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_parity_plot.png'), bbox_inches='tight'); plt.close()

        # 训练集散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(train_true, train_pred, alpha=0.2, s=5, c='green')
        lims = [min(train_true.min(), train_pred.min()), max(train_true.max(), train_pred.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)'); plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'GNN-label (Optimized) Train Parity Plot\n'
                  f'R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_parity_plot.png'), bbox_inches='tight'); plt.close()

    return model


@torch.no_grad()
def predict_external_fast(model, node_mats, adj_mats, outputs, batch_size, device):
    """在外部数据集上做预测"""
    model.eval()
    n = len(node_mats)
    preds, trues = [], []
    for i in range(0, n, batch_size):
        bi = slice(i, min(i + batch_size, n))
        pred = model(node_mats[bi], adj_mats[bi]).squeeze()
        preds.append(pred)
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    return r2, mae, rmse, true, pred


def main():
    parser = argparse.ArgumentParser(description='GNN-label Optuna优化 (GPU高速版)')
    parser.add_argument('--dataset_path', type=str,
                        default='_PROJ_ROOT/nics-nics1zz-out-no3.csv')
    parser.add_argument('--n_trials', type=int, default=25)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--final_epochs', type=int, default=300)
    parser.add_argument('--node_vec_len', type=int, default=60)
    parser.add_argument('--max_atoms', type=int, default=75)
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT/gnn_label_optimized')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_num_threads(96)
    else:
        torch.set_num_threads(96)
    print(f"CPU threads: {torch.get_num_threads()}")

    # 1. 加载数据到GPU
    print(f"\n========== 加载数据: {args.dataset_path} ==========")
    t0 = time.time()
    node_mats, adj_mats, outputs, train_idx, val_idx, n, data = load_data_to_gpu(
        args.dataset_path, args.node_vec_len, args.max_atoms, 'homa_value', device)
    print(f"数据集大小: {n} (train={len(train_idx)}, val={len(val_idx)})  ({time.time()-t0:.1f}s)")
    print(f"GPU显存: node_mats={node_mats.element_size()*node_mats.nelement()/1e6:.1f}MB, "
          f"adj_mats={adj_mats.element_size()*adj_mats.nelement()/1e6:.1f}MB")

    # 2. Optuna 优化
    print(f"\n========== Optuna 优化 ({args.n_trials} trials, {args.n_epochs} epochs/trial) ==========")
    study = optuna.create_study(
        direction='minimize', study_name='gnn_label_opt_gpu',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, node_mats, adj_mats, outputs, train_idx, val_idx,
                                args.node_vec_len, device, args.n_epochs),
        n_trials=args.n_trials, show_progress_bar=False
    )

    print(f"\n{'='*60}")
    print(f"Optuna 完成! 试验数: {len(study.trials)}")
    print(f"最佳 val_loss: {study.best_trial.value:.6f}")
    print("最佳参数:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, 'all_trials.csv'), index=False)
    best_params = study.best_trial.params.copy()
    pd.DataFrame([best_params]).to_csv(
        os.path.join(args.output_dir, 'best_parameters.csv'), index=False)

    # 3. 最终训练
    print(f"\n========== 用最优超参训练最终模型 ({args.final_epochs} epochs) ==========")
    model = train_final_fast(
        best_params, node_mats, adj_mats, outputs, train_idx, val_idx,
        args.node_vec_len, device, args.final_epochs, args.output_dir, plot=True)

    # 4. 外部数据集评估
    print(f"\n========== 外部数据集评估 ==========")
    external_datasets = {
        'collet_homa_0702': '_PROJ_ROOT/collet_homa_0702.csv',
        'lunci2_mbcout': '_PROJ_ROOT/outcsv/lunci2-mbcout.csv',
    }

    ext_results = {}
    for name, path in external_datasets.items():
        if not os.path.exists(path):
            print(f"  [{name}] 文件不存在: {path}")
            continue
        print(f"\n  [{name}] 加载: {path}")
        t0 = time.time()
        ext_node, ext_adj, ext_out, _, _, ext_n, _ = load_data_to_gpu(
            path, args.node_vec_len, args.max_atoms, 'homa_value', device, seed=99)
        print(f"  [{name}] 数据量: {ext_n}  ({time.time()-t0:.1f}s)")

        # 方式A: 零样本迁移
        t0 = time.time()
        r2_a, mae_a, rmse_a, true_a, pred_a = predict_external_fast(
            model, ext_node, ext_adj, ext_out, 256, device)
        print(f"  [{name}] 零样本迁移: R²={r2_a:.4f}, MAE={mae_a:.4f}, "
              f"RMSE={rmse_a:.4f}  ({time.time()-t0:.1f}s)")

        # 方式B: 独立训练
        t0 = time.time()
        ext_dir = os.path.join(args.output_dir, f'ext_{name}')
        # 独立训练时需要重新划分 train/val
        rng = np.random.RandomState(42)
        idx = np.arange(ext_n)
        rng.shuffle(idx)
        split = int(np.floor(0.8 * ext_n))
        ext_train_idx = torch.tensor(idx[:split], device=device)
        ext_val_idx = torch.tensor(idx[split:], device=device)

        model_b = train_final_fast(
            best_params, ext_node, ext_adj, ext_out, ext_train_idx, ext_val_idx,
            args.node_vec_len, device, args.final_epochs, ext_dir, plot=True)
        r2_b, mae_b, rmse_b, _, _ = predict_external_fast(
            model_b, ext_node, ext_adj, ext_out, 256, device)
        print(f"  [{name}] 独立训练: R²={r2_b:.4f}, MAE={mae_b:.4f}, "
              f"RMSE={rmse_b:.4f}  ({time.time()-t0:.1f}s)")

        # 零样本散点图
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(true_a, pred_a, alpha=0.4, s=8, c='purple')
        lims = [min(true_a.min(), pred_a.min()), max(true_a.max(), pred_a.max())]
        plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
        plt.xlabel('True Values (HOMA)'); plt.ylabel('Predicted Values (HOMA)')
        plt.title(f'{name} (Zero-shot Transfer)\n'
                  f'R²={r2_a:.4f}, MAE={mae_a:.4f}, RMSE={rmse_a:.4f}')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{name}_zeroshot_parity.png'),
                    bbox_inches='tight'); plt.close()

        ext_results[name] = {'zeroshot': (r2_a, mae_a, rmse_a),
                             'independent': (r2_b, mae_b, rmse_b)}
        del model_b, ext_node, ext_adj, ext_out
        torch.cuda.empty_cache()

    # 保存外部汇总
    with open(os.path.join(args.output_dir, 'external_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'mode', 'R2', 'MAE', 'RMSE'])
        for name, res in ext_results.items():
            for mode in ['zeroshot', 'independent']:
                r2, mae, rmse = res[mode]
                w.writerow([name, mode, f'{r2:.6f}', f'{mae:.6f}', f'{rmse:.6f}'])

    print(f"\n{'='*60}")
    print("全部完成! 结果保存到:", args.output_dir)


if __name__ == '__main__':
    main()
