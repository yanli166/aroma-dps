"""
统一的训练和评估工具函数
"""
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class Standardizer:
    """输出标准化器"""
    def __init__(self, X):
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0) + 1e-8

    def standardize(self, X):
        return (X - self.mean) / self.std

    def restore(self, Z):
        return self.mean + Z * self.std

    def state(self):
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


def get_model_inputs(batch, mode, device):
    """根据模式返回模型输入关键字参数

    mode:
        - 'label': node_mat, adj_mat - 环标记已在节点特征中
        - 'mask':  node_mat, adj_mat, mask_mat - 使用mask过滤
        - 'pool':  node_mat, adj_mat, ring_indices - 使用环索引池化
    """
    (node_mat, adj_mat, mask_mat, ring_indices), outputs, _ = batch
    node_mat = node_mat.to(device, non_blocking=True)
    adj_mat = adj_mat.to(device, non_blocking=True)
    mask_mat = mask_mat.to(device, non_blocking=True)
    ring_indices = ring_indices.to(device, non_blocking=True)
    outputs = outputs.to(device, non_blocking=True)

    if mode == 'label':
        return {'node_mat': node_mat, 'adj_mat': adj_mat}, outputs
    elif mode == 'mask':
        return {'node_mat': node_mat, 'adj_mat': adj_mat, 'mask_mat': mask_mat}, outputs
    elif mode == 'pool':
        return {'node_mat': node_mat, 'adj_mat': adj_mat, 'ring_indices': ring_indices}, outputs
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, mode,
                   max_atoms, node_vec_len, scheduler=None, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    all_preds, all_outputs = [], []
    total_loss, total_mae = 0.0, 0.0
    n_batches = 0

    for batch in dataloader:
        inputs, targets = get_model_inputs(batch, mode, device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(**inputs).squeeze()
        loss = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_mae += torch.abs(preds - targets).mean().item()
        n_batches += 1
        all_preds.append(preds.detach().cpu())
        all_outputs.append(targets.detach().cpu())

    if scheduler is not None:
        scheduler.step(total_loss / n_batches)

    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    all_preds = torch.cat(all_preds)
    all_outputs = torch.cat(all_outputs)
    r2 = r2_score(all_outputs.numpy(), all_preds.numpy())
    return avg_loss, avg_mae, r2, all_outputs, all_preds


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, mode, max_atoms, node_vec_len):
    """评估模型"""
    model.eval()
    all_preds, all_outputs = [], []
    total_loss, total_mae = 0.0, 0.0
    n_batches = 0

    for batch in dataloader:
        inputs, targets = get_model_inputs(batch, mode, device)
        preds = model(**inputs).squeeze()
        loss = loss_fn(preds, targets)

        total_loss += loss.item()
        total_mae += torch.abs(preds - targets).mean().item()
        n_batches += 1
        all_preds.append(preds.detach().cpu())
        all_outputs.append(targets.detach().cpu())

    avg_loss = total_loss / max(n_batches, 1)
    avg_mae = total_mae / max(n_batches, 1)
    all_preds = torch.cat(all_preds)
    all_outputs = torch.cat(all_outputs)
    r2 = r2_score(all_outputs.numpy(), all_preds.numpy())
    rmse = torch.sqrt(((all_preds - all_outputs) ** 2).mean()).item()
    return avg_loss, avg_mae, r2, rmse, all_outputs, all_preds


def plot_loss_curve(save_dir, epochs, train_losses, val_losses):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    if val_losses:
        best_idx = int(np.argmin(val_losses))
        plt.scatter(epochs[best_idx], val_losses[best_idx],
                   color='gold', s=100, zorder=5,
                   label=f'Best Val (Epoch {epochs[best_idx]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close()

    # 保存数据
    with open(os.path.join(save_dir, 'loss_data.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for e, tl, vl in zip(epochs, train_losses, val_losses):
            writer.writerow([e, tl, vl])


def save_parity_plot(save_dir, true_values, predicted_values, title='Parity Plot'):
    """绘制真实值 vs 预测值对比图"""
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    r2 = r2_score(true_values, predicted_values)
    rmse = float(np.sqrt(((true_values - predicted_values) ** 2).mean()))
    mae = float(np.abs(true_values - predicted_values).mean())

    plt.figure(figsize=(8, 8), dpi=300)
    if len(true_values) > 5000:
        plt.hexbin(true_values, predicted_values, gridsize=50, cmap='Blues', mincnt=1)
    else:
        plt.scatter(true_values, predicted_values, alpha=0.5, s=10, c='blue')
    lims = [min(true_values.min(), predicted_values.min()),
            max(true_values.max(), predicted_values.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.png'), bbox_inches='tight')
    plt.close()
