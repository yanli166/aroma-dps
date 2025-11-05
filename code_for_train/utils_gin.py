"""Utility functions for GIN model training and evaluation"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class Standardizer:
    """
    数据标准化器，用于输入数据的标准化和还原
    """
    def __init__(self, X: torch.Tensor):
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0) + 1e-8
    
    def standardize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std
    
    def restore(self, Z: torch.Tensor) -> torch.Tensor:
        return self.mean + Z * self.std
    
    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}
    
    def load(self, state: dict):
        self.mean = state["mean"]
        self.std = state["std"]

def train_model(
    epoch: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: torch.device,
    max_atoms: int,
    node_vec_len: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    max_grad_norm: float = 1.0
) -> tuple:
    """
    训练一个epoch的GIN模型
    
    改进点：
    1. 添加梯度裁剪防止梯度爆炸
    2. 使用set_to_none优化内存
    3. 支持混合精度训练
    """
    model.train()
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_preds = [], []
    
    for (node_mat, adj_mat, mask_mat), outputs, _ in train_loader:
        # 数据整形与设备转移
        batch_size = len(outputs)
        inputs = (
            node_mat.view(batch_size, max_atoms, node_vec_len).to(device, non_blocking=True),
            adj_mat.view(batch_size, max_atoms, max_atoms).to(device, non_blocking=True),
            mask_mat.view(batch_size, max_atoms, node_vec_len).to(device, non_blocking=True)
        )
        targets = outputs.to(device, non_blocking=True)

        # 混合精度前向传播
        with amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(*inputs).squeeze()
            loss = loss_fn(preds, targets)
        
        # 反向传播与梯度裁剪
        scaler.scale(loss).backward()
        if max_grad_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # 内存优化
        
        # 指标计算
        with torch.no_grad():
            mae = torch.abs(preds - targets).mean()
            total_loss += loss.item()
            total_mae += mae.item()
            all_outputs.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # 计算epoch指标
    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)
    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    r2 = r2_score(all_outputs, all_preds)
    
    # 学习率调度
    if scheduler:
        scheduler.step(avg_loss)
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f}")
    return avg_loss, avg_mae, r2, all_outputs, all_preds

def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device,
    max_atoms: int,
    node_vec_len: int
) -> tuple:
    """
    评估GIN模型性能
    
    改进点：
    1. 统一输入数据预处理逻辑
    2. 添加RMSE指标计算
    """
    model.eval()
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_preds = [], []
    
    with torch.no_grad():
        for (node_mat, adj_mat, mask_mat), outputs, _ in test_loader:
            batch_size = len(outputs)
            inputs = (
                node_mat.view(batch_size, max_atoms, node_vec_len).to(device),
                adj_mat.view(batch_size, max_atoms, max_atoms).to(device),
                mask_mat.view(batch_size, max_atoms, node_vec_len).to(device)
            )
            targets = outputs.to(device)
            
            preds = model(*inputs).squeeze()
            loss = loss_fn(preds, targets)
            mae = torch.abs(preds - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            all_outputs.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # 计算最终指标
    avg_loss = total_loss / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    r2 = r2_score(all_outputs, all_preds)
    rmse = np.sqrt(mean_squared_error(all_outputs, all_preds))
    
    print(f"Test Results: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
    return avg_loss, avg_mae, r2, rmse, all_outputs, all_preds

def save_parity_plot(
    save_dir: str, 
    true_values: np.ndarray, 
    predicted_values: np.ndarray, 
    title: str
):
    """
    生成预测值-真实值对比图
    
    改进点：
    1. 自动处理大数据集(>5000样本)
    2. 同时保存原始数据
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算指标
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    
    # 创建图表
    plt.figure(figsize=(8, 6), dpi=300)
    
    # 大数据集使用hexbin
    if len(true_values) > 5000:
        plt.hexbin(true_values, predicted_values, 
                   gridsize=100, cmap='viridis', bins='log')
    else:
        plt.scatter(true_values, predicted_values, 
                    alpha=0.6, s=20, edgecolor='w', linewidth=0.3)
    
    # 添加参考线
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    # 设置样式
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f"{title}\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存结果
    filename = title.lower().replace(' ', '_')
    plot_path = os.path.join(save_dir, f"{filename}_parity.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存原始数据
    data = np.column_stack((true_values, predicted_values))
    csv_path = os.path.join(save_dir, f"{filename}_data.csv")
    np.savetxt(csv_path, data, delimiter=",", 
               header="true,predicted", comments='', fmt='%.6f')
    
    print(f"Saved parity plot to {plot_path}")
    print(f"Saved data to {csv_path}")

def plot_loss_curve(
    save_dir: str, 
    epochs: list, 
    train_losses: list, 
    val_losses: list
):
    """
    绘制训练/验证损失曲线
    
    改进点：
    1. 自动标记最佳验证点
    2. 同时保存损失数据
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    # 标记最佳epoch
    best_epoch = np.argmin(val_losses)
    plt.scatter(epochs[best_epoch], val_losses[best_epoch],
                s=100, c='gold', edgecolor='k',
                label=f'Best Epoch: {best_epoch+1}')
    
    # 设置样式
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # 保存结果
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存损失数据
    data = np.column_stack((epochs, train_losses, val_losses))
    csv_path = os.path.join(save_dir, "loss_history.csv")
    np.savetxt(csv_path, data, delimiter=",",
               header="epoch,train_loss,val_loss", comments='', fmt='%.6f')
    
    print(f"Saved loss curve to {plot_path}")
    print(f"Saved loss data to {csv_path}")