"""Utility functions for GNN training and evaluation."""
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
    def __init__(self, X):
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0) + 1e-8
    
    def standardize(self, X):
        """标准化数据"""
        return (X - self.mean) / self.std
    
    def restore(self, Z):
        """还原标准化数据"""
        return self.mean + Z * self.std
    
    def state(self):
        """获取当前状态"""
        return {"mean": self.mean, "std": self.std}
    
    def load(self, state):
        """加载保存的状态"""
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
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
):
    """
    训练一个epoch的GNN模型
    
    参数:
        epoch: 当前epoch编号
        model: GNN模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 计算设备
        max_atoms: 最大原子数
        node_vec_len: 节点向量长度
        scheduler: 学习率调度器
        
    返回:
        avg_loss: 平均训练损失
        avg_mae: 平均MAE
        r2: R²分数
        all_outputs: 所有真实值
        all_predictions: 所有预测值
    """
    model.train()
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_predictions = [], []
    
    for i, ((node_mat, adj_mat, mask_mat), outputs, _) in enumerate(train_loader):
        # 准备输入数据
        batch_size = len(outputs)
        inputs = (
            node_mat.view(batch_size, max_atoms, node_vec_len).to(device),
            adj_mat.view(batch_size, max_atoms, max_atoms).to(device),
            mask_mat.view(batch_size, max_atoms, node_vec_len).to(device)
        )
        targets = outputs.to(device)

        # 混合精度训练
        with amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            predictions = model(*inputs).squeeze()
            loss = loss_fn(predictions, targets)
        
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 计算指标
        with torch.no_grad():
            mae = torch.abs(predictions - targets).mean()
            total_loss += loss.item()
            total_mae += mae.item()
            all_outputs.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)
    all_outputs = np.concatenate(all_outputs)
    all_predictions = np.concatenate(all_predictions)
    r2 = r2_score(all_outputs, all_predictions)
    
    # 更新学习率
    if scheduler:
        scheduler.step(avg_loss)
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f}")
    return avg_loss, avg_mae, r2, all_outputs, all_predictions

def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device,
    max_atoms: int,
    node_vec_len: int
):
    """
    测试GNN模型
    
    参数:
        model: GNN模型
        test_loader: 测试数据加载器
        loss_fn: 损失函数
        device: 计算设备
        max_atoms: 最大原子数
        node_vec_len: 节点向量长度
        
    返回:
        test_loss: 测试损失
        test_mae: 测试MAE
        test_r2: 测试R²分数
        test_rmse: 测试RMSE
        all_outputs: 所有真实值
        all_predictions: 所有预测值
    """
    model.eval()
    model.to(device)
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_predictions = [], []
    
    with torch.no_grad():
        for (node_mat, adj_mat, mask_mat), outputs, _ in test_loader:
            # 准备输入数据
            batch_size = len(outputs)
            inputs = (
                node_mat.view(batch_size, max_atoms, node_vec_len).to(device),
                adj_mat.view(batch_size, max_atoms, max_atoms).to(device),
                mask_mat.view(batch_size, max_atoms, node_vec_len).to(device)
            )
            targets = outputs.to(device)
            
            # 预测
            predictions = model(*inputs).squeeze()
            loss = loss_fn(predictions, targets)
            mae = torch.abs(predictions - targets).mean()
            
            # 累积指标
            total_loss += loss.item()
            total_mae += mae.item()
            all_outputs.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # 计算最终指标
    test_loss = total_loss / len(test_loader)
    test_mae = total_mae / len(test_loader)
    all_outputs = np.concatenate(all_outputs)
    all_predictions = np.concatenate(all_predictions)
    test_r2 = r2_score(all_outputs, all_predictions)
    test_rmse = np.sqrt(mean_squared_error(all_outputs, all_predictions))
    
    print(f"Test Results: Loss={test_loss:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
    return test_loss, test_mae, test_r2, test_rmse, all_outputs, all_predictions

def save_parity_plot(save_dir: str, true_values: np.ndarray, predicted_values: np.ndarray, title: str):
    """
    保存真实值与预测值的对比图
    
    参数:
        save_dir: 保存目录
        true_values: 真实值数组
        predicted_values: 预测值数组
        title: 图表标题
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算评估指标
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    
    # 创建图表
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(true_values, predicted_values, alpha=0.6, s=20, edgecolor='w', linewidth=0.3)
    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    # 设置图表样式
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f"{title}\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存图表
    filename = title.lower().replace(' ', '_')
    plot_path = os.path.join(save_dir, f"{filename}_parity_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存数据
    data = np.column_stack((true_values, predicted_values))
    csv_path = os.path.join(save_dir, f"{filename}_data.csv")
    np.savetxt(csv_path, data, delimiter=",", 
               header="true_values,predicted_values", comments='', fmt='%.6f')
    
    print(f"Saved parity plot to {plot_path}")
    print(f"Saved data to {csv_path}")

def loss_curve(save_dir: str, epochs: list, train_losses: list, val_losses: list):
    """
    绘制并保存训练和验证损失曲线
    
    参数:
        save_dir: 保存目录
        epochs: epoch列表
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    best_epoch = np.argmin(val_losses)
    plt.scatter(epochs[best_epoch], val_losses[best_epoch], 
                s=100, c='gold', edgecolor='k', 
                label=f'Best Val Loss: {val_losses[best_epoch]:.4f}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    data = np.column_stack((epochs, train_losses, val_losses))
    csv_path = os.path.join(save_dir, "loss_data.csv")
    np.savetxt(csv_path, data, delimiter=",", 
               header="epoch,train_loss,val_loss", comments='', fmt='%.6f')
    
    print(f"Saved loss curve to {plot_path}")
    print(f"Saved loss data to {csv_path}")