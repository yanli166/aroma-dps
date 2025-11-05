import os
import numpy as np
import torch
import torch.cuda.amp as amp  # 混合精度训练
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error  
import matplotlib.pyplot as plt  # 添加缺失的绘图库导入

class Standardizer:
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

def create_scheduler(optimizer, **kwargs):
    return ReduceLROnPlateau(optimizer, mode='min', **kwargs)

def train_model(
    epoch,
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,  
    max_atoms,
    node_vec_len,
    scheduler=None,
    max_grad_norm=1.0
):
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_preds = [], []
    
    for i, ((node_mat, adj_mat, mask_mat), outputs, _) in enumerate(dataloader):
        # 数据整形与GPU迁移
        batch_size = len(outputs)
        inputs = (
            node_mat.to(device, non_blocking=True),
            adj_mat.to(device, non_blocking=True),
            mask_mat.to(device, non_blocking=True)
        )
        targets = outputs.to(device, non_blocking=True)

        # 混合精度训练
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(*inputs).squeeze(-1)
            loss = loss_fn(preds, targets)
        
        # 反向传播与梯度裁剪
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

    # 计算整体指标
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    r2 = r2_score(all_outputs, all_preds)
    
    # 更新学习率
    if scheduler:
        scheduler.step(avg_loss)
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f}")
    return avg_loss, avg_mae, r2, all_outputs, all_preds

def test_model(
    model,
    dataloader,
    loss_fn,
    device,  
    max_atoms,
    node_vec_len
):
    model.eval()
    model.to(device)
    
    total_loss, total_mae = 0.0, 0.0
    all_outputs, all_preds = [], []
    
    with torch.no_grad():
        for (node_mat, adj_mat, mask_mat), outputs, _ in dataloader:
            batch_size = len(outputs)
            inputs = (
                node_mat.to(device, non_blocking=True),
                adj_mat.to(device, non_blocking=True),
                mask_mat.to(device, non_blocking=True)
            )
            targets = outputs.to(device)
            
            preds = model(*inputs).squeeze(-1)
            loss = loss_fn(preds, targets)
            mae = torch.abs(preds - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            all_outputs.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # 计算最终指标
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    r2 = r2_score(all_outputs, all_preds)
    rmse = np.sqrt(mean_squared_error(all_outputs, all_preds))
    
    print(f"Test Results: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
    return avg_loss, avg_mae, r2, rmse, all_outputs, all_preds

# 可视化工具函数（保留文档1实现）
def loss_curve(save_dir, epochs, train_losses, test_losses):
    """
    绘制并保存训练和验证损失曲线
    
    参数:
        save_dir: 保存图片的目录路径
        epochs: 包含所有epoch编号的列表
        train_losses: 训练损失值列表
        test_losses: 验证损失值列表
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建高质量图表
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', linewidth=2, label='Validation Loss')
    
    # 标记最佳验证点
    best_epoch = np.argmin(test_losses)
    plt.scatter(epochs[best_epoch], test_losses[best_epoch], 
                s=100, c='gold', edgecolor='k', 
                label=f'Best Val Loss: {test_losses[best_epoch]:.4f}')
    
    # 设置图表样式
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 同时保存损失数据为CSV
    data = np.column_stack((epochs, train_losses, test_losses))
    csv_path = os.path.join(save_dir, "loss_data.csv")
    np.savetxt(csv_path, data, delimiter=",", 
               header="epoch,train_loss,val_loss", comments='', fmt='%.6f')
    
    print(f"Loss curve saved to {plot_path}")
    print(f"Loss data saved to {csv_path}")

def save_parity_plot(save_dir, true_values, predicted_values, title):
    """
    绘制并保存真实值与预测值的对比图
    
    参数:
        save_dir: 保存图片的目录路径
        true_values: 真实值数组
        predicted_values: 预测值数组
        title: 图表的标题
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算评估指标
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    
    # 创建图表
    plt.figure(figsize=(8, 6), dpi=300)
    
    # 使用hexbin处理大数据集
    if len(true_values) > 5000:
        plt.hexbin(true_values, predicted_values, 
                   gridsize=100, cmap='viridis', bins='log')
    else:
        plt.scatter(true_values, predicted_values, 
                    alpha=0.6, s=20, edgecolor='w', linewidth=0.3)
    
    # 添加完美拟合线
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
    
    # 保存数据到CSV
    data = np.column_stack((true_values, predicted_values))
    csv_path = os.path.join(save_dir, f"{filename}_data.csv")
    np.savetxt(csv_path, data, delimiter=",", 
               header="true_values,predicted_values", comments='', fmt='%.6f')
    
    print(f"Parity plot saved to {plot_path}")
    print(f"Data saved to {csv_path}")