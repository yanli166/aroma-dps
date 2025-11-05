"""Advanced training script for Graph Isomorphism Network (GIN)."""
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gin.model_gin import GIN
from gin.graphs_gin import GraphData, collate_graph_dataset, process_and_save_data
from gin.utils_gin import StandardScaler, plot_loss_curve, save_parity_plot

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_data_loaders(dataset, indices, batch_size):
    """创建标准化的数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        collate_fn=collate_graph_dataset,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True
    )

def main():
    parser = argparse.ArgumentParser(description='Train GNN model for molecular property prediction')
    # 参数定义保持不变...
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    # 设置CPU线程
    torch.set_num_threads(4)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 准备数据集
    print(f"Processing dataset from: {args.dataset_path}")
    start_time = time.time()
    processed_data = process_and_save_data(
        dataset_path=args.dataset_path,
        node_vec_len=args.node_vec_len,
        max_atoms=args.max_atoms
    )
    dataset = GraphData(processed_data, args.node_vec_len, args.max_atoms)
    print(f"Dataset size: {len(dataset)} | Processing time: {time.time()-start_time:.2f}s")
    
    # 设备设置
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # 输出标准化
    all_outputs = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    output_scaler = StandardScaler(all_outputs)
    
    def train_model_fold(model, train_loader, val_loader, test_loader=None, fold=None):
        # 创建输出目录
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}") if fold else args.output_dir
        os.makedirs(fold_dir, exist_ok=True)
        
        # 优化器和调度器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience//3, 
            min_lr=args.min_lr,
            verbose=True
        )
        loss_fn = torch.nn.MSELoss()
        
        # 训练状态跟踪
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 指标跟踪
        train_losses, val_losses = [], []
        train_maes, val_maes = [], []
        train_r2s, val_r2s = [], []
        
        # 混合精度梯度缩放器
        grad_scaler = amp.GradScaler(enabled=use_gpu)
        
        # 创建训练日志
        epoch_csv_path = os.path.join(fold_dir, "epoch_results.csv")
        with open(epoch_csv_path, 'w') as f:
            f.write("epoch,train_loss,train_mae,train_r2,val_loss,val_mae,val_r2\n")
        
        for epoch in range(1, args.n_epochs + 1):
            # 训练阶段
            model.train()
            train_outputs, train_preds = [], []
            total_train_loss = 0.0
            
            for data, outputs, _ in train_loader:
                node_mat, adj_mat, mask_mat = data
                batch_size = len(outputs)
                
                # 准备输入
                inputs = (
                    node_mat.reshape(batch_size, args.max_atoms, args.node_vec_len).to(device, non_blocking=True),
                    adj_mat.reshape(batch_size, args.max_atoms, args.max_atoms).to(device, non_blocking=True),
                    mask_mat.reshape(batch_size, args.max_atoms, args.node_vec_len).to(device, non_blocking=True)
                )
                targets = output_scaler.transform(outputs).to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with amp.autocast(device_type=device.type, enabled=use_gpu):
                    preds = model(*inputs).squeeze()
                    loss = loss_fn(preds, targets)
                
                # 反向传播与梯度裁剪
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # 收集指标
                with torch.no_grad():
                    preds_orig = output_scaler.restore(preds.detach().cpu().float())
                    total_train_loss += loss.item() * batch_size
                    train_outputs.append(outputs.numpy())
                    train_preds.append(preds_orig.numpy())
            
            # 计算训练指标
            train_loss = total_train_loss / len(train_loader.dataset)
            train_outputs = np.concatenate(train_outputs)
            train_preds = np.concatenate(train_preds)
            train_mae = mean_absolute_error(train_outputs, train_preds)
            train_r2 = r2_score(train_outputs, train_preds)
            
            # 验证阶段
            val_loss, val_mae, val_r2, _, val_outputs, val_preds = test_model(
                model, val_loader, loss_fn, device, output_scaler
            )
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            
            # 保存epoch结果
            with open(epoch_csv_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_mae:.6f},{train_r2:.6f},"
                        f"{val_loss:.6f},{val_mae:.6f},{val_r2:.6f}\n")
            
            # 打印带颜色的进度
            print(f"Epoch {epoch:03d}/{args.n_epochs}: "
                  f"Train Loss: \033[94m{train_loss:.6f}\033[0m | "
                  f"Val Loss: \033[92m{val_loss:.6f}\033[0m | "
                  f"Train MAE: \033[94m{train_mae:.6f}\033[0m | "
                  f"Val MAE: \033[92m{val_mae:.6f}\033[0m | "
                  f"Train R²: \033[94m{train_r2:.4f}\033[0m | "
                  f"Val R²: \033[92m{val_r2:.4f}\033[0m")
            
            # 定期保存
            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'output_scaler': output_scaler.state()
                }, os.path.join(fold_dir, f"model_epoch_{epoch}.pth"))
                
                save_parity_plot(fold_dir, train_outputs, train_preds, f"Training_Epoch_{epoch}")
                save_parity_plot(fold_dir, val_outputs, val_preds, f"Validation_Epoch_{epoch}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # 保存最佳结果
                save_parity_plot(fold_dir, train_outputs, train_preds, "Best_Training")
                save_parity_plot(fold_dir, val_outputs, val_preds, "Best_Validation")
                
                # 测试评估
                if test_loader:
                    test_loss, test_mae, test_r2, test_rmse, test_outputs, test_preds = test_model(
                        model, test_loader, loss_fn, device, output_scaler
                    )
                    save_parity_plot(fold_dir, test_outputs, test_preds, "Best_Test")
                    print(f"Test Results: Loss={test_loss:.6f} | MAE={test_mae:.6f} | R²={test_r2:.4f} | RMSE={test_rmse:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 定期更新损失曲线
            if epoch % 10 == 0:
                plot_loss_curve(fold_dir, list(range(1, epoch+1)), train_losses, val_losses)
        
        # 最终保存
        plot_loss_curve(fold_dir, list(range(1, len(train_losses)+1)), train_losses, val_losses)
        torch.save({
            'model_state_dict': best_model_state,
            'output_scaler': output_scaler.state(),
            'config': vars(args),
            'metrics': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_maes': train_maes,
                'val_maes': val_maes,
                'train_r2s': train_r2s,
                'val_r2s': val_r2s
            }
        }, os.path.join(fold_dir, "best_model.pth"))
        
        return best_model_state, best_val_loss

    def test_model(model, loader, loss_fn, device, output_scaler):
        """评估模型性能"""
        model.eval()
        total_loss = 0.0
        all_outputs, all_preds = [], []
        
        with torch.no_grad():
            for data, outputs, _ in loader:
                node_mat, adj_mat, mask_mat = data
                batch_size = len(outputs)
                
                # 准备输入
                inputs = (
                    node_mat.reshape(batch_size, args.max_atoms, args.node_vec_len).to(device),
                    adj_mat.reshape(batch_size, args.max_atoms, args.max_atoms).to(device),
                    mask_mat.reshape(batch_size, args.max_atoms, args.node_vec_len).to(device)
                )
                targets = output_scaler.transform(outputs).to(device)
                
                # 预测
                preds = model(*inputs).squeeze()
                loss = loss_fn(preds, targets)
                
                # 反标准化
                preds_orig = output_scaler.restore(preds.cpu().float())
                
                # 收集指标
                total_loss += loss.item() * batch_size
                all_outputs.append(outputs.numpy())
                all_preds.append(preds_orig.numpy())
        
        # 计算最终指标
        final_loss = total_loss / len(loader.dataset)
        all_outputs = np.concatenate(all_outputs)
        all_preds = np.concatenate(all_preds)
        mae = mean_absolute_error(all_outputs, all_preds)
        r2 = r2_score(all_outputs, all_preds)
        rmse = np.sqrt(mean_squared_error(all_outputs, all_preds))
        
        return final_loss, mae, r2, rmse, all_outputs, all_preds

    # K折交叉验证
    if args.k_folds > 0:
        # K折实现保持不变...
        pass
    else:
        # 简单划分
        indices = list(range(len(dataset)))
        test_size = 1 - args.train_size - args.val_size
        
        # 分层划分（确保数据分布）
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=args.seed
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=args.val_size/(args.train_size + args.val_size),
            random_state=args.seed
        )
        
        # 创建数据加载器
        train_loader = create_data_loaders(dataset, train_indices, args.batch_size)
        val_loader = create_data_loaders(dataset, val_indices, args.batch_size)
        test_loader = create_data_loaders(dataset, test_indices, args.batch_size)
        
        # 初始化模型
        model = GIN(
            node_vec_len=args.node_vec_len,
            node_fea_len=args.node_vec_len,
            hidden_fea_len=args.hidden_nodes,
            n_conv=args.n_conv_layers,
            n_hidden=args.n_hidden_layers,
            n_outputs=1,
            p_dropout=args.p_dropout
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练
        best_model_state, best_val_loss = train_model_fold(
            model, train_loader, val_loader, test_loader
        )
        
        # 最终保存
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

if __name__ == "__main__":
    main()