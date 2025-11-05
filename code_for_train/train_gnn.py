import os
import numpy as np
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from gnn.model_gnn import ChemGNN
from gnn.graphs_gnn import GraphData, process_and_save_data, collate_graph_dataset
from gnn.utils_gnn import Standardizer, save_parity_plot, loss_curve
from contextlib import nullcontext

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 参数解析 - 增强超参数选项
    parser = argparse.ArgumentParser(description='Train GNN model for molecular property prediction')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save results and models')
    parser.add_argument('--node_vec_len', type=int, default=60, 
                        help='Length of node vector')
    parser.add_argument('--max_atoms', type=int, default=75, 
                        help='Maximum number of atoms in molecules')
    parser.add_argument('--train_size', type=float, default=0.7, 
                        help='Proportion of data for training')
    parser.add_argument('--val_size', type=float, default=0.15, 
                        help='Proportion of data for validation (only for simple split)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--hidden_nodes', type=int, default=60, 
                        help='Number of hidden nodes')
    parser.add_argument('--n_conv_layers', type=int, default=5, 
                        help='Number of graph convolution layers')
    parser.add_argument('--n_hidden_layers', type=int, default=4, 
                        help='Number of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--n_epochs', type=int, default=300, 
                        help='Number of training epochs')
    parser.add_argument('--p_dropout', type=float, default=0.2, 
                        help='Dropout probability')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='Use GPU if available')
    parser.add_argument('--k_folds', type=int, default=0, 
                        help='Number of folds for cross-validation (0 for simple train/test split)')
    parser.add_argument('--patience', type=int, default=30, 
                        help='Patience for early stopping')
    parser.add_argument('--save_every', type=int, default=10, 
                        help='Save intermediate results every N epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    # 添加缺失的save_model参数
    parser.add_argument('--save_model', action='store_true', 
                        help='Save trained models')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 保存参数配置
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved configuration to {config_path}")
    
    # 准备数据集
    print(f"Processing dataset from: {args.dataset_path}")
    start_time = time.time()
    processed_data = process_and_save_data(
        dataset_path=args.dataset_path,
        node_vec_len=args.node_vec_len,
        max_atoms=args.max_atoms
    )
    dataset = GraphData(processed_data, args.node_vec_len, args.max_atoms)
    print(f"Dataset size: {len(dataset)} molecules")
    print(f"Data processing time: {time.time() - start_time:.2f} seconds")
    
    # 检查GPU可用性
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 收集所有输出用于标准化
    all_outputs = []
    for i in range(len(dataset)):
        all_outputs.append(dataset[i][1].item())
    standardizer = Standardizer(torch.tensor(all_outputs))
    
    # 训练函数 - 整合了详细的日志记录和中间结果保存
    def train_model_fold(model, train_loader, val_loader, test_loader=None, fold=None):
        # 创建结果目录
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}") if fold else args.output_dir
        os.makedirs(fold_dir, exist_ok=True)
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience//3, 
            min_lr=args.min_lr,
            verbose=True
        )
        
        # 损失函数
        loss_fn = torch.nn.MSELoss()
        
        # 早期停止设置
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 跟踪指标
        train_losses, val_losses = [], []
        train_maes, val_maes = [], []
        train_r2s, val_r2s = [], []
        test_results = {}
        
        # 创建CSV记录每个epoch的结果
        epoch_csv_path = os.path.join(fold_dir, "epoch_results.csv")
        with open(epoch_csv_path, 'w') as f:
            f.write("epoch,train_loss,train_mae,train_r2,val_loss,val_mae,val_r2\n")
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=use_gpu)
        
        for epoch in range(1, args.n_epochs + 1):
            model.train()
            total_train_loss, total_train_mae = 0.0, 0.0
            train_outputs, train_preds = [], []
            
            # 训练步骤
            for i, (data, outputs, _) in enumerate(train_loader):
                node_mat, adj_mat, mask_mat = data
                # 准备输入数据
                batch_size = len(outputs)
                inputs = (
                    node_mat.view(batch_size, args.max_atoms, args.node_vec_len).to(device),
                    adj_mat.view(batch_size, args.max_atoms, args.max_atoms).to(device),
                    mask_mat.view(batch_size, args.max_atoms, args.node_vec_len).to(device)
                )
                targets = outputs.to(device)
                
                optimizer.zero_grad()
                
                # 修复混合精度上下文管理器语法
                autocast_context = torch.amp.autocast(device_type='cuda', enabled=use_gpu) if use_gpu else nullcontext()
                with autocast_context:
                    predictions = model(*inputs).squeeze()
                    loss = loss_fn(predictions, targets)
                
                # 反向传播和梯度裁剪
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # 计算指标
                with torch.no_grad():
                    mae = torch.abs(predictions - targets).mean()
                    total_train_loss += loss.item() * batch_size
                    total_train_mae += mae.item() * batch_size
                    train_outputs.append(targets.cpu().numpy())
                    train_preds.append(predictions.cpu().numpy())
            
            # 计算平均训练指标
            train_loss = total_train_loss / len(train_loader.dataset)
            train_mae = total_train_mae / len(train_loader.dataset)
            train_outputs = np.concatenate(train_outputs)
            train_preds = np.concatenate(train_preds)
            train_r2 = r2_score(train_outputs, train_preds)
            
            # 验证步骤
            val_loss, val_mae, val_r2, val_rmse, val_outputs, val_preds = test_model(
                model, val_loader, loss_fn, device, args.max_atoms, args.node_vec_len
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
            
            # 保存epoch结果到CSV
            with open(epoch_csv_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_mae:.6f},{train_r2:.6f},"
                        f"{val_loss:.6f},{val_mae:.6f},{val_r2:.6f}\n")
            
            # 带颜色的打印输出
            print(f"Epoch {epoch:03d}/{args.n_epochs}: "
                  f"Train Loss: \033[94m{train_loss:.6f}\033[0m, "
                  f"Val Loss: \033[92m{val_loss:.6f}\033[0m, "
                  f"Train MAE: \033[94m{train_mae:.6f}\033[0m, "
                  f"Val MAE: \033[92m{val_mae:.6f}\033[0m, "
                  f"Train R²: \033[94m{train_r2:.4f}\033[0m, "
                  f"Val R²: \033[92m{val_r2:.4f}\033[0m")
            
            # 定期保存中间结果
            if args.save_every > 0 and epoch % args.save_every == 0:
                # 保存中间模型
                model_path = os.path.join(fold_dir, f"model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'standardizer_state': standardizer.state()
                }, model_path)
                
                # 保存中间预测结果
                preds_path = os.path.join(fold_dir, f"predictions_epoch_{epoch}.npz")
                np.savez(preds_path,
                         train_outputs=train_outputs,
                         train_preds=train_preds,
                         val_outputs=val_outputs,
                         val_preds=val_preds)
            
            # 检查是否改进并更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # 保存最佳预测结果
                save_parity_plot(fold_dir, train_outputs, train_preds, "Training Set")
                save_parity_plot(fold_dir, val_outputs, val_preds, "Validation Set")
                
                # 如果有测试集，进行测试
                if test_loader:
                    test_loss, test_mae, test_r2, test_rmse, test_outputs, test_preds = test_model(
                        model, test_loader, loss_fn, device, args.max_atoms, args.node_vec_len
                    )
                    test_results = {
                        'loss': test_loss,
                        'mae': test_mae,
                        'r2': test_r2,
                        'rmse': test_rmse,
                        'outputs': test_outputs,
                        'preds': test_preds
                    }
                    save_parity_plot(fold_dir, test_outputs, test_preds, "Test Set")
                    print(f"Test Results: Loss={test_loss:.6f}, MAE={test_mae:.6f}, R²={test_r2:.4f}, RMSE={test_rmse:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 定期更新损失曲线
            if epoch % 10 == 0:
                loss_curve(fold_dir, list(range(1, epoch+1)), train_losses, val_losses)
        
        # 保存最终损失曲线
        loss_curve(fold_dir, list(range(1, len(train_losses)+1)), train_losses, val_losses)
        
        # 保存最佳模型
        model_path = os.path.join(fold_dir, "best_model.pth")
        torch.save({
            'model_state_dict': best_model_state,
            'standardizer_state': standardizer.state(),
            'config': vars(args),
            'metrics': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_maes': train_maes,
                'val_maes': val_maes,
                'train_r2s': train_r2s,
                'val_r2s': val_r2s
            }
        }, model_path)
        print(f"Saved best model to {model_path}")
        
        # 保存结果摘要
        summary = {
            'best_val_loss': best_val_loss,
            'min_val_mae': min(val_maes),
            'max_val_r2': max(val_r2s),
            'final_val_mae': val_maes[-1],
            'final_val_r2': val_r2s[-1]
        }
        
        if test_results:
            summary.update({
                'test_loss': test_results['loss'],
                'test_mae': test_results['mae'],
                'test_r2': test_results['r2'],
                'test_rmse': test_results['rmse']
            })
        
        summary_path = os.path.join(fold_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved summary to {summary_path}")
        
        # 在函数结尾添加以下代码
        test_results['metrics'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_maes': train_maes,
            'val_maes': val_maes,
            'train_r2s': train_r2s,
            'val_r2s': val_r2s
        }

        # 保存最终模型
        if args.save_model:
            final_model_path = os.path.join(fold_dir, "final_model.pth")
            torch.save(model.state_dict(), final_model_path)

        return best_model_state, best_val_loss, test_results
    
    # 测试函数
    def test_model(model, test_loader, loss_fn, device, max_atoms, node_vec_len):
        model.eval()
        total_loss, total_mae = 0.0, 0.0
        all_outputs, all_preds = [], []
        
        with torch.no_grad():
            for data, outputs, _ in test_loader:
                node_mat, adj_mat, mask_mat = data
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
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                all_outputs.append(targets.cpu().numpy())
                all_preds.append(predictions.cpu().numpy())
        
        # 计算最终指标
        test_loss = total_loss / len(test_loader.dataset)
        test_mae = total_mae / len(test_loader.dataset)
        all_outputs = np.concatenate(all_outputs)
        all_preds = np.concatenate(all_preds)
        test_r2 = r2_score(all_outputs, all_preds)
        test_rmse = np.sqrt(mean_squared_error(all_outputs, all_preds))
        
        return test_loss, test_mae, test_r2, test_rmse, all_outputs, all_preds
    
    # K折交叉验证
    if args.k_folds > 0:
        print(f"\nStarting {args.k_folds}-fold cross-validation")
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        fold_results = []
        all_val_losses = []
        
        for fold, (train_val_indices, test_indices) in enumerate(kfold.split(dataset)):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{args.k_folds}")
            
            # 从训练集中划分验证集
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.15,
                random_state=args.seed
            )
            
            print(f"Dataset splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
            
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=collate_graph_dataset,
                num_workers=min(16, os.cpu_count())
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                collate_fn=collate_graph_dataset,
                num_workers=min(16, os.cpu_count())
            )
            
            test_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=test_sampler,
                collate_fn=collate_graph_dataset,
                num_workers=min(16, os.cpu_count())
            )
            
            # 初始化模型
            model = ChemGNN(
                node_vec_len=args.node_vec_len,
                node_fea_len=args.node_vec_len,
                hidden_fea_len=args.hidden_nodes,
                n_conv=args.n_conv_layers,
                n_hidden=args.n_hidden_layers,
                n_outputs=1,
                p_dropout=args.p_dropout
            ).to(device)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # 训练和验证
            best_model_state, best_val_loss, test_results = train_model_fold(
                model, train_loader, val_loader, test_loader, fold=fold+1
            )
            
            # 保存结果
            fold_results.append({
                'fold': fold+1,
                'best_val_loss': best_val_loss,
                'test_loss': test_results['loss'] if test_results else None,
                'test_mae': test_results['mae'] if test_results else None,
                'test_r2': test_results['r2'] if test_results else None,
                'test_rmse': test_results['rmse'] if test_results else None
            })
            all_val_losses.append(test_results['loss'] if test_results else best_val_loss)
        
        # 报告交叉验证结果
        print("\nCross-validation results:")
        avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
        avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
        avg_test_r2 = np.mean([r['test_r2'] for r in fold_results])
        
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        print(f"Average Test MAE: {avg_test_mae:.6f}")
        print(f"Average Test R²: {avg_test_r2:.4f}")
        
        # 保存交叉验证结果
        results_path = os.path.join(args.output_dir, "cv_results.csv")
        with open(results_path, 'w') as f:
            f.write("fold,best_val_loss,test_loss,test_mae,test_r2,test_rmse\n")
            for res in fold_results:
                f.write(f"{res['fold']},{res['best_val_loss']:.6f},{res['test_loss']:.6f},"
                        f"{res['test_mae']:.6f},{res['test_r2']:.6f},{res['test_rmse']:.6f}\n")
            f.write(f"average,{avg_val_loss:.6f},,{avg_test_mae:.6f},{avg_test_r2:.6f},\n")
        
        # 绘制所有fold的验证损失曲线
        plt.figure(figsize=(10, 6))
        for i in range(args.k_folds):
            fold_dir = os.path.join(args.output_dir, f"fold_{i+1}")
            if os.path.exists(os.path.join(fold_dir, "epoch_results.csv")):
                df = pd.read_csv(os.path.join(fold_dir, "epoch_results.csv"))
                plt.plot(df['val_loss'], label=f"Fold {i+1}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Across Folds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "cv_val_losses.png"))
        plt.close()
    
    # 简单训练/验证/测试划分
    else:
        # 划分数据
        indices = list(range(len(dataset)))
        train_val_indices, test_indices = train_test_split(
            indices, 
            test_size=1 - args.train_size - args.val_size,
            random_state=args.seed
        )
        
        # 从训练集中划分验证集
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=args.val_size/(args.train_size + args.val_size),
            random_state=args.seed
        )
        
        print(f"Dataset splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_indices),
            collate_fn=collate_graph_dataset,
            num_workers=min(16, os.cpu_count())
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_indices),
            collate_fn=collate_graph_dataset,
            num_workers=min(16, os.cpu_count())
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_indices),
            collate_fn=collate_graph_dataset,
            num_workers=min(16, os.cpu_count())
        )
        
        # 初始化模型
        model = ChemGNN(
            node_vec_len=args.node_vec_len,
            node_fea_len=args.node_vec_len,
            hidden_fea_len=args.hidden_nodes,
            n_conv=args.n_conv_layers,
            n_hidden=args.n_hidden_layers,
            n_outputs=1,
            p_dropout=args.p_dropout
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        best_model_state, best_val_loss, test_results = train_model_fold(
            model, train_loader, val_loader, test_loader
        )
        
        # 最终测试评估
        if test_results and 'metrics' in test_results:
            print("\nFinal Test Results:")
            print(f"Loss: {test_results['loss']:.6f} | MAE: {test_results['mae']:.6f} | "
                  f"R²: {test_results['r2']:.4f} | RMSE: {test_results['rmse']:.6f}")
            
            # 绘制综合性能图
            plt.figure(figsize=(14, 10))
            
            # 损失曲线
            plt.subplot(2, 2, 1)
            plt.plot(range(1, len(test_results['metrics']['train_losses'])+1), 
                    test_results['metrics']['train_losses'], 'b-', label='Training Loss')
            plt.plot(range(1, len(test_results['metrics']['val_losses'])+1), 
                    test_results['metrics']['val_losses'], 'r-', label='Validation Loss')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # MAE曲线
            plt.subplot(2, 2, 2)
            plt.plot(range(1, len(test_results['metrics']['train_maes'])+1), 
                    test_results['metrics']['train_maes'], 'b-', label='Training MAE')
            plt.plot(range(1, len(test_results['metrics']['val_maes'])+1), 
                    test_results['metrics']['val_maes'], 'r-', label='Validation MAE')
            plt.title('Training & Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            
            # R²曲线
            plt.subplot(2, 2, 3)
            plt.plot(range(1, len(test_results['metrics']['train_r2s'])+1), 
                    test_results['metrics']['train_r2s'], 'b-', label='Training R²')
            plt.plot(range(1, len(test_results['metrics']['val_r2s'])+1), 
                    test_results['metrics']['val_r2s'], 'r-', label='Validation R²')
            plt.title('Training & Validation R²')
            plt.xlabel('Epoch')
            plt.ylabel('R²')
            plt.legend()
            plt.grid(True)
            
            # 测试集预测散点图
            plt.subplot(2, 2, 4)
            plt.scatter(test_results['outputs'], test_results['preds'], alpha=0.6, edgecolor='k')
            min_val = min(test_results['outputs'].min(), test_results['preds'].min())
            max_val = max(test_results['outputs'].max(), test_results['preds'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title('Test Set Predictions')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "performance_summary.png"))
            plt.close()
        elif test_results:
            print("\nFinal Test Results:")
            print(f"Loss: {test_results['loss']:.6f} | MAE: {test_results['mae']:.6f} | "
                  f"R²: {test_results['r2']:.4f} | RMSE: {test_results['rmse']:.6f}")
            print("Warning: Metrics history not available for plotting")
          
if __name__ == "__main__":
    main()