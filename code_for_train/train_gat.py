import os
import numpy as np
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from gat.model_gat import ChemGCN
from gat.graphs_gat import GraphData, process_and_save_data, collate_graph_dataset
from gat.utils_gat import train_model, test_model, loss_curve, save_parity_plot, create_scheduler, Standardizer

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def main():
    # 参数解析 - 增强超参数选项
    parser = argparse.ArgumentParser(description='Train GAT model for molecular property prediction')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save results and models')
    parser.add_argument('--node_vec_len', type=int, default=60, 
                        help='Length of node vector')
    parser.add_argument('--max_atoms', type=int, default=75, 
                        help='Maximum number of atoms in molecules')
    parser.add_argument('--train_size', type=float, default=0.8, 
                        help='Proportion of data for training')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--hidden_nodes', type=int, default=60, 
                        help='Number of hidden nodes')
    parser.add_argument('--n_conv_layers', type=int, default=3, 
                        help='Number of graph convolution layers')
    parser.add_argument('--n_hidden_layers', type=int, default=2, 
                        help='Number of hidden layers')
    parser.add_argument('--n_heads', type=int, default=2, 
                        help='Number of attention heads in GAT')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
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
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 保存参数配置
    config_df = pd.DataFrame([vars(args)])
    config_df.to_csv(os.path.join(args.output_dir, "config.csv"), index=False)
    
    # 准备数据集
    print(f"Processing dataset from: {args.dataset_path}")
    processed_data = process_and_save_data(
        dataset_path=args.dataset_path,
        node_vec_len=args.node_vec_len,
        max_atoms=args.max_atoms
    )
    
    dataset = GraphData(processed_data, args.node_vec_len, args.max_atoms)
    print(f"Dataset size: {len(dataset)} molecules")
    
    # 检查GPU可用性
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # 创建完整数据加载器用于标准化
    full_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_dataset,
        num_workers=min(32, os.cpu_count())
    )
    
    # 收集所有输出用于标准化
    all_outputs = []
    for data in full_loader:
        all_outputs.append(data[1])
    all_outputs = torch.cat(all_outputs)
    standardizer = Standardizer(all_outputs)
    
    # 训练函数 - 整合了详细的日志记录和中间结果保存
    def train_model_fold(model, train_loader, val_loader, fold=None):
        # 创建优化器和调度器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        scheduler = create_scheduler(
            optimizer, 
            patience=args.patience//3, 
            factor=0.5, 
            min_lr=args.min_lr
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
        
        # 创建结果目录
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}") if fold else args.output_dir
        os.makedirs(fold_dir, exist_ok=True)
        
        # 创建CSV记录每个epoch的结果
        epoch_csv_path = os.path.join(fold_dir, "epoch_results.csv")
        with open(epoch_csv_path, 'w') as f:
            f.write("epoch,train_loss,train_mae,train_r2,val_loss,val_mae,val_r2\n")
        
        for epoch in range(1, args.n_epochs + 1):
            # 训练步骤
            train_loss, train_mae, train_r2, train_outputs, train_preds = train_model(
                epoch, 
                model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                device=device,
                max_atoms=args.max_atoms,
                node_vec_len=args.node_vec_len,
                scheduler=scheduler
            )
            
            # 验证步骤
            val_loss, val_mae, val_r2, rmse, val_outputs, val_preds = test_model(
              model, 
              val_loader, 
              loss_fn, 
              device=device,
              max_atoms=args.max_atoms,
              node_vec_len=args.node_vec_len
            )
            
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
                  f"Train R²: \033[94m{train_r2:.6f}\033[0m, "
                  f"Val R²: \033[92m{val_r2:.6f}\033[0m")
            
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
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 定期更新损失曲线
            if epoch % 10 == 0:
                loss_curve(fold_dir, list(range(1, epoch+1)), train_losses, val_losses)
        
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
        
        # 保存最终损失曲线
        loss_curve(fold_dir, list(range(1, len(train_losses)+1)), train_losses, val_losses)
        
        # 保存结果摘要
        summary_path = os.path.join(fold_dir, "summary.csv")
        with open(summary_path, 'w') as f:
            f.write("metric,value\n")
            f.write(f"best_val_loss,{best_val_loss:.6f}\n")
            f.write(f"final_val_mae,{val_maes[-1]:.6f}\n")
            f.write(f"final_val_r2,{val_r2s[-1]:.6f}\n")
            f.write(f"min_val_mae,{min(val_maes):.6f}\n")
            f.write(f"max_val_r2,{max(val_r2s):.6f}\n")
        
        return best_model_state, best_val_loss, val_outputs, val_preds, rmse, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_maes': train_maes,
            'val_maes': val_maes,
            'train_r2s': train_r2s,
            'val_r2s': val_r2s
        }
    
    # K折交叉验证
    if args.k_folds > 0:
        print(f"Starting {args.k_folds}-fold cross-validation")
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
        fold_results = []
        all_val_losses = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\nFold {fold+1}/{args.k_folds}")
            
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=collate_graph_dataset,
                num_workers=min(32, os.cpu_count())
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                collate_fn=collate_graph_dataset,
                num_workers=min(32, os.cpu_count())
            )
            
            # 初始化模型
            model = ChemGCN(
                node_vec_len=args.node_vec_len,
                node_fea_len=args.node_vec_len,
                hidden_fea_len=args.hidden_nodes,
                n_conv=args.n_conv_layers,
                n_hidden=args.n_hidden_layers,
                n_outputs=1,
                p_dropout=args.p_dropout,
                n_heads=args.n_heads
            ).to(device)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # 训练和验证
            result = train_model_fold(
                model, train_loader, val_loader, fold=fold+1
            )
            best_model_state, best_val_loss, val_outputs, val_preds, _, metrics = result
            
            # 保存结果
            fold_results.append({
                'fold': fold+1,
                'val_loss': val_loss,
                'val_mae': mean_absolute_error(val_outputs, val_preds),
                'val_r2': r2_score(val_outputs, val_preds),
                'metrics': metrics
            })
            all_val_losses.append(metrics['val_losses'])
        
        # 报告交叉验证结果
        print("\nCross-validation results:")
        avg_loss = np.mean([r['val_loss'] for r in fold_results])
        avg_mae = np.mean([r['val_mae'] for r in fold_results])
        avg_r2 = np.mean([r['val_r2'] for r in fold_results])
        
        print(f"Average Validation Loss: {avg_loss:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
        
        # 保存交叉验证结果
        results_path = os.path.join(args.output_dir, "cv_results.csv")
        with open(results_path, 'w') as f:
            f.write("fold,val_loss,val_mae,val_r2\n")
            for res in fold_results:
                f.write(f"{res['fold']},{res['val_loss']:.6f},{res['val_mae']:.6f},{res['val_r2']:.6f}\n")
            f.write(f"average,{avg_loss:.6f},{avg_mae:.6f},{avg_r2:.6f}\n")
        
        # 绘制所有fold的验证损失曲线
        plt.figure(figsize=(10, 6))
        for i, losses in enumerate(all_val_losses):
            plt.plot(range(1, len(losses)+1), losses, label=f"Fold {i+1}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Across Folds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "cv_val_losses.png"))
        plt.close()
    
    # 简单训练/测试划分
    else:
        # 划分数据
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(args.train_size * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        
        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_graph_dataset,
            num_workers=min(32, os.cpu_count())
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=collate_graph_dataset,
            num_workers=min(32, os.cpu_count())
        )
        
        # 初始化模型
        model = ChemGCN(
            node_vec_len=args.node_vec_len,
            node_fea_len=args.node_vec_len,
            hidden_fea_len=args.hidden_nodes,
            n_conv=args.n_conv_layers,
            n_hidden=args.n_hidden_layers,
            n_outputs=1,
            p_dropout=args.p_dropout,
            n_heads=args.n_heads
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        result = train_model_fold(model, train_loader, val_loader)
        best_model_state, best_val_loss, val_outputs, val_preds, _, metrics = result
        
        # 保存最终模型
        model_path = os.path.join(args.output_dir, "best_model.pth")
        torch.save({
            'model_state_dict': best_model_state,
            'standardizer_state': standardizer.state(),
            'config': vars(args),
            'metrics': metrics
        }, model_path)
        
        # 测试最佳模型
        model.load_state_dict(best_model_state)
        test_loss, test_mae, test_r2, rmse, test_outputs, test_preds = test_model(
            model, val_loader, torch.nn.MSELoss(), 
            device=device,
            max_atoms=args.max_atoms,
            node_vec_len=args.node_vec_len
        )
        
        # 保存测试结果
        print("\nTest Results:")
        print(f"Loss: {test_loss:.6f} | MAE: {test_mae:.6f} | R²: {test_r2:.6f}")
        
        # 保存预测结果
        save_parity_plot(args.output_dir, test_outputs, test_preds, "Test Set")
        
        # 绘制综合性能图
        plt.figure(figsize=(14, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(metrics['train_losses'], 'b-', label='Training Loss')
        plt.plot(metrics['val_losses'], 'r-', label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # MAE曲线
        plt.subplot(2, 2, 2)
        plt.plot(metrics['train_maes'], 'b-', label='Training MAE')
        plt.plot(metrics['val_maes'], 'r-', label='Validation MAE')
        plt.title('Training & Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # R²曲线
        plt.subplot(2, 2, 3)
        plt.plot(metrics['train_r2s'], 'b-', label='Training R²')
        plt.plot(metrics['val_r2s'], 'r-', label='Validation R²')
        plt.title('Training & Validation R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        
        # 测试集预测散点图
        plt.subplot(2, 2, 4)
        plt.scatter(test_outputs, test_preds, alpha=0.6, edgecolor='k')
        min_val = min(test_outputs.min(), test_preds.min())
        max_val = max(test_outputs.max(), test_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title('Test Set Predictions')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "performance_summary.png"))
        plt.close()

if __name__ == "__main__":
    main()