"""
统一的训练脚本 - 支持5种模型 × 3种编码方式 = 15个实验
模型: GAT, GIN, GNN, MPNN, GraphSAGE
编码: label, mask, pool
"""
import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from collections import defaultdict

# 添加项目根目录到路径
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from unified_models.common.graphs import process_and_save_data, GraphData, collate_graph_dataset
from unified_models.common.utils import (
    Standardizer, train_one_epoch, evaluate,
    plot_loss_curve, save_parity_plot
)
from unified_models.gat.model import GATModel
from unified_models.gin.model import GINModel
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from unified_models.graphsage.model import GraphSAGEModel


MODELS = {
    'gat': GATModel,
    'gin': GINModel,
    'gnn': GNNModel,
    'mpnn': MPNNModel,
    'graphsage': GraphSAGEModel,
}


def get_scaffold(smiles, include_chirality=False):
    """生成Bemis-Murcko骨架"""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def split_dataset(dataset, smiles_list, train_size, splitter='random', seed=42):
    """划分数据集 (返回train, val Subset)"""
    n = len(dataset)
    rng = np.random.RandomState(seed)

    if splitter == 'random':
        indices = list(range(n))
        rng.shuffle(indices)
        split = int(np.floor(train_size * n))
        train_idx, val_idx = indices[:split], indices[split:]
    elif splitter == 'scaffold':
        # 按骨架划分
        scaffolds = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            try:
                scaf = get_scaffold(smi)
            except Exception:
                scaf = smi
            scaffolds[scaf].append(i)
        scaffold_sets = list(scaffolds.values())
        # 大骨架优先训练集
        scaffold_sets = sorted(scaffold_sets, key=lambda x: (len(x), x[0]), reverse=True)
        train_idx, val_idx = [], []
        n_train = int(train_size * n)
        for sset in scaffold_sets:
            if len(train_idx) + len(sset) > n_train:
                val_idx.extend(sset)
            else:
                train_idx.extend(sset)
    elif splitter == 'random_scaffold':
        scaffolds = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            try:
                scaf = get_scaffold(smi)
            except Exception:
                scaf = smi
            scaffolds[scaf].append(i)
        scaffold_sets = list(scaffolds.values())
        rng.shuffle(scaffold_sets)
        train_idx, val_idx = [], []
        n_train = int(train_size * n)
        for sset in scaffold_sets:
            if len(train_idx) + len(sset) > n_train:
                val_idx.extend(sset)
            else:
                train_idx.extend(sset)
    else:
        raise ValueError(f"Unknown splitter: {splitter}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    return train_subset, val_subset


def main():
    parser = argparse.ArgumentParser(description='Unified training for GNN models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gat', 'gin', 'gnn', 'mpnn', 'graphsage'],
                       help='Model type')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['label', 'mask', 'pool'],
                       help='Ring info encoding mode')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_col', type=str, default='homa_value')
    parser.add_argument('--node_vec_len', type=int, default=60)
    parser.add_argument('--max_atoms', type=int, default=75)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_conv_layers', type=int, default=3)
    parser.add_argument('--n_hidden_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4, help='For GAT only')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--p_dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splitter', type=str, default='random',
                       choices=['random', 'scaffold', 'random_scaffold'])
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting')
    parser.add_argument('--n_threads', type=int, default=0, help='CPU threads (0=all)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.n_threads > 0:
        torch.set_num_threads(args.n_threads)

    # 保存配置
    with open(os.path.join(args.output_dir, 'config.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['param', 'value'])
        for k, v in vars(args).items():
            writer.writerow([k, v])

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    print(f"[{args.model}-{args.mode}] Using device: {device}")

    # 加载数据
    print(f"Loading data from {args.dataset_path}...")
    data = process_and_save_data(args.dataset_path, args.node_vec_len,
                                  args.max_atoms, args.target_col)
    dataset = GraphData(data, args.node_vec_len, args.max_atoms)
    smiles_list = data['smiles']
    print(f"Dataset size: {len(dataset)}")

    # 划分
    train_set, val_set = split_dataset(dataset, smiles_list,
                                        args.train_size, args.splitter, args.seed)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_graph_dataset, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_graph_dataset, num_workers=0)

    # 标准化输出
    outputs_tensor = torch.Tensor(data['outputs']).squeeze()
    std = Standardizer(outputs_tensor)

    # 创建模型
    model_cls = MODELS[args.model]
    model_kwargs = dict(
        node_vec_len=args.node_vec_len,
        hidden_dim=args.hidden_dim,
        n_conv=args.n_conv_layers,
        n_hidden=args.n_hidden_layers,
        n_outputs=1,
        p_dropout=args.p_dropout,
        mode=args.mode,
    )
    if args.model == 'gat':
        model_kwargs['n_heads'] = args.n_heads
    model = model_cls(**model_kwargs).to(device)

    # 优化器、调度器、损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.patience // 3, min_lr=args.min_lr
    )
    loss_fn = nn.MSELoss()

    # 训练
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    epochs_list, train_losses, val_losses = [], [], []
    epoch_records = []

    for epoch in range(1, args.n_epochs + 1):
        train_loss, train_mae, train_r2, _, _ = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, args.mode,
            args.max_atoms, args.node_vec_len, scheduler
        )
        val_loss, val_mae, val_r2, val_rmse, _, _ = evaluate(
            model, val_loader, loss_fn, device, args.mode,
            args.max_atoms, args.node_vec_len
        )

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
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.n_epochs}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 保存最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

    # 评估
    train_loss, train_mae, train_r2, train_rmse, train_true, train_pred = evaluate(
        model, train_loader, loss_fn, device, args.mode, args.max_atoms, args.node_vec_len
    )
    val_loss, val_mae, val_r2, val_rmse, val_true, val_pred = evaluate(
        model, val_loader, loss_fn, device, args.mode, args.max_atoms, args.node_vec_len
    )

    print(f"\nFinal Train: Loss={train_loss:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}, RMSE={train_rmse:.4f}")
    print(f"Final Val:   Loss={val_loss:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}, RMSE={val_rmse:.4f}")

    # 保存结果
    with open(os.path.join(args.output_dir, 'epoch_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_mae', 'train_r2',
                                                'val_loss', 'val_mae', 'val_r2', 'val_rmse'])
        writer.writeheader()
        writer.writerows(epoch_records)

    with open(os.path.join(args.output_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['best_val_loss', best_val_loss])
        writer.writerow(['final_train_loss', train_loss])
        writer.writerow(['final_train_mae', train_mae])
        writer.writerow(['final_train_r2', train_r2])
        writer.writerow(['final_train_rmse', train_rmse])
        writer.writerow(['final_val_loss', val_loss])
        writer.writerow(['final_val_mae', val_mae])
        writer.writerow(['final_val_r2', val_r2])
        writer.writerow(['final_val_rmse', val_rmse])
        writer.writerow(['min_val_mae', min(r['val_mae'] for r in epoch_records)])
        writer.writerow(['max_val_r2', max(r['val_r2'] for r in epoch_records)])

    # 保存预测数据
    pd.DataFrame({'true': train_true.numpy(), 'pred': train_pred.numpy()}).to_csv(
        os.path.join(args.output_dir, 'train_set_data.csv'), index=False)
    pd.DataFrame({'true': val_true.numpy(), 'pred': val_pred.numpy()}).to_csv(
        os.path.join(args.output_dir, 'val_set_data.csv'), index=False)

    # 绘图
    if not args.no_plot:
        plot_loss_curve(args.output_dir, epochs_list, train_losses, val_losses)
        save_parity_plot(args.output_dir, val_true.numpy(), val_pred.numpy(),
                        title=f'{args.model}_{args.mode}_val_parity')

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
