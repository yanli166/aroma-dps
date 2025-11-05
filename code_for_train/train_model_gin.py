"""Training script for ChemGIN model."""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from model_GIN import ChemGIN
from utils_GIN import train_model, test_model, save_parity_plot, loss_curve
from graphs_GIN import GraphData, process_and_save_data, collate_graph_dataset

def main():
    parser = argparse.ArgumentParser(description='Train ChemGIN model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--node_vec_len', type=int, default=60, help='Node vector length')
    parser.add_argument('--max_atoms', type=int, default=75, help='Max atoms per molecule')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--n_conv', type=int, default=3, help='Number of GIN layers')
    parser.add_argument('--n_hidden', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--k_folds', type=int, default=5, help='K-fold cross-validation')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Process data
    print("Processing data...")
    data_dict = process_and_save_data(
        args.dataset_path, 
        args.node_vec_len, 
        args.max_atoms
    )
    dataset = GraphData(data_dict, args.node_vec_len, args.max_atoms)
    
    # K-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold+1}/{args.k_folds} ===")
        fold_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create data loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            collate_fn=collate_graph_dataset
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=val_sampler,
            collate_fn=collate_graph_dataset
        )
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChemGIN(
            node_vec_len=args.node_vec_len,
            node_fea_len=args.hidden_dim,
            hidden_fea_len=args.hidden_dim,
            n_conv=args.n_conv,
            n_hidden=args.n_hidden,
            n_outputs=1,
            p_dropout=args.dropout
        ).to(device)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        loss_fn = nn.MSELoss()
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_preds, train_targets = train_model(
                model, train_loader, optimizer, loss_fn, device
            )
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_preds, val_targets = test_model(
                model, val_loader, loss_fn, device
            )
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                
                # Save predictions
                save_parity_plot(
                    fold_dir, 
                    train_targets, train_preds,
                    val_targets, val_preds,
                    f"fold_{fold+1}_parity.png"
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Save loss curve
        loss_curve(
            fold_dir, 
            list(range(1, len(train_losses)+1)), 
            train_losses, val_losses,
            f"fold_{fold+1}_loss.png"
        )
        
        # Save fold results
        fold_results.append({
            'fold': fold+1,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_loss
        })
    
    # Save final results
    print("\n=== Cross-Validation Results ===")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Loss = {result['best_val_loss']:.4f}")
    
    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    print(f"\nAverage Validation Loss: {avg_val_loss:.4f}")
    
    # Save results summary
    with open(os.path.join(args.output_dir, "results_summary.txt"), "w") as f:
        f.write("Cross-Validation Results:\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: Best Val Loss = {result['best_val_loss']:.4f}\n")
        f.write(f"\nAverage Validation Loss: {avg_val_loss:.4f}\n")

if __name__ == "__main__":
    main()