
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
"""
Step 5: 主模型 — Siamese / Paired MPNN

基于现有 Stage I MPNN 建立共享权重的 Siamese MPNN。
  h_i = encoder(G_i)
  h_j = encoder(G_j)
  z = [h_i, h_j, h_i - h_j, |h_i - h_j|]
  MLP(z) -> delta_A

ring_flag=10, target ring 信息保留, scaffold-grouped split, 5 seeds。

输出:
  results/lunci10_delta_learning/04_siamese_mpnn/
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import GroupShuffleSplit

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, set_full_seed
from unified_models.mpnn.model import MPNNModel

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_v2/04_siamese_mpnn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIR_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_v2/01_pair_dataset')

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10
SEEDS = [42, 123, 456, 789, 2024]

TASK_COL_MAP = {
    'HOMA':      ('HOMA',    'homa_value'),
    'NICS_1zz':  ('NICS_ZZ', 'NICS_value'),
    'MBCO':      ('MBCO',    'mbco_value'),
}


class SiameseMPNN(nn.Module):
    """Siamese MPNN: 共享权重 encoder + pair MLP

    encoder = MPNN 的 init_transform + conv_layers + pooling + hidden_layers
    pair_head = MLP([h_i, h_j, h_i - h_j, |h_i - h_j|]) → delta_A
    """

    def __init__(self, node_vec_len, hidden_dim, n_conv, n_hidden,
                 p_dropout=0.2, mode='label'):
        super().__init__()

        # 共享 encoder (完整 MPNN, 输出 1 维 → 我们取 hidden 层输出)
        self.encoder = MPNNModel(
            node_vec_len=node_vec_len, hidden_dim=hidden_dim,
            n_conv=n_conv, n_hidden=n_hidden, n_outputs=1,
            p_dropout=p_dropout, mode=mode
        )

        # Pair head: 输入 = [h_i, h_j, h_i - h_j, |h_i - h_j|] = 4 * hidden_dim
        pair_dim = 4 * hidden_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, node_mat, adj_mat):
        """编码单个分子 → hidden representation"""
        enc = self.encoder
        batch_size, n_atoms, _ = node_mat.shape
        node_fea = enc.init_transform(
            node_mat.reshape(-1, enc.node_vec_len)
        ).reshape(batch_size, n_atoms, -1)

        for i, conv in enumerate(enc.conv_layers):
            residual = node_fea
            node_fea = conv(node_fea, adj_mat)
            if residual.size() == node_fea.size():
                node_fea = node_fea + residual
            node_fea = node_fea.transpose(1, 2)
            node_fea = getattr(enc, f'conv_bn_{i}')(node_fea)
            node_fea = node_fea.transpose(1, 2)
            node_fea = F.leaky_relu(node_fea, 0.2)

        pooled = enc.pooling_activation(enc.pooling(node_fea))

        for i in range(len(enc.hidden_layers)):
            pooled = enc.hidden_layers[i](pooled)
            pooled = enc.hidden_bns[i](pooled)
            pooled = F.leaky_relu(pooled, 0.2)
            pooled = F.dropout(pooled, 0.2, training=self.training)

        return pooled  # (batch, hidden_dim)

    def forward(self, node_mat_i, adj_mat_i, node_mat_j, adj_mat_j):
        h_i = self.encode(node_mat_i, adj_mat_i)
        h_j = self.encode(node_mat_j, adj_mat_j)

        z = torch.cat([h_i, h_j, h_i - h_j, (h_i - h_j).abs()], dim=-1)
        return self.pair_head(z).squeeze(-1)


def load_lunci10_pairs(task_name, task_info, device):
    """加载 lunci10 数据并构建 pair 索引

    返回:
      data: dict with node_mats, adj_mats, outputs, n
      df_pairs: pair dataset
      pair_indices: (idx_i, idx_j) for each pair in the lunci10 data
    """
    l10_col, target_col = TASK_COL_MAP[task_name]
    df_l10 = pd.read_csv(os.path.join(PROJ_ROOT, '..', 'lunci10', 'lunci10-test-corrected.csv'),
                         encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=[l10_col, 'SMILES']).reset_index(drop=True)

    # 保存为临时 CSV 供 load_adj_format 使用
    tmp_path = os.path.join('/tmp', f'lunci10_{task_name}_siamese.csv')
    out_df = df_l10[['SMILES', l10_col, 'Ring_ID', 'Ring_Atoms', 'New_ID']].copy()
    out_df.columns = ['smiles', target_col, 'Ring_ID', 'Ring_Atoms', 'New_ID']
    out_df['atom_on_ring'] = df_l10['Ring_Atoms']
    out_df.to_csv(tmp_path, index=False)

    l10_data = load_adj_format(tmp_path, target_col, NVL, MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    # 加载 pair dataset
    pair_file = os.path.join(PAIR_DIR, f'pair_dataset_{task_name.lower().replace("_1zz","")}.csv')
    df_pairs = pd.read_csv(pair_file)

    # 构建 (New_ID, Ring_ID) → row_idx 映射
    id_ring_to_idx = {}
    for idx, row in df_l10.iterrows():
        key = (row['New_ID'], row['Ring_ID'])
        id_ring_to_idx[key] = idx

    # 为每个 pair 找到对应的行索引
    pair_indices = []
    valid_pair_mask = []
    for _, pair in df_pairs.iterrows():
        key_i = (pair['new_id_i'], pair['ring_id'])
        key_j = (pair['new_id_j'], pair['ring_id'])
        if key_i in id_ring_to_idx and key_j in id_ring_to_idx:
            pair_indices.append((id_ring_to_idx[key_i], id_ring_to_idx[key_j]))
            valid_pair_mask.append(True)
        else:
            pair_indices.append((-1, -1))
            valid_pair_mask.append(False)

    valid_pair_mask = np.array(valid_pair_mask)
    df_pairs = df_pairs[valid_pair_mask].reset_index(drop=True)
    pair_indices = np.array(pair_indices)[valid_pair_mask]

    print(f"  Loaded {len(df_pairs)} valid pairs (out of {len(valid_pair_mask)})")
    return l10_data, df_pairs, pair_indices


def train_siamese(model, data, pair_indices, y_delta, groups,
                  train_idx, val_idx, device, n_epochs, patience, seed):
    """训练 Siamese MPNN"""
    set_full_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    y_tensor = torch.tensor(y_delta, dtype=torch.float32, device=device)

    best_val, best_state, pcount = float('inf'), None, 0
    batch_size = 64

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        for i in range(0, len(train_idx), batch_size):
            batch_pairs = train_idx[perm[i:i + batch_size]]
            if len(batch_pairs) < 2:
                continue

            idx_i = pair_indices[batch_pairs, 0]
            idx_j = pair_indices[batch_pairs, 1]
            idx_i = torch.tensor(idx_i, device=device, dtype=torch.long)
            idx_j = torch.tensor(idx_j, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            preds = model(
                node_mats[idx_i], adj_mats[idx_i],
                node_mats[idx_j], adj_mats[idx_j]
            )
            loss = loss_fn(preds, y_tensor[batch_pairs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(val_idx), batch_size):
                batch_pairs = val_idx[i:i + batch_size]
                if len(batch_pairs) < 2:
                    batch_pairs = val_idx[i:i + batch_size]
                idx_i = pair_indices[batch_pairs, 0]
                idx_j = pair_indices[batch_pairs, 1]
                idx_i = torch.tensor(idx_i, device=device, dtype=torch.long)
                idx_j = torch.tensor(idx_j, device=device, dtype=torch.long)
                vp.append(model(
                    node_mats[idx_i], adj_mats[idx_i],
                    node_mats[idx_j], adj_mats[idx_j]
                ))
            vp = torch.cat(vp)
            vloss = loss_fn(vp, y_tensor[val_idx]).item()

        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model


@torch.no_grad()
def evaluate_siamese(model, data, pair_indices, y_delta, eval_idx, device, batch_size=64):
    """评估 Siamese MPNN"""
    model.eval()
    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    y_tensor = torch.tensor(y_delta, dtype=torch.float32, device=device)

    preds = []
    for i in range(0, len(eval_idx), batch_size):
        batch_pairs = eval_idx[i:i + batch_size]
        if len(batch_pairs) < 2:
            batch_pairs = eval_idx[i:i + batch_size]
        idx_i = pair_indices[batch_pairs, 0]
        idx_j = pair_indices[batch_pairs, 1]
        idx_i = torch.tensor(idx_i, device=device, dtype=torch.long)
        idx_j = torch.tensor(idx_j, device=device, dtype=torch.long)
        preds.append(model(
            node_mats[idx_i], adj_mats[idx_i],
            node_mats[idx_j], adj_mats[idx_j]
        ))
    preds = torch.cat(preds).cpu().numpy()

    true = y_delta[eval_idx]
    r2, mae, rmse = compute_metrics(true, preds)
    rho, _ = spearmanr(true, preds)
    tau, _ = kendalltau(true, preds)

    correct = 0
    total = 0
    for td, pd in zip(true, preds):
        if td == 0 or pd == 0:
            continue
        if np.sign(td) == np.sign(pd):
            correct += 1
        total += 1
    pairwise_acc = correct / total if total > 0 else np.nan

    return {
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'spearman': rho, 'kendall': tau,
        'pairwise_acc': pairwise_acc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    task_map = {t['name']: t for t in TASKS}
    tasks = args.tasks.split(',')

    all_metrics = []

    for task_name in tasks:
        task = task_map[task_name]
        print(f"\n{'='*70}")
        print(f"# {task_name}")
        print(f"{'='*70}")

        # 加载数据和 pairs
        l10_data, df_pairs, pair_indices = load_lunci10_pairs(task_name, task, device)

        y_delta = df_pairs['delta_A'].values.astype(np.float32)
        groups = df_pairs['scaffold_id'].values

        print(f"  Pairs: {len(y_delta)}")
        print(f"  Scaffolds: {len(np.unique(groups))}")

        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")
            set_full_seed(seed)

            # scaffold-grouped split: 80/20
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_val_idx, test_idx = next(gss.split(np.arange(len(y_delta)), y_delta, groups))

            # 进一步划分 train/val: 87.5%/12.5%
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            train_idx, val_idx = next(gss2.split(train_val_idx, y_delta[train_val_idx],
                                                  groups[train_val_idx]))
            train_idx = train_val_idx[train_idx]
            val_idx = train_val_idx[val_idx]

            print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # 构建模型
            model = SiameseMPNN(
                node_vec_len=NVL, hidden_dim=128,
                n_conv=3, n_hidden=2, p_dropout=0.2, mode='label'
            )

            # 训练
            t0 = time.time()
            model = train_siamese(model, l10_data, pair_indices, y_delta, groups,
                                  train_idx, val_idx, device,
                                  n_epochs=200, patience=30, seed=seed)
            train_time = time.time() - t0

            # 评估
            metrics = evaluate_siamese(model, l10_data, pair_indices, y_delta,
                                        test_idx, device)
            metrics['task'] = task_name
            metrics['seed'] = seed
            metrics['train_time'] = train_time
            metrics['n_train'] = len(train_idx)
            metrics['n_test'] = len(test_idx)
            all_metrics.append(metrics)

            print(f"    Δ R²={metrics['r2']:.4f} | Δ MAE={metrics['mae']:.4f} | "
                  f"Spearman={metrics['spearman']:.4f} | Kendall={metrics['kendall']:.4f} | "
                  f"Pairwise={metrics['pairwise_acc']:.4f} ({train_time:.0f}s)")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 保存结果
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'siamese_per_seed.csv'), index=False)

    agg = df_metrics.groupby('task').agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg.to_csv(os.path.join(OUTPUT_DIR, 'siamese_agg.csv'), index=False)

    print(f"\n{'='*70}")
    print("Step 5: Siamese MPNN (mean±SD over 5 seeds)")
    print(f"{'='*70}")
    for _, r in agg.iterrows():
        print(f"\n  [{r['task']}]")
        print(f"    Δ R²       = {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"    Δ MAE      = {r['mae_mean']:.4f} ± {r['mae_std']:.4f}")
        print(f"    Δ RMSE     = {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"    Spearman ρ = {r['spearman_mean']:.4f} ± {r['spearman_std']:.4f}")
        print(f"    Kendall τ  = {r['kendall_mean']:.4f} ± {r['kendall_std']:.4f}")
        print(f"    Pairwise   = {r['pairwise_mean']:.4f} ± {r['pairwise_std']:.4f}")

    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
