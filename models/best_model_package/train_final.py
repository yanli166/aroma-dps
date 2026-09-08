"""最终模型训练脚本 (自包含, 修复 -1 bug)

为三个任务 (HOMA / NICS_1zz / MBCO) 训练最终交付模型:
  - 数据: 原 CSV (smiles, atom_on_ring, target)
  - split: 固定 80/20 holdout (seed=2026, group-aware), dev 内再 87.5/12.5 final split
  - 最优配置 (来自 final_membership):
      HOMA:     membership_proj (ring_flag=1 + nn.Embedding projection)
      NICS_1zz: membership_1    (ring_flag=1)
      MBCO:     membership_1    (ring_flag=1)
  - 训练: final_train_idx 训练 + final_val_idx early stopping, 保存 best val MAE 权重

输出 (best_model_package/):
  homa_best.pt / nics_best.pt / mbco_best.pt + metrics.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error

from graph_utils import build_graph
from model_arch import build_model

NODE_VEC_LEN, MAX_ATOMS = 60, 75
SPLIT_SEED = 2026
MODEL_SEED = 11
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}

DATA_ROOT = '/home/ubuntu/aroma-dps-code/code_end/data1_end'
TASKS = [
    {'name': 'HOMA',     'path': os.path.join(DATA_ROOT, 'collet_homa_0716.csv'), 'target': 'homa_value',  'proj': True},
    {'name': 'NICS_1zz', 'path': os.path.join(DATA_ROOT, 'collet_nics_0716.csv'), 'target': 'NICS_value',  'proj': False},
    {'name': 'MBCO',     'path': os.path.join(DATA_ROOT, 'collet_mbco_0716.csv'), 'target': 'mbco_value',  'proj': False},
]

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_aor(v):
    return eval(v) if isinstance(v, str) else list(v)


def load_data(task):
    df = pd.read_csv(task['path'])
    df = df.dropna(subset=[task['target'], 'smiles']).reset_index(drop=True)
    smiles = df['smiles'].tolist()
    aor = df['atom_on_ring'].apply(parse_aor).tolist()
    y = df[task['target']].astype(float).values
    return smiles, aor, y, df


def make_group_ids(smiles_list):
    from rdkit import Chem
    gs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        gs.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return np.asarray(gs)


def final_train_val_split(n, groups, seed=SPLIT_SEED):
    """dev 内 87.5/12.5 group-aware split -> (final_train_idx, final_val_idx)"""
    idx = np.arange(n)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tr_rel, va_rel = next(gss.split(idx, groups=groups))
    return idx[tr_rel], idx[va_rel]


def load_all_graphs(smiles, aor, ring_flag):
    """一次性构建全部分子图并堆叠为张量"""
    node_mats, adj_mats, ring_indices = [], [], []
    for smi, atoms in zip(smiles, aor):
        g = build_graph(smi, atoms, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=ring_flag)
        node_mats.append(g['node_mat'])
        adj_mats.append(g['adj_mat'])
        ring_indices.append(g['ring_indices'])
    return {
        'node_mats': torch.tensor(np.array(node_mats, dtype=np.float32)),
        'adj_mats': torch.tensor(np.array(adj_mats, dtype=np.float32)),
        'ring_indices': torch.tensor(np.array(ring_indices, dtype=np.int64)),
    }


def train_model(data, y, tr_idx, va_idx, use_proj, device, task_name):
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    model = build_model(use_projection=use_proj, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=PARAMS['hidden_dim'],
                        n_conv=PARAMS['n_conv_layers'],
                        n_hidden=PARAMS['n_hidden_layers'],
                        p_dropout=PARAMS['p_dropout'],
                        ring_flag_value=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=PARAMS['learning_rate'],
                                 weight_decay=PARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()
    bs = PARAMS['batch_size']

    node_mats = data['node_mats'].to(device)
    adj_mats = data['adj_mats'].to(device)
    ring_indices = data['ring_indices'].to(device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    tr_t = torch.tensor(np.asarray(tr_idx), dtype=torch.long, device=device)
    va_t = torch.tensor(np.asarray(va_idx), dtype=torch.long, device=device)

    best_vmae = float('inf')
    best_state = None
    best_epoch = 0
    bad = 0

    def forward_one(idx):
        return model(node_mats[idx], adj_mats[idx], ring_indices[idx]).reshape(-1)

    for ep in range(1, PARAMS['n_epochs'] + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        for i in range(0, len(tr_t), bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = forward_one(bi)
            loss = loss_fn(pred, yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for i in range(0, len(va_t), bs):
                bi = va_t[i:i + bs]
                preds.append(forward_one(bi))
                truths.append(yt[bi])
        pv = torch.cat(preds); tv = torch.cat(truths)
        vmae = (pv - tv).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_vmae - 1e-6:
            best_vmae = vmae
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PARAMS['patience']:
                break
        if ep % 20 == 0:
            print(f"    [{task_name}] ep={ep} val_mae={vmae:.4f} best={best_vmae:.4f}@{best_epoch}")

    model.load_state_dict(best_state)
    model.eval()

    # 报告 val 指标
    preds, truths = [], []
    with torch.no_grad():
        for i in range(0, len(va_t), bs):
            bi = va_t[i:i + bs]
            preds.append(forward_one(bi).cpu())
            truths.append(yt[bi].cpu())
    pv = torch.cat(preds).numpy()
    tv = torch.cat(truths).numpy()
    r2 = r2_score(tv, pv)
    mae = mean_absolute_error(tv, pv)
    rmse = float(np.sqrt(np.mean((tv - pv) ** 2)))

    return model, {'r2': float(r2), 'mae': float(mae), 'rmse': rmse,
                   'best_epoch': int(best_epoch), 'val_n': len(va_t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[train_final] device={device} seed={MODEL_SEED} split_seed={SPLIT_SEED}')

    metrics = {}
    for task in TASKS:
        print(f"\n=== {task['name']} (proj={task['proj']}) ===")
        smiles, aor, y, _ = load_data(task)
        groups = make_group_ids(smiles)
        n = len(smiles)

        # 80/20 holdout (seed=2026, group-aware)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
        dev_idx, test_idx = next(gss.split(np.arange(n), groups=groups))
        dev_idx = np.asarray(dev_idx); test_idx = np.asarray(test_idx)
        assert len(set(groups[dev_idx]) & set(groups[test_idx])) == 0, "holdout 泄漏!"

        # dev 内 87.5/12.5 final split
        tr_idx, va_idx = final_train_val_split(len(dev_idx), groups[dev_idx])
        tr_idx = dev_idx[tr_idx]; va_idx = dev_idx[va_idx]
        assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0, "final split 泄漏!"
        print(f"  n={n} dev={len(dev_idx)} test={len(test_idx)} "
              f"final_train={len(tr_idx)} final_val={len(va_idx)}")

        data = load_all_graphs(smiles, aor, ring_flag=1)
        model, m = train_model(data, y, tr_idx, va_idx, task['proj'], device, task['name'])
        print(f"  [DONE] val_r2={m['r2']:.4f} val_mae={m['mae']:.4f} "
              f"val_rmse={m['rmse']:.4f} best_epoch={m['best_epoch']}")

        out_path = os.path.join(PKG_DIR, f"{task['name'].lower()}_best.pt")
        torch.save({
            'state_dict': model.state_dict(),
            'config': {
                'use_projection': task['proj'], 'node_vec_len': NODE_VEC_LEN,
                'hidden_dim': PARAMS['hidden_dim'], 'n_conv': PARAMS['n_conv_layers'],
                'n_hidden': PARAMS['n_hidden_layers'], 'p_dropout': PARAMS['p_dropout'],
                'ring_flag_value': 1,
            },
            'metrics': m,
            'seed': MODEL_SEED, 'split_seed': SPLIT_SEED,
        }, out_path)
        print(f"  [saved] {out_path}")

        metrics[task['name']] = {
            'val_r2': m['r2'], 'val_mae': m['mae'], 'val_rmse': m['rmse'],
            'best_epoch': m['best_epoch'],
            'n_total': n, 'n_dev': len(dev_idx), 'n_test': len(test_idx),
            'n_final_train': len(tr_idx), 'n_final_val': len(va_idx),
            'use_projection': task['proj'], 'ring_flag_value': 1,
            'model_file': os.path.basename(f"{task['name'].lower()}_best.pt"),
        }
        del data, model

    metrics_path = os.path.join(PKG_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[metrics] {metrics_path}")


if __name__ == '__main__':
    main()
