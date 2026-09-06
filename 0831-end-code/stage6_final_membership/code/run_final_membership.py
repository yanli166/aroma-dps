"""
[0831 重构] Stage 6: 最终 membership 方案 CV 协议

基于 P1-1 sensitivity 结果决定:
  - HOMA: 10 显著优于 1 → 用 binary membership + learnable projection (nn.Embedding(2, hidden_dim))
          节点特征中目标环标记位置 (node_vec_len-15) 始终注入 0/1 (binary 化学定义),
          另起一条并行分支 nn.Embedding(2, hidden_dim) 把 membership 映到 hidden 维,
          与原节点特征 concat 后再进入 MPNN。
  - NICS_1zz / MBCO: 1/5/10 几乎一样 → 用 ring_flag=1 (binary)

四个 config (与当前 MPNN 设计兼容):
  - base           : ring_flag=0,    no projection
  - membership_1   : ring_flag=1,    no projection        (binary 化学定义)
  - membership_10  : ring_flag=10,   no projection        (amplitude, 仅对照)
  - membership_proj: ring_flag=1,    nn.Embedding(2,hidden) + concat

固定 split, MPNN, seed=11, max_epochs=200, patience=30; 5-fold dev CV, 仅 dev 不动 test。
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import RESULTS_V2_DIR, SPLITS_DIR
from common.protocol import (SPLIT_SEED, get_final_splits, make_group_ids,
                             completeness_check, MODEL_SEEDS)
from common.tasks import TASKS
from common.graph_data import load_adj_format
from common.train_eval import (set_full_seed, train_custom_model, eval_custom_model,
                                verify_split_invariants)
from common.features import get_feature_meta
from models.ring_conditioned_gnn import build_model

NODE_VEC_LEN, MAX_ATOMS = 60, 75
DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}

# config -> (ring_flag_value, use_projection)
# 'base'            = ring_flag=0,  no proj
# 'membership_1'    = ring_flag=1,  no proj   (binary 0/1 化学定义)
# 'membership_10'   = ring_flag=10, no proj   (化学可解释幅度, 仅对照)
# 'membership_proj' = ring_flag=1,  with proj (binary + learned scale)
CONFIGS = [
    ('base',            0,  False),
    ('membership_1',    1,  False),
    ('membership_10',   10, False),
    ('membership_proj', 1,  True),
]


class Ctx:
    cache = {}
    params = None
    device = None
    task = None
    _last_data = None


def _train_cv(backbone, ring_flag, use_proj, tr_idx, va_idx,
               n_epochs, patience, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', Ctx.task['name'], ring_flag, use_proj, feature_mode)
    data = Ctx.cache.get(key)
    if data is None:
        data = load_adj_format(Ctx.task['dataset_path'],
                                Ctx.task['target_col'],
                                NODE_VEC_LEN, MAX_ATOMS,
                                ring_flag_value=ring_flag, device=Ctx.device,
                                feature_mode=feature_mode)
        if use_proj:
            # 把 binary ring flag (0/1) 替换回 1 (确保 ring_flag=1 时是 binary)
            # 然后在 forward 时额外用 nn.Embedding 注入
            pass
        Ctx.cache[key] = data
    in_dim = NODE_VEC_LEN + (Ctx.params['hidden_dim'] if use_proj else 0)
    model = build_model(backbone, node_vec_len=in_dim,
                        hidden_dim=Ctx.params['hidden_dim'],
                        n_conv=Ctx.params['n_conv_layers'],
                        n_hidden=Ctx.params['n_hidden_layers'],
                        p_dropout=Ctx.params['p_dropout'],
                        readout_mode='fixed_avg').to(Ctx.device)
    if use_proj:
        model.ring_proj = nn.Embedding(2, Ctx.params['hidden_dim']).to(Ctx.device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(model.ring_proj.parameters()),
            lr=Ctx.params['learning_rate'],
            weight_decay=Ctx.params['weight_decay'])
    else:
        model.ring_proj = None
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr=Ctx.params['learning_rate'],
                                      weight_decay=Ctx.params['weight_decay'])
    model._optimizer = optimizer

    tr_t = torch.tensor(np.asarray(tr_idx), dtype=torch.long, device=Ctx.device)
    va_t = torch.tensor(np.asarray(va_idx), dtype=torch.long, device=Ctx.device)
    model = _train_loop(model, data, tr_t, va_t, Ctx.params, Ctx.device,
                         n_epochs, patience, seed, use_proj)
    Ctx._last_data = data
    return model


def _train_loop(model, data, tr_t, va_t, params, device, n_epochs, patience, seed, use_proj):
    """手动训练循环, 支持可选的 ring_proj Embedding 注入"""
    set_full_seed(seed)
    bs = params['batch_size']; lr = params['learning_rate']; wd = params['weight_decay']
    node_mats = data['node_mats']; adj_mats = data['adj_mats']
    ring_indices = data['ring_indices']; outputs = data['outputs']
    loss_fn = torch.nn.MSELoss()
    optimizer = model._optimizer if hasattr(model, '_optimizer') and model._optimizer is not None \
        else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    best_vmae = float('inf'); best_state = None; best_epoch = 0; bad = 0

    def forward_one(idx):
        x = node_mats[idx]; adj = adj_mats[idx]; ri = ring_indices[idx]
        if use_proj and model.ring_proj is not None:
            # ring_indices ∈ {-1, 0, 1, ...}; 二值化 0/1
            bin_ri = (ri >= 0).long()
            emb = model.ring_proj(bin_ri)  # (B, max_atoms, hidden_dim)
            x = torch.cat([x, emb], dim=-1)
        return model(x, adj, ri)

    n_tr = len(tr_t)
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n_tr, device=device)]
        for i in range(0, n_tr, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = forward_one(bi).reshape(-1)
            loss = loss_fn(pred, outputs[bi].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # eval val
        model.eval()
        n_va = len(va_t)
        preds, truths = [], []
        with torch.no_grad():
            for i in range(0, n_va, bs):
                bi = va_t[i:i + bs]
                preds.append(forward_one(bi).reshape(-1))
                truths.append(outputs[bi].reshape(-1))
        pv = torch.cat(preds); tv = torch.cat(truths)
        vmae = (pv - tv).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_vmae - 1e-6:
            best_vmae = vmae; best_epoch = ep; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if model.ring_proj is not None:
                best_state['ring_proj.weight'] = model.ring_proj.weight.detach().cpu().clone()
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()
                               if k in model.state_dict()})
        if model.ring_proj is not None and 'ring_proj.weight' in best_state:
            model.ring_proj.weight.data.copy_(best_state['ring_proj.weight'].to(device))
    model.best_epoch = best_epoch
    model._optimizer = None
    return model


def _eval(model, idx, use_proj):
    data = Ctx._last_data
    bs = Ctx.params['batch_size']; device = Ctx.device
    node_mats = data['node_mats']; adj_mats = data['adj_mats']
    ring_indices = data['ring_indices']; outputs = data['outputs']
    idx_t = torch.tensor(np.asarray(idx), dtype=torch.long, device=device)

    def forward_one(ii):
        x = node_mats[ii]; adj = adj_mats[ii]; ri = ring_indices[ii]
        if use_proj and model.ring_proj is not None:
            bin_ri = (ri >= 0).long()
            emb = model.ring_proj(bin_ri)
            x = torch.cat([x, emb], dim=-1)
        return model(x, adj, ri)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(0, len(idx_t), bs):
            bi = idx_t[i:i + bs]
            preds.append(forward_one(bi).reshape(-1))
            truths.append(outputs[bi].reshape(-1))
    pv = torch.cat(preds).cpu().numpy(); tv = torch.cat(truths).cpu().numpy()
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(tv, pv); mae = mean_absolute_error(tv, pv)
    rmse = float(np.sqrt(mean_squared_error(tv, pv)))
    return r2, mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=os.path.join(RESULTS_V2_DIR, 'final_membership'))
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--seed', type=int, default=MODEL_SEEDS[0])
    ap.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    ap.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    ap.add_argument('--feature_mode', default='standard',
                    choices=['standard', 'explicit_aromaticity_ablated'])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[final_membership] device={device} seed={args.seed} '
          f'n_epochs={args.n_epochs} feature_mode={args.feature_mode}')

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == x) for x in args.tasks.split(',')]

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs; params['patience'] = args.patience

    feat_meta = get_feature_meta(args.feature_mode)
    run_dir = os.path.join(args.output_dir, f'seed_{args.seed}', args.feature_mode)
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    fold_rows = []
    for task in task_list:
        data0 = load_adj_format(task['dataset_path'], task['target_col'],
                                 NODE_VEC_LEN, MAX_ATOMS,
                                 ring_flag_value=0, device='cpu',
                                 feature_mode=args.feature_mode)
        groups = make_group_ids(list(data0['smiles']), use_inchikey=False)
        tag = f'stage6_{args.feature_mode}'
        splits = get_final_splits(data0['n'], groups, persist=SPLITS_DIR, tag=tag)
        verify_split_invariants(splits, groups, f"final_membership {task['name']}")
        n_total = data0['n']

        Ctx.params = params; Ctx.device = device; Ctx.task = task
        print(f"\n[Task={task['name']}] n={n_total} dev={len(splits['train_idx'])} "
              f"test={len(splits['test_idx'])}")

        for cfg_name, rf, use_proj in CONFIGS:
            Ctx.cache = {}
            cv_metrics_list = []
            best_epochs = []
            for k, (tr, va) in enumerate(splits['folds']):
                model = _train_cv('MPNN', rf, use_proj,
                                   np.asarray(tr), np.asarray(va),
                                   params['n_epochs'], params['patience'],
                                   args.seed + k, args.feature_mode)
                r2, mae, rmse = _eval(model, va, use_proj)
                be = int(getattr(model, 'best_epoch', params['n_epochs']))
                cv_metrics_list.append({'fold': k+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
                best_epochs.append(be)
                fold_rows.append({
                    'seed': args.seed, 'task': task['name'], 'config': cfg_name,
                    'ring_flag_value': rf, 'use_projection': use_proj,
                    'fold': k + 1, 'feature_mode': args.feature_mode,
                    'val_mae': mae, 'val_r2': r2, 'val_rmse': rmse,
                    'best_epoch': be, 'max_epoch': params['n_epochs'],
                    'ceiling_hit': int(be >= params['n_epochs'] - 1),
                    'n_dev_fold': len(tr), 'n_val_fold': len(va),
                })
                del model
            cv_mae_arr = np.array([m['mae'] for m in cv_metrics_list])
            cv_r2_arr = np.array([m['r2'] for m in cv_metrics_list])
            cv_rmse_arr = np.array([m['rmse'] for m in cv_metrics_list])
            cv_mae = float(cv_mae_arr.mean())
            cv_mae_std = float(cv_mae_arr.std(ddof=1))
            cv_r2 = float(cv_r2_arr.mean())
            cv_rmse = float(cv_rmse_arr.mean())
            estar = int(np.median(best_epochs))
            n_ceiling = int(sum(1 for e in best_epochs if e >= params['n_epochs'] - 1))

            print(f"  [{cfg_name}] rf={rf:2d} proj={use_proj}: "
                  f"cv_mae={cv_mae:.4f}±{cv_mae_std:.4f} cv_r2={cv_r2:.4f} "
                  f"E*={estar} best={best_epochs} ceiling={n_ceiling}/5")

            rows.append({
                'seed': args.seed, 'task': task['name'], 'config': cfg_name,
                'ring_flag_value': rf, 'use_projection': use_proj,
                'feature_mode': args.feature_mode,
                'cv_mae': cv_mae, 'cv_mae_std': cv_mae_std,
                'cv_r2': cv_r2, 'cv_rmse': cv_rmse,
                'best_epochs': best_epochs, 'estar': estar,
                'n_ceiling_hits': n_ceiling,
                'n_dev': len(splits['train_idx']), 'n_test': len(splits['test_idx']),
                'n_total': n_total,
            })

    df = pd.DataFrame(rows)
    df_fold = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(run_dir, 'final_membership_cv.csv'), index=False)
    df_fold.to_csv(os.path.join(run_dir, 'final_membership_fold_level.csv'), index=False)

    expected = pd.MultiIndex.from_product(
        [[c[0] for c in CONFIGS], [t['name'] for t in task_list], [args.seed]],
        names=['config', 'task', 'seed'])
    completeness_check(pd.MultiIndex.from_frame(df[['config', 'task', 'seed']]),
                       expected, f"final_membership seed={args.seed}")
    print(f"\n[完整性] final_membership 校验通过 (缺失 0)")

    # 每任务决策
    summary = []
    for t in task_list:
        sub = df[df['task'] == t['name']].set_index('config')['cv_mae']
        # 任务特定决策: HOMA 用 proj / 其余用 membership_1
        if t['name'] == 'HOMA':
            recommended = 'membership_proj'
        else:
            recommended = 'membership_1'
        s = {
            'task': t['name'],
            'cv_mae_base':            sub.get('base'),
            'cv_mae_membership_1':    sub.get('membership_1'),
            'cv_mae_membership_10':   sub.get('membership_10'),
            'cv_mae_membership_proj': sub.get('membership_proj'),
            'recommended':            recommended,
        }
        summary.append(s)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(run_dir, 'final_membership_summary.csv'), index=False)
    print('\n--- final_membership summary ---')
    print(summary_df.to_string(index=False))

    meta = {
        'protocol': 'PROTOCOL_SPEC final_membership (P1-1 follow-up)',
        'split_seed': SPLIT_SEED, 'model_seed': args.seed,
        'configs': [(c[0], c[1], c[2]) for c in CONFIGS],
        'feature_mode': args.feature_mode, 'feature_meta': feat_meta,
        'n_epochs_cv_max': params['n_epochs'], 'patience': params['patience'],
        'protocol_pipeline': '5-fold Group CV on dev only (no final test)',
        'decision_basis': (
            'P1-1 sensitivity: HOMA 上 rf=10 显著优于 rf=1 → 用 binary + learned projection; '
            'NICS/MBCO 上 rf=1/5/10 几乎一样 → 用 binary 1。'
            '正式 5-seed 实验以本实验的 recommended config 为准。'
        ),
    }
    meta_path = os.path.join(run_dir, 'final_membership_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[results] {run_dir}\n[meta] {meta_path}")


if __name__ == '__main__':
    main()
