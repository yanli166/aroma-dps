"""
[0831 重构] P1-1: target-ring membership amplitude sensitivity test

目的: 验证 ring conditioning 收益是否由"任意非零幅度"造成 (e.g. 1 vs 5 vs 10)。
     若 ring_flag=1/5/10 在充分收敛后性能相近, 正式模型优先使用 binary 0/1 membership。
     若 10 仍显著优于 1, 不把 10 作为最终化学定义; 改为 binary membership + learnable
     projection (nn.Embedding(2, hidden_dim))。

协议:
  - backbone: 固定 MPNN
  - seed:     固定 11 (pre-production, 仅 dev CV, 不动 test)
  - ring_flag: {0, 1, 5, 10}
  - readout:  fixed_avg (与 Stage2 membership 同一 readout)
  - feature_mode: standard
  - 数据集: HOMA / NICS_1zz / MBCO
  - max_epochs=200, patience=30
  - 输出: fold-level (best_epoch, val_mae) + mean±SD + ceiling-hit
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

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
from common.estar_pipeline import run_gnn_estar
from models.ring_conditioned_gnn import build_model

RING_FLAGS = [0, 1, 5, 10]         # P1-1 扩展: 4 档 amplitude
BACKBONE = 'MPNN'
NODE_VEC_LEN, MAX_ATOMS = 60, 75
DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}


class Ctx:
    cache = {}
    params = None
    device = None
    task = None
    _last_data = None


def _train_cv(backbone, ring_flag, tr_idx, va_idx, n_epochs, patience, seed, feature_mode):
    set_full_seed(seed)
    key = ('adj', Ctx.task['name'], ring_flag, feature_mode)
    data = Ctx.cache.get(key)
    if data is None:
        data = load_adj_format(Ctx.task['dataset_path'],
                                Ctx.task['target_col'],
                                NODE_VEC_LEN, MAX_ATOMS,
                                ring_flag_value=ring_flag, device=Ctx.device,
                                feature_mode=feature_mode)
        Ctx.cache[key] = data
    model = build_model(backbone, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=Ctx.params['hidden_dim'],
                        n_conv=Ctx.params['n_conv_layers'],
                        n_hidden=Ctx.params['n_hidden_layers'],
                        p_dropout=Ctx.params['p_dropout'],
                        readout_mode='fixed_avg').to(Ctx.device)
    tr_t = torch.tensor(np.asarray(tr_idx), dtype=torch.long, device=Ctx.device)
    va_t = torch.tensor(np.asarray(va_idx), dtype=torch.long, device=Ctx.device)
    model = train_custom_model(model, Ctx.params, data, tr_t, va_t, Ctx.device,
                                n_epochs=n_epochs, patience=patience, seed=seed)
    Ctx._last_data = data
    return model


def _eval(model, idx):
    data = Ctx._last_data
    return eval_custom_model(model, data,
                              torch.tensor(np.asarray(idx), dtype=torch.long,
                                            device=Ctx.device),
                              Ctx.params['batch_size'], Ctx.device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=os.path.join(RESULTS_V2_DIR, 'ring_flag_sensitivity'))
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--seed', type=int, default=MODEL_SEEDS[0])
    ap.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    ap.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    ap.add_argument('--feature_mode', default='standard',
                    choices=['standard', 'explicit_aromaticity_ablated'])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[ring_flag_sens] device={device} seed={args.seed} '
          f'n_epochs={args.n_epochs} feature_mode={args.feature_mode}')

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == x) for x in args.tasks.split(',')]

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    feat_meta = get_feature_meta(args.feature_mode)

    run_dir = os.path.join(args.output_dir, f'seed_{args.seed}', args.feature_mode)
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    fold_rows = []  # fold-level
    for task in task_list:
        # 固定 split
        data0 = load_adj_format(task['dataset_path'], task['target_col'],
                                 NODE_VEC_LEN, MAX_ATOMS,
                                 ring_flag_value=0, device='cpu',
                                 feature_mode=args.feature_mode)
        groups = make_group_ids(list(data0['smiles']), use_inchikey=False)
        tag = f'stage5_{args.feature_mode}'
        splits = get_final_splits(data0['n'], groups, persist=SPLITS_DIR, tag=tag)
        verify_split_invariants(splits, groups, f"ring_flag_sens {task['name']}")
        n_total = data0['n']

        Ctx.params = params; Ctx.device = device; Ctx.task = task
        print(f"\n[Task={task['name']}] n={n_total} dev={len(splits['train_idx'])} "
              f"test={len(splits['test_idx'])}")

        for rf in RING_FLAGS:
            Ctx.cache = {}
            # 用 E*=median 流水线但只取 CV 部分 (不走 final)
            folds = splits['folds']
            cv_metrics_list = []
            best_epochs = []
            for k, (tr, va) in enumerate(folds):
                model = _train_cv(BACKBONE, rf, np.asarray(tr), np.asarray(va),
                                   params['n_epochs'], params['patience'],
                                   args.seed + k, args.feature_mode)
                r2, mae, rmse = _eval(model, va)
                be = int(getattr(model, 'best_epoch', params['n_epochs']))
                cv_metrics_list.append({'fold': k+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
                best_epochs.append(be)
                fold_rows.append({
                    'seed': args.seed, 'task': task['name'], 'backbone': BACKBONE,
                    'ring_flag': rf, 'fold': k + 1,
                    'feature_mode': args.feature_mode,
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
            cv_mae_std = float(cv_mae_arr.std(ddof=1)) if len(cv_mae_arr) > 1 else 0.0
            cv_r2 = float(cv_r2_arr.mean())
            cv_rmse = float(cv_rmse_arr.mean())
            estar = int(np.median(best_epochs))
            n_ceiling = int(sum(1 for e in best_epochs if e >= params['n_epochs'] - 1))

            print(f"  ring_flag={rf:2d}: cv_mae={cv_mae:.4f}±{cv_mae_std:.4f} "
                  f"cv_r2={cv_r2:.4f} E*={estar} best={best_epochs} ceiling={n_ceiling}/5")

            rows.append({
                'seed': args.seed, 'task': task['name'], 'backbone': BACKBONE,
                'ring_flag': rf, 'feature_mode': args.feature_mode,
                'cv_mae': cv_mae, 'cv_mae_std': cv_mae_std,
                'cv_r2': cv_r2, 'cv_rmse': cv_rmse,
                'best_epochs': best_epochs, 'estar': estar,
                'n_ceiling_hits': n_ceiling, 'n_dev': len(splits['train_idx']),
                'n_test': len(splits['test_idx']), 'n_total': n_total,
            })

    df = pd.DataFrame(rows)
    df_fold = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(run_dir, 'sensitivity_cv.csv'), index=False)
    df_fold.to_csv(os.path.join(run_dir, 'sensitivity_fold_level.csv'), index=False)

    # 完整性
    expected = pd.MultiIndex.from_product([RING_FLAGS, [t['name'] for t in task_list], [args.seed]],
                                          names=['ring_flag', 'task', 'seed'])
    completeness_check(pd.MultiIndex.from_frame(df[['ring_flag', 'task', 'seed']]),
                       expected, f"ring_flag_sensitivity seed={args.seed}")
    print(f"\n[完整性] ring_flag sensitivity 校验通过 (缺失 0)")

    # 结论: 1 vs 10 / 5 vs 10 / 1 vs 5 是否接近?
    summary = []
    for t in task_list:
        sub = df[df['task'] == t['name']].set_index('ring_flag')
        s = {}
        for k in [0, 1, 5, 10]:
            if k in sub.index:
                s[k] = sub.loc[k, 'cv_mae']
        delta_1_vs_10 = abs(s.get(1, 0) - s.get(10, 0))
        delta_5_vs_10 = abs(s.get(5, 0) - s.get(10, 0))
        delta_1_vs_0 = abs(s.get(1, 0) - s.get(0, 0))
        summary.append({'task': t['name'],
                        'cv_mae_rf0':  s.get(0),
                        'cv_mae_rf1':  s.get(1),
                        'cv_mae_rf5':  s.get(5),
                        'cv_mae_rf10': s.get(10),
                        '|1-10|': round(delta_1_vs_10, 4),
                        '|5-10|': round(delta_5_vs_10, 4),
                        '|1-0|':  round(delta_1_vs_0, 4)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(run_dir, 'sensitivity_summary.csv'), index=False)
    print("\n--- sensitivity summary (4 scales) ---")
    print(summary_df.to_string(index=False))

    meta = {
        'protocol': 'PROTOCOL_SPEC ring_flag_sensitivity (P1-1 expanded)',
        'split_seed': SPLIT_SEED, 'model_seed': args.seed,
        'ring_flags': RING_FLAGS, 'backbone': BACKBONE,
        'feature_mode': args.feature_mode, 'feature_meta': feat_meta,
        'n_epochs_cv_max': params['n_epochs'], 'patience': params['patience'],
        'protocol_pipeline': '5-fold Group CV on dev only (no final test)',
        'decision_rule': (
            '若 max( |1-10|, |5-10| ) <= 0.5 * |1-0|, '
            '则判定 binary 1 与 amplitude 10 等价, 正式模型优先 binary 0/1 membership (ring_flag=1)。'
            '若 10 仍显著优于 1, 不把 10 作为最终化学定义; '
            '改为 binary membership + learnable projection (nn.Embedding(2, hidden_dim)), '
            '使 membership 保持 0/1, 由网络学习其尺度。'
        ),
    }
    meta_path = os.path.join(run_dir, 'sensitivity_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[results] {run_dir}\n[meta] {meta_path}")


if __name__ == '__main__':
    main()
