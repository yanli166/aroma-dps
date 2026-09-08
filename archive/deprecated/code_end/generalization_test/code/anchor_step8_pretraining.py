
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
Anchor-based Δ-Learning Step 8: Pretraining 对照

在 anchor-based dataset 上比较:
  A. random-init Siamese
  B. Stage-I-pretrained Siamese

只测试: 100%, 20%, 10% training data
使用 F anchor (主 anchor), 所有 3 个 task
5-fold GroupKFold + 5 seeds
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'
STAGE_I_WEIGHTS = '_PROJ_ROOT + "/unified_models"/0427_unified_results/mpnn_label/best_model.pth'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import set_full_seed
from unified_models.mpnn.model import MPNNModel
from generalization_test.code.step5_siamese_mpnn import SiameseMPNN
from generalization_test.code.anchor_step3_run_models import (
    load_molecules_for_graph, train_siamese_anchor, predict_siamese_anchor,
    compute_metrics, compute_macro_metrics, make_splits,
    NVL, MAX_ATOMS, RING_FLAG_VALUE, SEEDS, N_FOLDS
)

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_anchor_delta_final')
PAIR_FILE = os.path.join(OUTPUT_DIR, 'anchor_pair_dataset.csv')

ANCHOR = 'F'
FRACTIONS = [1.0, 0.2, 0.1]
MODES = ['random', 'pretrained']

ENCODER_PREFIXES = (
    'init_transform.', 'conv_layers.', 'conv_bn_',
    'hidden_layers.', 'hidden_bns.',
)


def build_siamese(mode='random', seed=42):
    set_full_seed(seed)
    model = SiameseMPNN(
        node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2,
        p_dropout=0.2, mode='label'
    )
    if mode == 'pretrained':
        state_dict = torch.load(STAGE_I_WEIGHTS, map_location='cpu', weights_only=False)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith(ENCODER_PREFIXES):
                encoder_state[f'encoder.{k}'] = v
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"    [pretrained] Loaded {len(encoder_state)} encoder tensors")
    return model


def subsample_train_idx(train_idx, fraction, seed):
    if fraction >= 1.0:
        return train_idx
    n_total = len(train_idx)
    n_keep = max(2, int(fraction * n_total))
    rng = np.random.RandomState(seed + 7919)
    perm = rng.permutation(n_total)[:n_keep]
    return train_idx[perm]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--anchor', type=str, default=ANCHOR)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Anchor: {args.anchor}")
    print(f"Fractions: {FRACTIONS}")
    print(f"Modes: {MODES}")
    print(f"Stage I weights: {STAGE_I_WEIGHTS}")

    df_all = pd.read_csv(PAIR_FILE)
    task_map = {t['name']: t for t in TASKS}

    all_results = []

    for task_name in ['HOMA', 'NICS_1zz', 'MBCO']:
        task = task_map[task_name]
        target_col = task['target_col']

        print(f"\n{'='*70}")
        print(f"Task={task_name} | Anchor={args.anchor}")
        print(f"{'='*70}")

        df_pairs = df_all[(df_all['task'] == task_name) & (df_all['anchor'] == args.anchor)].copy().reset_index(drop=True)
        print(f"  Pairs: {len(df_pairs)} | Ring types: {df_pairs['scaffold_id'].nunique()}")

        # 加载图数据
        ref_data = load_molecules_for_graph(df_pairs, 'ref', target_col, device)
        sub_data = load_molecules_for_graph(df_pairs, 'sub', target_col, device)
        print(f"  Loaded graphs: {ref_data['n']}")

        groups = df_pairs['scaffold_id'].values
        y_delta = df_pairs['delta_A'].values.astype(np.float32)

        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")
            set_full_seed(seed)
            splits = make_splits(groups, N_FOLDS, seed)

            for fold_id, (train_idx_full, val_idx, test_idx) in enumerate(splits):
                print(f"    Fold {fold_id+1}/{N_FOLDS}: train_full={len(train_idx_full)}, val={len(val_idx)}, test={len(test_idx)}")

                for fraction in FRACTIONS:
                    train_idx = subsample_train_idx(train_idx_full, fraction, seed)
                    print(f"      Fraction={fraction:.0%} | Train: {len(train_idx)}")

                    for mode in MODES:
                        print(f"        Mode={mode}", end=' ', flush=True)
                        t0 = time.time()

                        set_full_seed(seed)
                        model = build_siamese(mode=mode, seed=seed)

                        model = train_siamese_anchor(
                            model, ref_data, sub_data, y_delta,
                            train_idx, val_idx, device,
                            n_epochs=200, patience=30, seed=seed
                        )
                        pred_delta = predict_siamese_anchor(model, ref_data, sub_data, test_idx, device)

                        df_pred = df_pairs.iloc[test_idx].copy().reset_index(drop=True)
                        df_pred['pred_delta'] = pred_delta

                        micro = compute_metrics(df_pred['delta_A'].values, df_pred['pred_delta'].values)
                        macro, _ = compute_macro_metrics(df_pred)

                        elapsed = time.time() - t0
                        print(f"micro: R²={micro['r2']:.4f} MAE={micro['mae']:.4f} "
                              f"Sp={micro['spearman']:.4f} Pw={micro['pairwise_acc']:.4f} "
                              f"| macro: R²={macro['r2']:.4f} MAE={macro['mae']:.4f} "
                              f"Sp={macro['spearman']:.4f} ({elapsed:.0f}s)")

                        res = {
                            'task': task_name, 'anchor': args.anchor,
                            'mode': mode, 'fraction': fraction,
                            'seed': seed, 'fold': fold_id,
                            'n_train': len(train_idx), 'n_test': len(test_idx),
                            'elapsed_sec': elapsed,
                        }
                        for k, v in micro.items():
                            res[f'micro_{k}'] = v
                        for k, v in macro.items():
                            res[f'macro_{k}'] = v
                        all_results.append(res)

                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        del ref_data, sub_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'pretraining_transfer_raw.csv'), index=False)

    # 聚合
    metric_cols = [c for c in df_results.columns if c.startswith('micro_') or c.startswith('macro_')]
    agg_rows = []
    for (task, fraction, mode), grp in df_results.groupby(['task', 'fraction', 'mode']):
        row = {'task': task, 'fraction': fraction, 'mode': mode, 'n_runs': len(grp)}
        for c in metric_cols:
            vals = grp[c].dropna()
            row[f'{c}_mean'] = vals.mean() if len(vals) > 0 else float('nan')
            row[f'{c}_std'] = vals.std(ddof=1) if len(vals) > 1 else float('nan')
        agg_rows.append(row)
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, 'pretraining_transfer.csv'), index=False)

    # 计算 Δ (pretrained - random)
    pivot = df_agg.pivot_table(
        index=['task', 'fraction'],
        columns='mode',
        values=['micro_r2_mean', 'micro_mae_mean', 'micro_spearman_mean']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()
    pivot['delta_r2'] = pivot['micro_r2_mean_pretrained'] - pivot['micro_r2_mean_random']
    pivot['delta_mae'] = pivot['micro_mae_mean_pretrained'] - pivot['micro_mae_mean_random']
    pivot['delta_spearman'] = pivot['micro_spearman_mean_pretrained'] - pivot['micro_spearman_mean_random']
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'pretraining_delta.csv'), index=False)

    print(f"\n{'='*70}")
    print("Pretraining Δ (pretrained - random)")
    print(f"{'='*70}")
    print(pivot[['task', 'fraction', 'delta_r2', 'delta_mae', 'delta_spearman']].to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
