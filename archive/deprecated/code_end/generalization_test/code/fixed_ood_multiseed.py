
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
Experiment B (multi-seed): Fixed OOD Benchmark with 5 ring-type sampling seeds

对每个 fraction, 用不同的 seed 采样 ring types, 报告 mean±SD。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.constants import ORIG_MODELS_ROOT
from common.tasks import TASKS, DEFAULT_SEED, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, train_model, eval_model, set_full_seed
from generalization_test.code.ring_utils import identify_ring_type
from generalization_test.code.run_learning_curve_rerun import (
    L10_CSV, TMP_DIR, NVL, MAX_ATOMS, EXT_MAX_ATOMS, RING_FLAG_VALUE,
    BEST_MODELS, TASK_MAP, TARGET_MAP, L10_COL_MAP,
    prepare_combined_csv, save_temp_csv, save_l10_test_csv
)

RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/fixed_ood_multiseed'
os.makedirs(RESULTS_DIR, exist_ok=True)

FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
N_FIXED_TEST_RTS = 20
SAMPLING_SEEDS = [42, 123, 456, 789, 2024]


def prepare_fixed_test_split(task_name, sampling_seed, n_test_rts=N_FIXED_TEST_RTS):
    """固定 test set: 用指定 seed 随机选 20 个 ring types 作为 OOD test"""
    df_l10 = pd.read_csv(L10_CSV)
    target_col_l10 = L10_COL_MAP[task_name]
    df_l10 = df_l10.dropna(subset=[target_col_l10]).reset_index(drop=True)

    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}

    rt_to_smiles = {}
    for smi, rt in smi_to_rt.items():
        rt_to_smiles.setdefault(rt, []).append(smi)

    rt_list = sorted(rt_to_smiles.keys())
    rng = np.random.RandomState(sampling_seed)
    rng.shuffle(rt_list)

    test_rts = set(rt_list[-n_test_rts:])
    pool_rts = set(rt_list[:-n_test_rts])

    test_smiles = set()
    for rt in test_rts:
        test_smiles.update(rt_to_smiles[rt])
    df_l10_test = df_l10[df_l10['SMILES'].isin(test_smiles)].reset_index(drop=True)

    return df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10


def run_single(task_name, model_name, fraction, sampling_seed, training_seed, device, params,
               df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10):
    """运行单次实验"""
    # 从 pool 采样
    if fraction == 0.0:
        df_l10_train = pd.DataFrame()
        train_smiles = set()
    else:
        pool_list = sorted(pool_rts)
        rng = np.random.RandomState(sampling_seed)
        rng.shuffle(pool_list)
        n_select = int(len(pool_list) * fraction)
        selected_rts = set(pool_list[:n_select])
        train_smiles = set()
        for rt in selected_rts:
            train_smiles.update(rt_to_smiles[rt])
        df_l10_train = df_l10[df_l10['SMILES'].isin(train_smiles)].reset_index(drop=True)

    combined_df = prepare_combined_csv(task_name, fraction, df_l10_train)
    combined_csv = save_temp_csv(combined_df, f'fixed_ms_{sampling_seed}', task_name, fraction)
    test_csv = save_l10_test_csv(df_l10_test, task_name, f'fixed_ms_{sampling_seed}_{fraction}')

    target_col = TARGET_MAP[task_name]
    train_data = load_adj_format(combined_csv, target_col, NVL, MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_data = load_adj_format(test_csv, target_col, NVL, EXT_MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    n = train_data['n']
    rng = np.random.RandomState(training_seed)
    perm = rng.permutation(n)
    n_val = max(int(0.125 * n), 2)
    val_idx = torch.tensor(perm[:n_val], device=device)
    train_idx = torch.tensor(perm[n_val:], device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    model = train_model(model_name, params, train_data, train_idx, val_idx,
                       device, params['n_epochs'], params['patience'], training_seed)

    tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, train_data, train_idx,
                                              params['batch_size'], device)
    te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, test_data, test_idx,
                                                           params['batch_size'], device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'task': task_name, 'fraction': fraction,
        'sampling_seed': sampling_seed, 'training_seed': training_seed,
        'test_r2': te_r2, 'test_mae': te_mae,
        'n_test_smiles': len(test_smiles),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA')
    parser.add_argument('--sampling_seeds', type=str, default='42,123,456,789,2024')
    parser.add_argument('--training_seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS)

    tasks = args.tasks.split(',')
    sampling_seeds = [int(s) for s in args.sampling_seeds.split(',')]

    all_results = []
    for task_name in tasks:
        model_name = BEST_MODELS[task_name]
        print(f"\n{'='*60}")
        print(f"# {task_name} / {model_name} ({len(sampling_seeds)} sampling seeds)")
        print(f"{'='*60}")

        for s_seed in sampling_seeds:
            print(f"\n  --- sampling_seed={s_seed} ---")
            df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10 = \
                prepare_fixed_test_split(task_name, s_seed)
            print(f"    test: {len(test_smiles)} SMILES, {len(df_l10_test)} samples")

            for frac in FRACTIONS:
                try:
                    result = run_single(task_name, model_name, frac, s_seed,
                                       args.training_seed, device, params,
                                       df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10)
                    all_results.append(result)
                    print(f"    frac={frac}: R2={result['test_r2']:.4f}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"    [失败] frac={frac}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, 'fixed_ood_multiseed.csv')
        df.to_csv(csv_path, index=False)

        # 聚合
        agg = df.groupby(['task', 'fraction']).agg(
            test_r2_mean=('test_r2', 'mean'),
            test_r2_std=('test_r2', 'std'),
            test_mae_mean=('test_mae', 'mean'),
            test_mae_std=('test_mae', 'std'),
            n_seeds=('sampling_seed', 'count'),
        ).reset_index()
        agg.to_csv(os.path.join(RESULTS_DIR, 'fixed_ood_multiseed_agg.csv'), index=False)

        print(f"\n保存: {csv_path}")
        print(f"聚合: {os.path.join(RESULTS_DIR, 'fixed_ood_multiseed_agg.csv')}")
        print("\n" + agg.to_string(index=False))


if __name__ == '__main__':
    main()
