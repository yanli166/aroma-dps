
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
Experiment B: Fixed OOD Benchmark

固定 20 个 ring types 作为永远不进入训练的 OOD test set。
其余 47 个 ring types 作为 adaptation pool, 按不同 fraction 采样加入训练。

对比 Experiment A (progressive residual OOD):
  - Exp A: train coverage ↑, test = remaining unseen types (测试集随 fraction 变化)
  - Exp B: train coverage ↑, fixed test set (测试集始终是同样 20 种 ring types)

输出:
  - fraction vs R² (固定 test set)
  - 对比 Exp A 和 Exp B 的曲线
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
from generalization_test.code.train_eval import DEFAULT_PARAMS, train_model, eval_model
from generalization_test.code.ring_utils import identify_ring_type
from generalization_test.code.run_learning_curve_rerun import (
    L10_CSV, TMP_DIR, NVL, MAX_ATOMS, EXT_MAX_ATOMS, RING_FLAG_VALUE,
    BEST_MODELS, TASK_MAP, TARGET_MAP, L10_COL_MAP,
    prepare_combined_csv, save_temp_csv, save_l10_test_csv
)

RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/fixed_ood_benchmark'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = DEFAULT_SEED
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
N_FIXED_TEST_RTS = 20  # 固定 20 个 ring types 作为 test


def prepare_fixed_test_split(task_name, n_test_rts=N_FIXED_TEST_RTS, seed=SEED):
    """固定 test set: 随机选 20 个 ring types 永远作为 OOD test"""
    df_l10 = pd.read_csv(L10_CSV)
    target_col_l10 = L10_COL_MAP[task_name]
    df_l10 = df_l10.dropna(subset=[target_col_l10]).reset_index(drop=True)

    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}

    rt_to_smiles = {}
    for smi, rt in smi_to_rt.items():
        rt_to_smiles.setdefault(rt, []).append(smi)

    rt_list = sorted(rt_to_smiles.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(rt_list)

    # 固定 test: 最后 20 个 ring types (永远不进入训练)
    test_rts = set(rt_list[-n_test_rts:])
    pool_rts = set(rt_list[:-n_test_rts])  # 47 个 ring types 作为 adaptation pool

    # 固定 test set SMILES
    test_smiles = set()
    for rt in test_rts:
        test_smiles.update(rt_to_smiles[rt])
    df_l10_test = df_l10[df_l10['SMILES'].isin(test_smiles)].reset_index(drop=True)

    print(f"  固定 test: {len(test_rts)} ring types, {len(test_smiles)} SMILES, {len(df_l10_test)} samples")
    print(f"  Adaptation pool: {len(pool_rts)} ring types")

    return df_l10_test, test_smiles, pool_rts, rt_to_smiles


def prepare_adaptation_train(task_name, fraction, pool_rts, rt_to_smiles, df_l10):
    """从 pool 中按 fraction 采样 ring types 加入训练"""
    if fraction == 0.0:
        return pd.DataFrame(), set()

    pool_list = sorted(pool_rts)
    rng = np.random.RandomState(SEED)
    rng.shuffle(pool_list)

    n_select = int(len(pool_list) * fraction)
    selected_rts = set(pool_list[:n_select])

    train_smiles = set()
    for rt in selected_rts:
        train_smiles.update(rt_to_smiles[rt])

    df_l10_train = df_l10[df_l10['SMILES'].isin(train_smiles)].reset_index(drop=True)
    return df_l10_train, train_smiles


def run_single_fixed(task_name, model_name, fraction, device, params,
                     df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10):
    """运行固定 test set 的单次实验"""
    print(f"\n  Fraction={fraction*100:.0f}% (fixed test)")

    # 从 pool 中采样训练数据
    df_l10_train, train_smiles = prepare_adaptation_train(
        task_name, fraction, pool_rts, rt_to_smiles, df_l10)

    if len(df_l10_train) > 0:
        print(f"    adaptation: {len(train_smiles)} SMILES, {len(df_l10_train)} samples")
    else:
        print(f"    adaptation: 0 (baseline, no lunci10 in training)")

    # 合并主训练数据 + adaptation 数据
    combined_df = prepare_combined_csv(task_name, fraction, df_l10_train)
    combined_csv = save_temp_csv(combined_df, 'fixed_combined', task_name, fraction)
    test_csv = save_l10_test_csv(df_l10_test, task_name, fraction)

    target_col = TARGET_MAP[task_name]

    # 加载数据
    print(f"    Loading data...")
    train_data = load_adj_format(combined_csv, target_col, NVL, MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_data = load_adj_format(test_csv, target_col, NVL, EXT_MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    n = train_data['n']
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = max(int(0.125 * n), 2)
    val_idx = torch.tensor(perm[:n_val], device=device)
    train_idx = torch.tensor(perm[n_val:], device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    # 训练
    t0 = time.time()
    model = train_model(model_name, params, train_data, train_idx, val_idx,
                       device, params['n_epochs'], params['patience'], SEED)
    train_time = time.time() - t0

    # 评估
    tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, train_data, train_idx,
                                              params['batch_size'], device)
    te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, test_data, test_idx,
                                                           params['batch_size'], device)

    print(f"    Train R2={tr_r2:.4f} | Test R2={te_r2:.4f} MAE={te_mae:.4f} ({train_time:.0f}s)")

    # 保存结果
    frac_dir = os.path.join(RESULTS_DIR, task_name, f'frac_{int(fraction*100)}')
    os.makedirs(frac_dir, exist_ok=True)
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(frac_dir, 'predictions.csv'), index=False)

    import csv
    with open(os.path.join(frac_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['model', model_name])
        w.writerow(['task', task_name])
        w.writerow(['fraction', fraction])
        w.writerow(['n_test_rts', N_FIXED_TEST_RTS])
        w.writerow(['n_pool_rts', len(pool_rts)])
        w.writerow(['n_adaptation_rts', int(len(pool_rts) * fraction)])
        w.writerow(['n_train_smiles', len(train_smiles)])
        w.writerow(['n_test_smiles', len(test_smiles)])
        w.writerow(['train_r2', tr_r2])
        w.writerow(['test_r2', te_r2])
        w.writerow(['test_mae', te_mae])
        w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'task': task_name, 'fraction': fraction,
        'test_r2': te_r2, 'test_mae': te_mae,
        'n_adaptation_rts': int(len(pool_rts) * fraction),
        'n_test_rts': N_FIXED_TEST_RTS,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Experiment B: Fixed OOD Benchmark')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    parser.add_argument('--fractions', type=str, default='0.0,0.05,0.1,0.2,0.3,0.5,0.7')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment B: Fixed OOD Benchmark ({N_FIXED_TEST_RTS} ring types always as test)")

    params = dict(DEFAULT_PARAMS)
    tasks = args.tasks.split(',')
    fractions = [float(f) for f in args.fractions.split(',')]

    all_results = []
    for task_name in tasks:
        model_name = BEST_MODELS[task_name]
        print(f"\n{'='*60}")
        print(f"# {task_name} / {model_name} (Fixed Test)")
        print(f"{'='*60}")

        # 准备固定 test set
        df_l10_test, test_smiles, pool_rts, rt_to_smiles = prepare_fixed_test_split(task_name)
        df_l10 = pd.read_csv(L10_CSV).dropna(subset=[L10_COL_MAP[task_name]]).reset_index(drop=True)

        for frac in fractions:
            try:
                result = run_single_fixed(task_name, model_name, frac, device, params,
                                         df_l10_test, test_smiles, pool_rts, rt_to_smiles, df_l10)
                all_results.append(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [失败] {task_name} frac={frac}: {e}")

    # 保存汇总
    if all_results:
        df_out = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, 'fixed_ood_benchmark.csv')
        df_out.to_csv(csv_path, index=False)
        print(f"\n汇总保存: {csv_path}")
        print(df_out.to_string(index=False))


if __name__ == '__main__':
    main()
