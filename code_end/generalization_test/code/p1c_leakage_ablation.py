
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
P1c: Ring Membership / Aromatic Flag Leakage Ablation

四组消融:
  1. no_ring_info:      置零 membership + aromatic flag
  2. membership_only:   保留 membership, 置零 aromatic flag
  3. aromatic_only:     置零 membership, 保留 aromatic flag
  4. membership+aromatic: 原始 (两者都保留)

实现: 通过修改加载后的 node_mats 来精确控制特征
  - node_mat[:, NVL-5]: aromatic flag
  - node_mat[:, NVL-15]: target ring membership
"""
import os
import sys
import time
import copy
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

RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/p1c_leakage_ablation'
os.makedirs(RESULTS_DIR, exist_ok=True)

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10
SEED = DEFAULT_SEED

AROMATIC_IDX = -5   # node_mat[:, NVL-5] = aromatic flag
MEMBERSHIP_IDX = -15  # node_mat[:, NVL-15] = membership * ring_flag_value


def ablate_features(data, mode):
    """修改 node_mats 实现消融

    mode:
      - 'no_ring_info':      membership=0, aromatic=0
      - 'membership_only':   membership 保留, aromatic=0
      - 'aromatic_only':     membership=0, aromatic 保留
      - 'membership+aromatic': 原始 (不修改)
    """
    data_ablated = {k: v for k, v in data.items()}
    node_mats = data['node_mats'].clone()

    if mode == 'no_ring_info':
        node_mats[:, :, AROMATIC_IDX] = 0
        node_mats[:, :, MEMBERSHIP_IDX] = 0
    elif mode == 'membership_only':
        node_mats[:, :, AROMATIC_IDX] = 0
    elif mode == 'aromatic_only':
        node_mats[:, :, MEMBERSHIP_IDX] = 0
    elif mode == 'membership+aromatic':
        pass  # 不修改

    data_ablated['node_mats'] = node_mats
    return data_ablated


def train_and_eval(model_name, data, device, params, mode, seed=SEED):
    """训练并评估单个 ablation 配置

    使用 parent-molecule grouped split (与 Stage 1 一致):
      - canonical_splits: GroupShuffleSplit + GroupKFold (5-fold CV)
      - 同一分子的所有环级样本进入同一 split, 防止数据泄漏
    """
    from common.tasks import canonical_splits

    set_full_seed(seed)

    # 消融特征
    data_abl = ablate_features(data, mode)

    n = data_abl['n']
    groups = data_abl['smiles']  # parent-molecule grouping

    # canonical_splits 返回: test_idx, cv_folds, final_train_idx, final_val_idx
    test_idx_np, cv_folds, final_train_idx, final_val_idx = canonical_splits(
        n, seed=seed, groups=groups)

    # 5-fold CV (主要评估)
    fold_results = []
    for fold_i, (train_idx, val_idx) in enumerate(cv_folds):
        train_idx = torch.tensor(train_idx, device=device)
        val_idx = torch.tensor(val_idx, device=device)
        test_idx = torch.tensor(test_idx_np, device=device)

        t0 = time.time()
        model = train_model(model_name, params, data_abl, train_idx, val_idx,
                           device, params['n_epochs'], params['patience'], seed)
        train_time = time.time() - t0

        tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, data_abl, train_idx,
                                                  params['batch_size'], device)
        te_r2, te_mae, te_rmse, _, _ = eval_model(model, data_abl, test_idx,
                                                   params['batch_size'], device)

        fold_results.append({
            'fold': fold_i,
            'train_r2': tr_r2, 'test_r2': te_r2,
            'test_mae': te_mae, 'test_rmse': te_rmse,
            'train_time': train_time,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5-fold 平均
    avg = {k: np.mean([r[k] for r in fold_results]) for k in
           ['train_r2', 'test_r2', 'test_mae', 'test_rmse', 'train_time']}
    avg['fold_results'] = fold_results
    return avg


def main():
    import argparse
    parser = argparse.ArgumentParser(description='P1c: Leakage Ablation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    parser.add_argument('--model', type=str, default='MPNN')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS)

    tasks = args.tasks.split(',')
    task_map = {t['name']: t for t in TASKS}
    target_map = {'HOMA': 'homa_value', 'NICS_1zz': 'NICS_value', 'MBCO': 'mbco_value'}
    seeds = [int(s) for s in args.seeds.split(',')]

    modes = ['no_ring_info', 'membership_only', 'aromatic_only', 'membership+aromatic']

    all_results = []
    for task_name in tasks:
        task = task_map[task_name]
        target_col = target_map[task_name]
        print(f"\n{'='*60}")
        print(f"# {task_name} / {args.model} ({len(seeds)} seeds)")
        print(f"{'='*60}")

        # 加载数据 (原始, ring_flag=10)
        data = load_adj_format(task['dataset_path'], target_col, NVL, MAX_ATOMS,
                              ring_flag_value=RING_FLAG_VALUE, device=device)

        # 验证特征位置
        nm = data['node_mats']
        print(f"  node_mats shape: {nm.shape}")
        print(f"  aromatic flag (NVL-5) 非零比例: {(nm[:, :, AROMATIC_IDX] != 0).float().mean():.4f}")
        print(f"  membership (NVL-15) 非零比例: {(nm[:, :, MEMBERSHIP_IDX] != 0).float().mean():.4f}")

        for mode in modes:
            print(f"\n  --- {mode} ---")
            for seed in seeds:
                try:
                    result = train_and_eval(args.model, data, device, params, mode, seed)
                    result['task'] = task_name
                    result['config'] = mode
                    result['model'] = args.model
                    result['seed'] = seed
                    all_results.append(result)
                    print(f"    seed={seed}: Train R2={result['train_r2']:.4f} | "
                          f"Test R2={result['test_r2']:.4f} MAE={result['test_mae']:.4f}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"    [失败] seed={seed}: {e}")

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, 'leakage_ablation.csv')
        df.to_csv(csv_path, index=False)

        # 聚合 (mean ± SD)
        agg = df.groupby(['task', 'config']).agg(
            test_r2_mean=('test_r2', 'mean'),
            test_r2_std=('test_r2', 'std'),
            test_mae_mean=('test_mae', 'mean'),
            test_mae_std=('test_mae', 'std'),
            n_seeds=('seed', 'count'),
        ).reset_index()

        agg_csv = os.path.join(RESULTS_DIR, 'leakage_ablation_agg.csv')
        agg.to_csv(agg_csv, index=False)
        print(f"\n保存: {csv_path}")
        print(f"聚合: {agg_csv}")
        print("\n" + agg.to_string(index=False))

        # 画图
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=120)
        task_labels = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS(1)zz', 'MBCO': 'MBCO'}
        mode_order = ['no_ring_info', 'membership_only', 'aromatic_only', 'membership+aromatic']
        mode_labels = ['No Ring\nInfo', 'Membership\nOnly', 'Aromatic\nFlag Only', 'Membership+\nAromatic']
        colors = ['gray', 'steelblue', 'coral', 'green']

        for i, task_name in enumerate(['HOMA', 'NICS_1zz', 'MBCO']):
            if task_name not in tasks:
                continue
            sub = agg[agg['task'] == task_name].copy()
            sub['config'] = pd.Categorical(sub['config'], categories=mode_order, ordered=True)
            sub = sub.sort_values('config')

            ax = axes[i]
            x = range(len(sub))
            bars = ax.bar(x, sub['test_r2_mean'], yerr=sub['test_r2_std'],
                         color=colors, alpha=0.8, capsize=5, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(mode_labels, fontsize=9)
            ax.set_ylabel('Test R² (mean ± SD)', fontsize=11)
            ax.set_title(task_labels.get(task_name, task_name), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for j, (_, row) in enumerate(sub.iterrows()):
                ax.text(j, row['test_r2_mean'] + row['test_r2_std'] + 0.02,
                        f"{row['test_r2_mean']:.3f}±{row['test_r2_std']:.3f}",
                        ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, 'leakage_ablation.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        print(f"图表: {fig_path}")


if __name__ == '__main__':
    main()
