
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
Step 6: 加入 Stage I Pretraining 对照

在 Siamese MPNN 上比较:
  A. random initialization
  B. Stage I aromaticity-pretrained encoder (mpnn_label, HOMA-trained on full 0716 dataset)

其它条件完全一致。重点比较:
  100% / 50% / 20% / 10% training data
看 Stage I static aromaticity representation 是否能提高 low-data Δ-learning.

输出:
  results/lunci10_delta_learning/05_pretraining_transfer/
"""
import os
import sys
import time
import json
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
STAGE_I_WEIGHTS = '_PROJ_ROOT + "/unified_models"/0427_unified_results/mpnn_label/best_model.pth'

sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, compute_metrics
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import set_full_seed
from unified_models.mpnn.model import MPNNModel
# 复用 Step 5 的 SiameseMPNN 结构和数据加载函数
from generalization_test.code.step5_siamese_mpnn import (
    SiameseMPNN, load_lunci10_pairs, train_siamese, evaluate_siamese
)

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/05_pretraining_transfer')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIR_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_delta_learning/01_pair_dataset')

NVL = 60
MAX_ATOMS = 75
RING_FLAG_VALUE = 10
SEEDS = [42, 123, 456, 789, 2024]
FRACTIONS = [1.0, 0.5, 0.2, 0.1]  # 100%, 50%, 20%, 10% training data
MODES = ['random', 'pretrained']

TASK_COL_MAP = {
    'HOMA':      ('HOMA',    'homa_value'),
    'NICS_1zz':  ('NICS_ZZ', 'NICS_value'),
    'MBCO':      ('MBCO',    'mbco_value'),
}

# Encoder-only keys (排除 output_layer, 因为 SiameseMPNN 不使用 encoder.output_layer)
ENCODER_PREFIXES = (
    'init_transform.', 'conv_layers.', 'conv_bn_',
    'hidden_layers.', 'hidden_bns.',
)


def build_siamese(mode='random', seed=42):
    """构建 SiameseMPNN, 根据 mode 决定是否加载 Stage I 预训练权重"""
    set_full_seed(seed)
    model = SiameseMPNN(
        node_vec_len=NVL, hidden_dim=128,
        n_conv=3, n_hidden=2, p_dropout=0.2, mode='label'
    )

    if mode == 'pretrained':
        if not os.path.exists(STAGE_I_WEIGHTS):
            raise FileNotFoundError(f"Stage I weights not found: {STAGE_I_WEIGHTS}")

        # 加载 Stage I MPNN label 模型权重 (seed=42, 在 HOMA 上训练)
        state_dict = torch.load(STAGE_I_WEIGHTS, map_location='cpu', weights_only=False)

        # 仅加载 encoder 部分 (排除 output_layer, 因为 SiameseMPNN 不使用)
        encoder_state = {}
        skipped = []
        for k, v in state_dict.items():
            if k.startswith(ENCODER_PREFIXES):
                encoder_state[f'encoder.{k}'] = v
            else:
                skipped.append(k)

        # encoder.* 前缀匹配 SiameseMPNN 中 self.encoder 命名
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        # missing 应该包含: pair_head.* 和 encoder.output_layer.*
        # unexpected 应该为空
        n_loaded = len(encoder_state)
        print(f"    [pretrained] Loaded {n_loaded} encoder tensors from Stage I (skipped: {len(skipped)} output_layer keys)")
        print(f"    [pretrained] missing keys (expected, pair_head + output_layer): {len(missing)}")
        if unexpected:
            print(f"    [pretrained] WARNING unexpected keys: {unexpected[:5]}")

    return model


def subsample_train_idx(train_idx, fraction, seed):
    """按 fraction 子采样训练索引 (固定 seed)"""
    if fraction >= 1.0:
        return train_idx
    n_total = len(train_idx)
    n_keep = max(2, int(fraction * n_total))  # 至少 2 个样本以避免 batch=1
    rng = np.random.RandomState(seed + 7919)  # 与训练 seed 解耦的子采样种子
    perm = rng.permutation(n_total)[:n_keep]
    return train_idx[perm]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO')
    parser.add_argument('--fractions', type=str, default=','.join(str(f) for f in FRACTIONS))
    parser.add_argument('--modes', type=str, default=','.join(MODES))
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Fractions: {args.fractions}")
    print(f"Modes: {args.modes}")
    print(f"Stage I weights: {STAGE_I_WEIGHTS}")

    fractions = [float(x) for x in args.fractions.split(',')]
    modes = args.modes.split(',')

    task_map = {t['name']: t for t in TASKS}
    tasks = args.tasks.split(',')

    all_metrics = []

    for task_name in tasks:
        task = task_map[task_name]
        print(f"\n{'='*70}")
        print(f"# Task: {task_name}")
        print(f"{'='*70}")

        # 加载数据和 pairs (与 Step 5 完全一致)
        l10_data, df_pairs, pair_indices = load_lunci10_pairs(task_name, task, device)

        y_delta = df_pairs['delta_A'].values.astype(np.float32)
        groups = df_pairs['scaffold_id'].values

        print(f"  Pairs: {len(y_delta)}")
        print(f"  Scaffolds: {len(np.unique(groups))}")

        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")
            set_full_seed(seed)

            # scaffold-grouped split: 80/20 (与 Step 5 完全一致)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_val_idx, test_idx = next(gss.split(np.arange(len(y_delta)), y_delta, groups))

            # 进一步划分 train/val: 87.5%/12.5% (与 Step 5 完全一致)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            train_rel, val_rel = next(gss2.split(train_val_idx, y_delta[train_val_idx],
                                                  groups[train_val_idx]))
            train_idx_full = train_val_idx[train_rel]
            val_idx = train_val_idx[val_rel]

            print(f"    Full Train: {len(train_idx_full)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            for fraction in fractions:
                # 子采样训练集
                train_idx = subsample_train_idx(train_idx_full, fraction, seed)
                print(f"\n    Fraction={fraction:.0%} | Train: {len(train_idx)}")

                for mode in modes:
                    print(f"\n      Mode={mode}")
                    set_full_seed(seed)

                    # 构建模型
                    model = build_siamese(mode=mode, seed=seed)

                    # 训练
                    t0 = time.time()
                    model = train_siamese(
                        model, l10_data, pair_indices, y_delta, groups,
                        train_idx, val_idx, device,
                        n_epochs=args.n_epochs, patience=args.patience, seed=seed
                    )
                    train_time = time.time() - t0

                    # 评估 (测试集与 Step 5 完全一致)
                    metrics = evaluate_siamese(
                        model, l10_data, pair_indices, y_delta, test_idx, device
                    )
                    metrics['task'] = task_name
                    metrics['seed'] = seed
                    metrics['fraction'] = fraction
                    metrics['mode'] = mode
                    metrics['train_time'] = train_time
                    metrics['n_train'] = len(train_idx)
                    metrics['n_val'] = len(val_idx)
                    metrics['n_test'] = len(test_idx)
                    all_metrics.append(metrics)

                    print(f"      ΔR²={metrics['r2']:.4f} | ΔMAE={metrics['mae']:.4f} | "
                          f"Spearman={metrics['spearman']:.4f} | "
                          f"Kendall={metrics['kendall']:.4f} | "
                          f"Pairwise={metrics['pairwise_acc']:.4f} ({train_time:.0f}s)")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # 保存 per-seed 结果
    df_metrics = pd.DataFrame(all_metrics)
    per_seed_path = os.path.join(OUTPUT_DIR, 'pretraining_per_seed.csv')
    df_metrics.to_csv(per_seed_path, index=False)
    print(f"\nPer-seed results saved: {per_seed_path}")

    # 聚合: 按 (task, fraction, mode) 计算 mean±SD
    agg = df_metrics.groupby(['task', 'fraction', 'mode']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        spearman_mean=('spearman', 'mean'), spearman_std=('spearman', 'std'),
        kendall_mean=('kendall', 'mean'), kendall_std=('kendall', 'std'),
        pairwise_mean=('pairwise_acc', 'mean'), pairwise_std=('pairwise_acc', 'std'),
        train_time_mean=('train_time', 'mean'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    agg_path = os.path.join(OUTPUT_DIR, 'pretraining_agg.csv')
    agg.to_csv(agg_path, index=False)
    print(f"Aggregated results saved: {agg_path}")

    # 计算 ΔR², ΔMAE, ΔSpearman (pretrained - random)
    pivot = agg.pivot_table(
        index=['task', 'fraction'],
        columns='mode',
        values=['r2_mean', 'mae_mean', 'spearman_mean']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    pivot['delta_r2'] = pivot['r2_mean_pretrained'] - pivot['r2_mean_random']
    pivot['delta_mae'] = pivot['mae_mean_pretrained'] - pivot['mae_mean_random']
    pivot['delta_spearman'] = pivot['spearman_mean_pretrained'] - pivot['spearman_mean_random']

    delta_path = os.path.join(OUTPUT_DIR, 'pretraining_delta.csv')
    pivot.to_csv(delta_path, index=False)
    print(f"Delta comparison saved: {delta_path}")

    # 文字报告
    report_path = os.path.join(OUTPUT_DIR, 'pretraining_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Step 6: Stage I Pretraining 对照报告\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- **Stage I 预训练权重**: `{STAGE_I_WEIGHTS}`\n")
        f.write("- Stage I MPNN (mode=label, seed=42) 在原始 0716 训练集上训练 HOMA, "
                "其 encoder 权重作为 SiameseMPNN encoder 的初始化.\n")
        f.write("- **Pair head 始终随机初始化**, 只迁移 encoder.\n")
        f.write(f"- **Seeds**: {SEEDS}\n")
        f.write(f"- **Training fractions**: {fractions}\n")
        f.write(f"- **Modes**: {modes}\n")
        f.write("- 其它超参与 Step 5 (Siamese MPNN) 完全一致.\n\n")

        f.write("## Δ (pretrained - random) 性能差异\n\n")
        f.write("| Task | Fraction | ΔR² | ΔMAE | ΔSpearman |\n")
        f.write("|------|----------|-----|------|-----------|\n")
        for _, row in pivot.iterrows():
            f.write(f"| {row['task']} | {row['fraction']:.0%} | "
                    f"{row['delta_r2']:+.4f} | {row['delta_mae']:+.4f} | "
                    f"{row['delta_spearman']:+.4f} |\n")

        f.write("\n## 详细聚合指标 (mean±SD over 5 seeds)\n\n")
        for task_name in tasks:
            f.write(f"### {task_name}\n\n")
            f.write("| Fraction | Mode | R² | MAE | Spearman ρ | Pairwise Acc |\n")
            f.write("|----------|------|----|-----|------------|--------------|\n")
            sub = agg[agg['task'] == task_name].sort_values(['fraction', 'mode'])
            for _, row in sub.iterrows():
                f.write(f"| {row['fraction']:.0%} | {row['mode']} | "
                        f"{row['r2_mean']:.4f}±{row['r2_std']:.4f} | "
                        f"{row['mae_mean']:.4f}±{row['mae_std']:.4f} | "
                        f"{row['spearman_mean']:.4f}±{row['spearman_std']:.4f} | "
                        f"{row['pairwise_mean']:.4f}±{row['pairwise_std']:.4f} |\n")
            f.write("\n")

        # 关键发现: 是否 low-data 提升明显而 full-data 提升小
        f.write("## 关键发现\n\n")
        f.write("**判定标准**: 如果 full-data (100%) ΔR² ≈ 0 (或负), "
                "但 low-data (≤20%) ΔR² 显著为正, "
                "则说明 Stage I 预训练编码器在低数据情境下提供实质性迁移收益, "
                "但在数据充足时被 from-scratch 训练追平.\n\n")

        for task_name in tasks:
            sub = pivot[pivot['task'] == task_name].sort_values('fraction')
            full_delta_r2 = sub[sub['fraction'] == 1.0]['delta_r2'].iloc[0] if len(sub[sub['fraction'] == 1.0]) > 0 else float('nan')
            low_data_deltas = sub[sub['fraction'] <= 0.2]['delta_r2']
            low_data_avg = low_data_deltas.mean() if len(low_data_deltas) > 0 else float('nan')
            f.write(f"- **{task_name}**: full-data ΔR² = {full_delta_r2:+.4f}, "
                    f"low-data (≤20%) avg ΔR² = {low_data_avg:+.4f}. ")
            if full_delta_r2 < 0.02 and low_data_avg > 0.02:
                f.write("**符合\"full-data 提升很小, 但 low-data 提升明显\"模式.**\n")
            elif low_data_avg > full_delta_r2:
                f.write("low-data 提升大于 full-data, 但差距有限.\n")
            else:
                f.write("未观察到 low-data 优势.\n")

    print(f"Report saved: {report_path}")

    # 打印汇总
    print(f"\n{'='*70}")
    print("Step 6: Stage I Pretraining 对照 — Δ (pretrained - random) 汇总")
    print(f"{'='*70}")
    print(pivot[['task', 'fraction', 'delta_r2', 'delta_mae', 'delta_spearman']].to_string(index=False))


if __name__ == '__main__':
    main()
