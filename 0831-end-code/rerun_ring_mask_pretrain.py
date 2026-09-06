#!/usr/bin/env python3
"""
Stage3 ring_mask_pretrain 重跑脚本

修复 mask_pretrain.py 中 ring_mask 函数的 3D valid_mask 维度不匹配 bug 后,
重新运行 ring_mask_pretrain 模式 (之前所有 seed×task 均失败)。

仅运行 ring_mask_pretrain (direct_supervised 和 random_mask_pretrain 已完成),
结果追加到 stage3_g0/g1/g2 的 per_seed_results.csv 中。

用法:
  python rerun_ring_mask_pretrain.py --gpu 0 --task HOMA
  python rerun_ring_mask_pretrain.py --gpu 1 --task NICS_1zz
  python rerun_ring_mask_pretrain.py --gpu 2 --task MBCO
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

LAST_END_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, canonical_splits, get_task, DEFAULT_SEED
from common.graph_data import load_adj_format
from common.train_eval import set_full_seed
from common.constants import PER_SEED_COLUMNS
from stage3_mask_pretraining.code.run_pretrain_eval import (
    run_one_mode, select_best_stage2_config, DEFAULT_PARAMS,
    NODE_VEC_LEN, MAX_ATOMS)
from stage3_mask_pretraining.code.mask_pretrain import ring_mask  # 验证修复可导入

SEEDS = [42, 123, 456, 789, 2024]
STAGE3_DIRS = {
    'HOMA':      os.path.join(LAST_END_ROOT, 'results', 'stage3_g0'),
    'NICS_1zz':  os.path.join(LAST_END_ROOT, 'results', 'stage3_g1'),
    'MBCO':      os.path.join(LAST_END_ROOT, 'results', 'stage3_g2'),
}


def find_stage2_csv():
    """定位 Stage2 合并后的 CSV"""
    p = os.path.join(LAST_END_ROOT, 'results', 'stage2_ring_conditioning',
                     'per_seed_results.csv')
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"未找到 Stage2 CSV: {p}")


def load_existing_results(output_root, seeds_to_replace=None):
    """加载现有的 per_seed_results.csv

    若 seeds_to_replace 提供, 仅移除这些 seed 的 ring_mask_pretrain 行 (保留其他 seed 的结果);
    否则移除所有 ring_mask_pretrain 行 (向后兼容)。
    """
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'config' in df.columns:
            mask = df['config'] == 'ring_mask_pretrain'
            if seeds_to_replace is not None:
                # 仅移除要重跑的 seed 的 ring_mask_pretrain 行
                mask = mask & df['seed'].isin(seeds_to_replace)
            df = df[~mask].copy()
        return df
    return pd.DataFrame()


def append_and_save(output_root, existing_df, new_results):
    """将新结果追加到现有 CSV, 保持列顺序"""
    new_df = pd.DataFrame(new_results)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df

    for col in PER_SEED_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[[c for c in PER_SEED_COLUMNS if c in combined.columns]]
    combined = combined.sort_values(['seed', 'task', 'config']).reset_index(drop=True)

    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    combined.to_csv(csv_path, index=False)
    print(f"  [保存] {csv_path}: {len(combined)} 行")
    return combined


def run_ring_mask_for_task(task_name, gpu, seeds=None):
    """为指定任务运行 ring_mask_pretrain (所有种子)"""
    if seeds is None:
        seeds = SEEDS

    task = get_task(task_name)
    output_root = STAGE3_DIRS[task_name]
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"重跑 ring_mask_pretrain | task={task_name} | device={device}")
    print(f"输出目录: {output_root}")
    print(f"{'='*60}", flush=True)

    # 1. 选 Stage2 最佳配置
    stage2_csv = find_stage2_csv()
    best_cfg = select_best_stage2_config(stage2_csv, task_name)
    print(f"Stage2 最佳配置: backbone={best_cfg['backbone']}, "
          f"ring_flag={best_cfg['ring_flag']}, readout={best_cfg['readout_mode']}")

    # 2. 加载数据 (用最佳 ring_flag)
    data = load_adj_format(task['dataset_path'], task['target_col'],
                           NODE_VEC_LEN, MAX_ATOMS,
                           ring_flag_value=best_cfg['ring_flag'], device=device)

    # 3. 加载现有结果 (仅移除要重跑的 seed 的 ring_mask_pretrain 行)
    existing_df = load_existing_results(output_root, seeds_to_replace=seeds)
    print(f"现有结果: {len(existing_df)} 行 (已排除 seeds={seeds} 的 ring_mask_pretrain)")

    # 4. 参数
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = 200
    params['patience'] = 30

    # 5. 逐种子运行 ring_mask_pretrain
    new_results = []
    for seed in seeds:
        print(f"\n{'#'*50}\n# Seed={seed} | ring_mask_pretrain | {task_name}\n{'#'*50}")

        # parent-molecule-level split
        splits = canonical_splits(data['n'], seed=seed, groups=data['smiles'])
        test_t = torch.tensor(splits[0], dtype=torch.long, device=device)
        ftrain_t = torch.tensor(splits[2], dtype=torch.long, device=device)
        fval_t = torch.tensor(splits[3], dtype=torch.long, device=device)
        cv_t = [(torch.tensor(tr, dtype=torch.long, device=device),
                 torch.tensor(va, dtype=torch.long, device=device))
                for tr, va in splits[1]]
        splits_t = (test_t, cv_t, ftrain_t, fval_t)

        try:
            summary = run_one_mode(task, 'ring_mask_pretrain', best_cfg, params,
                                   data, splits_t, device, output_root, seed,
                                   pretrain_epochs=100)
            summary['seed'] = seed
            new_results.append(summary)
            print(f"  [Seed={seed}] 成功: test_r2={summary.get('test_r2', 'N/A')}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [Seed={seed}] 失败: {e}", flush=True)

    # 6. 追加保存
    if new_results:
        combined = append_and_save(output_root, existing_df, new_results)
        print(f"\n[{task_name}] 完成: 新增 {len(new_results)} 条, 总计 {len(combined)} 条")
    else:
        print(f"\n[{task_name}] 无新结果")

    return new_results


def main():
    parser = argparse.ArgumentParser(description='重跑 Stage3 ring_mask_pretrain')
    parser.add_argument('--task', type=str, required=True,
                        choices=['HOMA', 'NICS_1zz', 'MBCO'],
                        help='任务名称')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)),
                        help='逗号分隔的种子列表')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_ring_mask_for_task(args.task, args.gpu, seeds)


if __name__ == '__main__':
    main()
