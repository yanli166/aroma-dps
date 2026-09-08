#!/usr/bin/env python3
"""
Stage3 补跑缺失种子 (支持所有3种 training mode)

处理缺失项:
  - NICS_1zz seed 123: 全部3种 config (direct_supervised, random_mask_pretrain, ring_mask_pretrain)
  - MBCO ring_mask_pretrain: seeds 42, 123, 456, 789

用法:
  # NICS seed 123 (全部3种 mode)
  python resume_stage3_missing.py --task NICS_1zz --gpu 0 --seeds 123 --modes all

  # MBCO ring_mask_pretrain (仅 ring_mask)
  python resume_stage3_missing.py --task MBCO --gpu 1 --seeds 42,123,456,789 --modes ring_mask_pretrain
"""
import os
import sys
import argparse
import pandas as pd
import torch

LAST_END_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAST_END_ROOT)

from common.tasks import get_task, canonical_splits
from common.graph_data import load_adj_format
from common.constants import PER_SEED_COLUMNS
from stage3_mask_pretraining.code.run_pretrain_eval import (
    run_one_mode, select_best_stage2_config, DEFAULT_PARAMS,
    NODE_VEC_LEN, MAX_ATOMS, TRAINING_MODES)

STAGE3_DIRS = {
    'HOMA':      os.path.join(LAST_END_ROOT, 'results', 'stage3_g0'),
    'NICS_1zz':  os.path.join(LAST_END_ROOT, 'results', 'stage3_g1'),
    'MBCO':      os.path.join(LAST_END_ROOT, 'results', 'stage3_g2'),
}


def find_stage2_csv():
    p = os.path.join(LAST_END_ROOT, 'results', 'stage2_ring_conditioning',
                     'per_seed_results.csv')
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"未找到 Stage2 CSV: {p}")


def load_existing(output_root, modes_to_replace, seeds_to_replace):
    """加载现有 CSV, 仅移除指定 mode+seed 的行"""
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'config' in df.columns and 'seed' in df.columns:
            mask = df['config'].isin(modes_to_replace) & df['seed'].isin(seeds_to_replace)
            df = df[~mask].copy()
        return df
    return pd.DataFrame()


def append_and_save(output_root, existing_df, new_results):
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


def main():
    parser = argparse.ArgumentParser(description='Stage3 补跑缺失种子')
    parser.add_argument('--task', type=str, required=True,
                        choices=['HOMA', 'NICS_1zz', 'MBCO'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=str, required=True,
                        help='逗号分隔的种子列表, 如 123 或 42,123,456,789')
    parser.add_argument('--modes', type=str, default='all',
                        help='all 或逗号分隔的 mode 列表 (direct_supervised,random_mask_pretrain,ring_mask_pretrain)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    if args.modes == 'all':
        modes = TRAINING_MODES
    else:
        modes = args.modes.split(',')

    task = get_task(args.task)
    output_root = STAGE3_DIRS[args.task]
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Stage3 补跑 | task={args.task} | seeds={seeds} | modes={modes}")
    print(f"device={device} | output={output_root}")
    print(f"{'='*60}", flush=True)

    # Stage2 最佳配置
    stage2_csv = find_stage2_csv()
    best_cfg = select_best_stage2_config(stage2_csv, args.task)
    print(f"Stage2 最佳配置: backbone={best_cfg['backbone']}, "
          f"ring_flag={best_cfg['ring_flag']}, readout={best_cfg['readout_mode']}")

    # 加载数据
    data = load_adj_format(task['dataset_path'], task['target_col'],
                           NODE_VEC_LEN, MAX_ATOMS,
                           ring_flag_value=best_cfg['ring_flag'], device=device)

    # 加载现有结果 (仅移除要重跑的 mode+seed 行)
    existing_df = load_existing(output_root, modes, seeds)
    print(f"现有结果: {len(existing_df)} 行 (已排除 modes={modes} seeds={seeds})")

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = 200
    params['patience'] = 30

    new_results = []
    for seed in seeds:
        splits = canonical_splits(data['n'], seed=seed, groups=data['smiles'])
        test_t = torch.tensor(splits[0], dtype=torch.long, device=device)
        ftrain_t = torch.tensor(splits[2], dtype=torch.long, device=device)
        fval_t = torch.tensor(splits[3], dtype=torch.long, device=device)
        cv_t = [(torch.tensor(tr, dtype=torch.long, device=device),
                 torch.tensor(va, dtype=torch.long, device=device))
                for tr, va in splits[1]]
        splits_t = (test_t, cv_t, ftrain_t, fval_t)

        for mode in modes:
            print(f"\n{'#'*50}\n# Seed={seed} | {mode} | {args.task}\n{'#'*50}")
            try:
                summary = run_one_mode(task, mode, best_cfg, params,
                                       data, splits_t, device, output_root, seed,
                                       pretrain_epochs=100)
                summary['seed'] = seed
                new_results.append(summary)
                print(f"  [Seed={seed}|{mode}] 成功: test_r2={summary.get('test_r2', 'N/A')}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [Seed={seed}|{mode}] 失败: {e}", flush=True)

    if new_results:
        combined = append_and_save(output_root, existing_df, new_results)
        print(f"\n[{args.task}] 完成: 新增 {len(new_results)} 条, 总计 {len(combined)} 条")
    else:
        print(f"\n[{args.task}] 无新结果")


if __name__ == '__main__':
    main()
