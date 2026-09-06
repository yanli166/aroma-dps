#!/usr/bin/env python3
"""
Stage 3 并行结果合并脚本

合并来源:
  - stage3_g0 (GPU0, HOMA)
  - stage3_g1 (GPU1, NICS_1zz)
  - stage3_g2 (GPU2, MBCO)

输出:
  results/stage3_mask_pretraining/per_seed_results.csv  (合并后)
  results/stage3_mask_pretraining/all_stage3_summary.csv  (聚合)
"""
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

SOURCES = ["stage3_g0", "stage3_g1", "stage3_g2"]


def merge_stage3():
    main_dir = RESULTS / "stage3_mask_pretraining"
    main_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for sub_dir in SOURCES:
        csv_path = RESULTS / sub_dir / "per_seed_results.csv"
        if not csv_path.exists():
            print(f"  [跳过] {sub_dir}: 无 per_seed_results.csv")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  [跳过] {sub_dir}: CSV 为空")
            continue
        print(f"  [读取] {sub_dir}: {len(df)} 行 (tasks={df['task'].unique().tolist() if 'task' in df else 'N/A'})")
        all_dfs.append(df)

    if not all_dfs:
        print("错误: 无可合并的 Stage 3 结果")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  [合并前] 总行数: {len(merged)}")

    # 去重 (同 seed+task+config 只保留一条)
    dedup_keys = ["seed", "task", "config"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_keys, keep="first")
    after = len(merged)
    if before != after:
        print(f"  [去重] {before} -> {after} 行")

    # 排序
    sort_keys = [k for k in ["seed", "task", "config"] if k in merged.columns]
    merged = merged.sort_values(sort_keys).reset_index(drop=True)

    # 保存
    out_path = main_dir / "per_seed_results.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n  [保存] {out_path}: {len(merged)} 行")

    # 覆盖度检查
    expected = 5 * 3 * 3  # seeds × tasks × configs = 45
    print(f"\n  [覆盖度] {len(merged)}/{expected} 组合")
    for seed in [42, 123, 456, 789, 2024]:
        for task in ["HOMA", "NICS_1zz", "MBCO"]:
            for cfg in ["direct_supervised", "random_mask_pretrain", "ring_mask_pretrain"]:
                n = len(merged[(merged["seed"] == seed) & (merged["task"] == task) &
                               (merged["config"] == cfg)])
                if n == 0:
                    print(f"    缺失: seed={seed} task={task} config={cfg}")

    # 聚合
    from common.constants import METRIC_COLS_FOR_AGG
    metric_cols = [c for c in METRIC_COLS_FOR_AGG if c in merged.columns]
    group_keys = ["task", "config"]
    agg = merged.groupby(group_keys)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    agg_path = main_dir / "all_stage3_summary.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  [保存] {agg_path}")

    # 显示关键结果
    display_cols = [c for c in ["task", "config",
                                "test_r2_mean", "test_r2_std",
                                "test_mae_mean", "test_rmse_mean"]
                    if c in agg.columns]
    print(f"\n  [Stage 3 聚合结果]")
    print(agg[display_cols].to_string(index=False))

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3 并行结果合并")
    print("=" * 60)
    merge_stage3()
    print("\n完成。")
