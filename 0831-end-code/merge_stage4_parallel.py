#!/usr/bin/env python
"""
Stage 4 并行结果合并脚本

合并来源:
  - stage4_g3          (GPU3 原进程, 全部 backbone, 含 seed_42 完整结果)
  - stage4_g0_parallel (GPU0, MPNN, seeds 123-2024)
  - stage4_g1_parallel (GPU1, GIN,  seeds 123-2024)
  - stage4_g2_parallel (GPU2, GAT,  seeds 123-2024)

去重策略:
  对相同 (seed, task, model, config) 的记录, 优先保留 parallel worker 结果
  (因为 parallel worker 有独占 GPU, 训练更稳定)。
  DMPNN 仅存在于 stage4_g3, 无冲突。

输出:
  results/stage4_cross_arch/per_seed_results.csv  (合并去重后)
  results/stage4_cross_arch/all_stage4_cross_arch_summary.csv  (聚合)
"""
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

# 合并来源 (按优先级降序: parallel > g3)
SOURCES = [
    ("stage4_g0_parallel", "MPNN"),   # GPU0 专跑 MPNN
    ("stage4_g1_parallel", "GIN"),    # GPU1 专跑 GIN
    ("stage4_g2_parallel", "GAT"),    # GPU2 专跑 GAT
    ("stage4_dmpnn_g0", "DMPNN"),     # GPU0 补跑 DMPNN seed 2024
    ("stage4_dmpnn_g1", "DMPNN"),     # GPU1 补跑 DMPNN seeds 123,456
    ("stage4_dmpnn_g2", "DMPNN"),     # GPU2 补跑 DMPNN seeds 123,456
    ("stage4_dmpnn_g3", "DMPNN"),     # GPU3 补跑 DMPNN seeds 789,2024
    ("stage4_g3", None),              # GPU3 原进程 (含 DMPNN seed_42 + 全部 seed_42)
]

# 去重键
DEDUP_KEYS = ["seed", "task", "model", "config"]


def merge_stage4():
    """合并 Stage 4 并行结果, 去重, 保存"""
    main_dir = RESULTS / "stage4_cross_arch"
    main_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    source_info = []

    for sub_dir, expected_bb in SOURCES:
        csv_path = RESULTS / sub_dir / "per_seed_results.csv"
        if not csv_path.exists():
            print(f"  [跳过] {sub_dir}: 无 per_seed_results.csv")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  [跳过] {sub_dir}: CSV 为空")
            continue
        df["_source"] = sub_dir
        print(f"  [读取] {sub_dir}: {len(df)} 行 (backbones={df['model'].unique().tolist() if 'model' in df else 'N/A'})")
        all_dfs.append(df)
        source_info.append((sub_dir, len(df)))

    if not all_dfs:
        print("错误: 无可合并的 Stage 4 结果")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  [合并前] 总行数: {len(merged)}")

    # 检查重复
    dup_mask = merged.duplicated(subset=DEDUP_KEYS, keep=False)
    if dup_mask.any():
        dups = merged[dup_mask].sort_values(DEDUP_KEYS)
        print(f"  [去重] 发现 {len(dups)} 行重复 (按 {DEDUP_KEYS})")
        print(f"    重复组合:\n{dups[DEDUP_KEYS].drop_duplicates().to_string()}")

        # 去重: 优先保留 parallel worker (source 含 'parallel'), 其次 g3
        merged["_priority"] = merged["_source"].apply(
            lambda s: 0 if "parallel" in s else 1)
        merged = merged.sort_values(DEDUP_KEYS + ["_priority"])
        before = len(merged)
        merged = merged.drop_duplicates(subset=DEDUP_KEYS, keep="first")
        after = len(merged)
        print(f"    去重: {before} -> {after} 行 (保留 parallel 优先)")
    else:
        print(f"  [去重] 无重复记录")

    # 清理临时列
    merged = merged.drop(columns=["_source", "_priority"], errors="ignore")

    # 按 seed, task, model, config 排序
    sort_keys = [k for k in ["seed", "task", "model", "config"] if k in merged.columns]
    merged = merged.sort_values(sort_keys).reset_index(drop=True)

    # 保存
    out_path = main_dir / "per_seed_results.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n  [保存] {out_path}: {len(merged)} 行")

    # 验证: 检查覆盖度
    expected_combos = 5 * 3 * 4 * 2  # seeds × tasks × backbones × configs = 120
    print(f"\n  [覆盖度] {len(merged)}/{expected_combos} 组合")
    for seed in [42, 123, 456, 789, 2024]:
        for task in ["HOMA", "NICS_1zz", "MBCO"]:
            for bb in ["MPNN", "GIN", "GAT", "DMPNN"]:
                for cfg in ["base", "ring_conditioned"]:
                    n = len(merged[(merged["seed"] == seed) & (merged["task"] == task) &
                                   (merged["model"] == bb) & (merged["config"] == cfg)])
                    if n == 0:
                        print(f"    缺失: seed={seed} task={task} model={bb} config={cfg}")
    print(f"  [覆盖度] 检查完成")

    # 聚合
    print(f"\n  [聚合] 生成 summary...")
    from common.constants import METRIC_COLS_FOR_AGG
    metric_cols = [c for c in METRIC_COLS_FOR_AGG if c in merged.columns]
    group_keys = ["task", "model", "config"]
    agg = merged.groupby(group_keys)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    agg_path = main_dir / "all_stage4_cross_arch_summary.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  [保存] {agg_path}")

    # 显示关键结果
    display_cols = [c for c in ["task", "model", "config",
                                "test_r2_mean", "test_r2_std",
                                "test_mae_mean", "test_rmse_mean"]
                    if c in agg.columns]
    print(f"\n  [Stage 4 聚合结果]")
    print(agg[display_cols].to_string(index=False))

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 4 并行结果合并")
    print("=" * 60)
    merge_stage4()
    print("\n完成。")
