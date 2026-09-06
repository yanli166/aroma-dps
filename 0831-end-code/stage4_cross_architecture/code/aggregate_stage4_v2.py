#!/usr/bin/env python3
"""
Stage4 (v2) 汇总：合并各 backbone 部分结果 → 完整性/泄漏 → ΔMAE per-backbone 报告。

读取:
  results_v2/stage4/partial/{BACKBONE}/per_seed_results.csv  (并行 worker 产物, 可选)
  results_v2/stage4/per_seed_results.csv                     (单进程/总产物, 主来源)
输出 (results_v2/stage4/):
  per_seed_results.csv      合并去重后的逐 run 记录
  cv_results.csv            (task, model, config, n, cv_mae, cv_rmse, cv_r2, n_seeds)
  final_test_results.csv    (task, model, config, test_mae, test_rmse, test_r2, n_test_seeds)
  delta_mae_per_backbone.csv  per-backbone ΔMAE (membership−base), %MAE 下降, ΔR2
并打印汇总表与完整性/泄漏结论。
"""
import os
import argparse
import numpy as np
import pandas as pd

from common.constants import RESULTS_V2_DIR
from common.protocol import completeness_check

MAIN = os.path.join(RESULTS_V2_DIR, 'stage4')
PARTIAL = os.path.join(MAIN, 'partial')


def collect(seed):
    frames = []
    root = os.path.join(MAIN, 'per_seed_results.csv')
    if os.path.exists(root):
        frames.append(pd.read_csv(root))
    if os.path.isdir(PARTIAL):
        for bb_dir in sorted(os.listdir(PARTIAL)):
            p = os.path.join(PARTIAL, bb_dir, 'per_seed_results.csv')
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"[Stage4v2-agg] 未找到任何 per_seed_results.csv in {MAIN}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['seed', 'task', 'model', 'config'], keep='first')
    df['model'] = df['model'].astype(str)
    df['config'] = df['config'].astype(str)
    df['task'] = df['task'].astype(str)
    return df[df['seed'] == seed].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()

    df = collect(args.seed)
    if df.empty:
        raise SystemExit(f"[Stage4v2-agg] seed={args.seed} 无任何记录")

    models = sorted(df['model'].unique())
    configs = sorted(df['config'].unique())
    tasks = sorted(df['task'].unique())

    # ---------- 完整性检查 ----------
    expected_tuples = {(t, m, c) for t in tasks for m in models for c in configs}
    actual_tuples = set(map(tuple, df[['task', 'model', 'config']].to_numpy()))
    missing = expected_tuples.difference(actual_tuples)
    if missing:
        raise RuntimeError(f"[Stage4v2] 实验缺失 {len(missing)} 组:\n" +
                           "\n".join(map(str, sorted(missing))))
    # 协议 completeness_check
    completeness_check(pd.MultiIndex.from_tuples(actual_tuples),
                       pd.MultiIndex.from_tuples(expected_tuples),
                       'Stage4v2')

    # 合并后的统一 per_seed_results.csv 写回主目录 (协议要求保留该文件)
    df.to_csv(os.path.join(MAIN, 'per_seed_results.csv'), index=False)
    print(f"[Stage4v2] 已合并 per_seed_results.csv -> {MAIN}")
    errors = df[df['run_status'] != 'ok']
    print(f"[Stage4v2] 完整性检查通过: 期望 {len(expected_tuples)} 组, 全部存在.")
    if len(errors):
        print(f"[Stage4v2] 警告: 存在 error run {len(errors)} 行:\n{errors[['task','model','config','error_message']].to_string(index=False)}")

    # ---------- 一致性: 同一 (task, backbone) 的 test 集必须固定 ----------
    # 数据划分由 split_seed 固定, 与 config/model_seed 无关, 故 n (样本数) 应一致。
    n_uniq_per_combo = df.groupby(['task', 'model'])['n'].nunique()
    if (n_uniq_per_combo > 1).any():
        print("[Stage4v2] 警告: 同一 (task,backbone) 在不同 config 下 n 不一致!")

    # ---------- cv_results.csv ----------
    cv = df[df['cv_mae'].notna()].copy()
    cv_out = cv.groupby(['task', 'model', 'config'], as_index=False).agg(
        n=('n', 'first'), cv_mae=('cv_mae', 'mean'), cv_rmse=('cv_rmse', 'mean'),
        cv_r2=('cv_r2', 'mean'), best_epochs=('best_epochs', 'first'),
        E_star=('E_star', 'first'), n_seeds=('seed', 'count'))
    cv_out = cv_out.sort_values(['task', 'model', 'config'])
    cv_out.to_csv(os.path.join(MAIN, 'cv_results.csv'), index=False)

    # ---------- final_test_results.csv ----------
    ft = df[df['test_mae'].notna()].copy()
    ft_out = ft.groupby(['task', 'model', 'config'], as_index=False).agg(
        n=('n', 'first'), test_mae=('test_mae', 'mean'), test_rmse=('test_rmse', 'mean'),
        test_r2=('test_r2', 'mean'), train_mae=('train_mae', 'mean'),
        test_time_sec=('train_time_sec', 'mean'), n_test_seeds=('seed', 'count'))
    ft_out = ft_out.sort_values(['task', 'model', 'config'])
    ft_out.to_csv(os.path.join(MAIN, 'final_test_results.csv'), index=False)

    # ---------- ΔMAE per backbone ----------
    rows = []
    for (t, m), grp in ft.groupby(['task', 'model']):
        b = grp[grp['config'] == 'base']
        mm = grp[grp['config'] == 'membership']
        if b.empty or mm.empty:
            continue
        b = b.iloc[0]; mm = mm.iloc[0]
        d_mae = b['test_mae'] - mm['test_mae']
        pct = (d_mae / b['test_mae']) * 100 if b['test_mae'] else np.nan
        rows.append({'task': t, 'backbone': m,
                     'base_test_mae': b['test_mae'],
                     'membership_test_mae': mm['test_mae'],
                     'delta_mae': d_mae,
                     'delta_mae_pct': pct,          # %MAE 下降 (正=membership更好)
                     'delta_r2': mm['test_r2'] - b['test_r2']})
    dmae = pd.DataFrame(rows)
    if not dmae.empty:
        dmae.to_csv(os.path.join(MAIN, 'delta_mae_per_backbone.csv'), index=False)

    print("\n===== cv MAE (V2) =====")
    print(cv_out[['task', 'model', 'config', 'cv_mae', 'E_star']].to_string(index=False))
    print("\n===== Final Test (V2) =====")
    print(ft_out[['task', 'model', 'config', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
    print("\n===== ΔMAE per backbone (membership − base, 正=membership更优) =====")
    if not dmae.empty:
        print(dmae.to_string(index=False))
        print("\nDeltaMAE per backbone (跨 3 任务平均, 单位 task 尺度):")
        agg = dmae.groupby('backbone')['delta_mae'].mean().round(5)
        print(agg.to_string())
    print("\n[Stage4v2] 汇总完成 →", MAIN)


if __name__ == '__main__':
    main()