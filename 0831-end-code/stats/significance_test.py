"""
统计显著性检验脚本 (四阶段版)

对各阶段多种子结果做配对统计检验:
  1. paired t-test (scipy.stats.ttest_rel)
  2. Wilcoxon signed-rank test (scipy.stats.wilcoxon) — 小样本非参数检验

检验对象:
  - Stage2 内: base vs membership/learnable_readout/joint (环条件化增益)
  - Stage3 内: direct_supervised vs random_mask_pretrain/ring_mask_pretrain (掩码预训练增益)
  - Stage4 内: base vs ring_conditioned (跨架构环条件化增益, 每个 backbone)
  - 跨阶段: Stage1 GNN 最优 vs Stage2 最优 vs Stage3 最优 vs Stage4 最优

输入: results/{stage1_traditional_ml,stage1_gnn,stage2_ring_conditioning,
              stage3_mask_pretraining,stage4_cross_arch}/per_seed_results.csv
输出: results/significance_tests.csv

注意: CSV 统一使用 'config' 列 (替代原 'encoding' 列)。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# 将 last_end_code 根目录加入 sys.path, 以便 import common.constants
LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import PER_SEED_COLUMNS  # noqa: F401  (统一列名校验)

RESULTS_ROOT = os.path.join(LAST_END_ROOT, 'results')

# 各阶段目录名
STAGE1_ML_DIR = 'stage1_traditional_ml'
STAGE1_GNN_DIR = 'stage1_gnn'
STAGE2_DIR = 'stage2_ring_conditioning'
STAGE3_DIR = 'stage3_mask_pretraining'
STAGE4_DIR = 'stage4_cross_arch'


def paired_test(a, b, alpha=0.05):
    """配对检验: paired t-test + Wilcoxon signed-rank test

    Returns: dict with n_seeds, mean_a, mean_b, diff_mean, diff_std,
             t_stat, t_pvalue, w_stat, w_pvalue, significant
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(a)
    assert len(b) == n, f"长度不一致: {len(a)} vs {len(b)}"

    result = {'n_seeds': n, 'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
              'diff_mean': float(np.mean(a - b)),
              'diff_std': float(np.std(a - b, ddof=1)) if n > 1 else 0.0}

    # paired t-test
    try:
        t_stat, t_p = stats.ttest_rel(a, b)
        result['t_stat'] = float(t_stat)
        result['t_pvalue'] = float(t_p)
    except Exception:
        result['t_stat'] = float('nan')
        result['t_pvalue'] = float('nan')

    # Wilcoxon signed-rank test (需要 n >= 5, 且 a != b)
    try:
        if n >= 5 and not np.all(a == b):
            w_stat, w_p = stats.wilcoxon(a, b)
            result['w_stat'] = float(w_stat)
            result['w_pvalue'] = float(w_p)
        else:
            result['w_stat'] = float('nan')
            result['w_pvalue'] = float('nan')
    except Exception:
        result['w_stat'] = float('nan')
        result['w_pvalue'] = float('nan')

    # 显著性判断 (以 Wilcoxon 为主, 小样本更稳健)
    result['significant'] = (result['w_pvalue'] < alpha
                             if not np.isnan(result['w_pvalue']) else False)
    return result


def run_within_stage_tests(df, stage_name, baseline_config, compare_configs,
                           metric='test_r2'):
    """阶段内对比: baseline_config vs 各 compare_config (按 seed 配对)

    分组键: task [+ model]. 对每个 (task, model) 子组, 按 seed 对齐 baseline 与 compare。
    """
    results = []
    if df.empty or 'config' not in df.columns:
        return results

    # 分组维度: task + model (若存在)
    group_cols = ['task', 'model'] if 'model' in df.columns else ['task']

    for task in df['task'].unique():
        for model in (df['model'].unique() if 'model' in df.columns else [None]):
            if model is None:
                sub = df[df['task'] == task]
            else:
                sub = df[(df['task'] == task) & (df['model'] == model)]
            if len(sub) < 2:
                continue

            # 按 seed 对齐各 config 的 metric
            pivot = sub.pivot_table(index='seed', columns='config',
                                    values=metric, aggfunc='first')
            if baseline_config not in pivot.columns:
                continue
            base_vals = pivot[baseline_config].dropna()
            if len(base_vals) < 3:
                continue

            for cmp_cfg in compare_configs:
                if cmp_cfg not in pivot.columns:
                    continue
                cmp_vals = pivot[cmp_cfg].dropna()
                common_seeds = base_vals.index.intersection(cmp_vals.index)
                if len(common_seeds) < 3:
                    continue
                a = base_vals.loc[common_seeds].values
                b = cmp_vals.loc[common_seeds].values
                res = paired_test(a, b)
                res.update({'test_type': f'{stage_name}_within', 'task': task,
                            'model': model if model is not None else '',
                            'baseline': baseline_config, 'compare': cmp_cfg,
                            'metric': metric})
                results.append(res)
    return results


def run_stage2_tests(df):
    """Stage2 内部对比: base vs membership/learnable_readout/joint"""
    return run_within_stage_tests(df, 'stage2', 'base',
                                  ['membership', 'learnable_readout', 'joint'])


def run_stage3_tests(df):
    """Stage3 内部对比: direct_supervised vs random_mask_pretrain/ring_mask_pretrain"""
    return run_within_stage_tests(df, 'stage3', 'direct_supervised',
                                  ['random_mask_pretrain', 'ring_mask_pretrain'])


def run_stage4_tests(df):
    """Stage4 内部对比: base vs ring_conditioned (每个 backbone)"""
    return run_within_stage_tests(df, 'stage4', 'base', ['ring_conditioned'])


def _best_per_task(df, metric='test_r2'):
    """对每个 task 选最优 (task, model, config) 组合, 返回 {task: (sort by seed 的 metric 值)}"""
    best = {}
    if df.empty:
        return best
    for task in df['task'].unique():
        sub = df[df['task'] == task]
        # 按 (model, config) 分组取均值, 选最优组合
        group_cols = [c for c in ['model', 'config'] if c in sub.columns]
        if not group_cols:
            continue
        agg = sub.groupby(group_cols)[metric].mean()
        best_key = agg.idxmax()
        if not isinstance(best_key, tuple):
            best_key = (best_key,)
        mask = np.ones(len(sub), dtype=bool)
        for col, val in zip(group_cols, best_key):
            mask &= (sub[col] == val).values
        best_sub = sub[mask].sort_values('seed')
        best[task] = best_sub[metric].values
    return best


def run_cross_stage_tests(stage_dfs, metric='test_r2'):
    """跨阶段对比: 每个任务各阶段最优组合的配对检验

    stage_dfs: dict {stage_name: DataFrame}
    """
    results = []
    # 各阶段每个 task 的最优 metric 序列
    per_stage_best = {name: _best_per_task(df, metric) for name, df in stage_dfs.items()}

    # 收集所有任务
    all_tasks = set()
    for d in per_stage_best.values():
        all_tasks.update(d.keys())
    stage_names = list(stage_dfs.keys())

    for task in sorted(all_tasks):
        # 取各阶段该 task 的最优序列
        series = {}
        for name in stage_names:
            if task in per_stage_best[name]:
                series[name] = per_stage_best[name][task]
        if len(series) < 2:
            continue

        # 相邻阶段两两对比
        for i in range(len(stage_names) - 1):
            s1, s2 = stage_names[i], stage_names[i + 1]
            if s1 not in series or s2 not in series:
                continue
            a, b = series[s1], series[s2]
            n = min(len(a), len(b))
            if n < 3:
                print(f"  {task}: {s1} vs {s2} 种子数不足 ({n}), 跳过")
                continue
            res = paired_test(a[:n], b[:n])
            res.update({'test_type': 'cross_stage', 'task': task, 'model': '',
                        'baseline': s1, 'compare': s2, 'metric': metric})
            results.append(res)
    return results


def _load_stage_csv(results_root, stage_dir):
    """加载某阶段的 per_seed_results.csv, 不存在返回空 DataFrame"""
    csv_path = os.path.join(results_root, stage_dir, 'per_seed_results.csv')
    if not os.path.exists(csv_path):
        print(f"  [跳过] 未找到 {csv_path}")
        return pd.DataFrame()
    print(f"  [加载] {csv_path}")
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(description='统计显著性检验 (四阶段版)')
    parser.add_argument('--results_root', type=str, default=RESULTS_ROOT,
                        help='结果根目录 (默认 last_end_code/results)')
    parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平')
    args = parser.parse_args()

    print(f"结果根目录: {args.results_root}")
    print(f"显著性水平: alpha={args.alpha}")

    # 加载各阶段数据
    df_s1ml  = _load_stage_csv(args.results_root, STAGE1_ML_DIR)
    df_s1gnn = _load_stage_csv(args.results_root, STAGE1_GNN_DIR)
    df_s2    = _load_stage_csv(args.results_root, STAGE2_DIR)
    df_s3    = _load_stage_csv(args.results_root, STAGE3_DIR)
    df_s4    = _load_stage_csv(args.results_root, STAGE4_DIR)

    all_results = []

    # ----- Stage2 内部对比 -----
    if not df_s2.empty:
        print("\n" + "=" * 60)
        print("Stage2 内部对比 (base vs membership/learnable_readout/joint)")
        print("=" * 60)
        s2_tests = run_stage2_tests(df_s2)
        all_results.extend(s2_tests)
        for r in s2_tests:
            sig = "***" if r.get('significant') else ""
            print(f"  {r['task']:10s} | {str(r['model']):12s} | base vs {r['compare']:18s} | "
                  f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
                  f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # ----- Stage3 内部对比 -----
    if not df_s3.empty:
        print("\n" + "=" * 60)
        print("Stage3 内部对比 (direct_supervised vs random/ring_mask_pretrain)")
        print("=" * 60)
        s3_tests = run_stage3_tests(df_s3)
        all_results.extend(s3_tests)
        for r in s3_tests:
            sig = "***" if r.get('significant') else ""
            print(f"  {r['task']:10s} | {str(r['model']):12s} | direct vs {r['compare']:22s} | "
                  f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
                  f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # ----- Stage4 内部对比 -----
    if not df_s4.empty:
        print("\n" + "=" * 60)
        print("Stage4 内部对比 (base vs ring_conditioned, 每个 backbone)")
        print("=" * 60)
        s4_tests = run_stage4_tests(df_s4)
        all_results.extend(s4_tests)
        for r in s4_tests:
            sig = "***" if r.get('significant') else ""
            print(f"  {r['task']:10s} | {str(r['model']):12s} | base vs {r['compare']:16s} | "
                  f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
                  f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # ----- 跨阶段对比 -----
    cross_dfs = {}
    if not df_s1gnn.empty:
        cross_dfs['stage1_gnn'] = df_s1gnn
    if not df_s2.empty:
        cross_dfs['stage2'] = df_s2
    if not df_s3.empty:
        cross_dfs['stage3'] = df_s3
    if not df_s4.empty:
        cross_dfs['stage4'] = df_s4

    if len(cross_dfs) >= 2:
        print("\n" + "=" * 60)
        print("跨阶段对比 (各阶段最优组合, 按 seed 配对)")
        print("=" * 60)
        cross_tests = run_cross_stage_tests(cross_dfs)
        all_results.extend(cross_tests)
        for r in cross_tests:
            sig = "***" if r.get('significant') else ""
            print(f"  {r['task']:10s} | {r['baseline']:12s} vs {r['compare']:12s} | "
                  f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
                  f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # ----- 保存 -----
    if all_results:
        df_out = pd.DataFrame(all_results)
        out_csv = os.path.join(args.results_root, 'significance_tests.csv')
        df_out.to_csv(out_csv, index=False)
        print(f"\n结果保存: {out_csv} ({len(df_out)} 条)")

        sig_count = int(df_out['significant'].sum())
        total = len(df_out)
        print(f"显著性汇总: {sig_count}/{total} 个对比在 alpha={args.alpha} 下显著")
    else:
        print("\n无可检验的对比 (请确认各阶段 per_seed_results.csv 已生成)")


if __name__ == '__main__':
    main()
