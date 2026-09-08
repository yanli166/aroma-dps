"""
M4: 统计显著性检验脚本

对三层多种子结果做配对统计检验:
  1. paired t-test (scipy.stats.ttest_rel)
  2. Wilcoxon signed-rank test (scipy.stats.wilcoxon) — 小样本非参数检验

检验对象:
  - Layer 3 内: label vs none (环编码增益), label vs mask/pool/combined (编码范式对比)
  - 跨层: Layer 1 最优 vs Layer 2 最优 vs Layer 3 最优

输入: code_end/results/{layer1_ml,layer2_gnn,layer3_ring}/per_seed_results.csv
输出: code_end/results/significance_tests.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(PROJ_ROOT, 'code_end', 'results')


def paired_test(a, b, alpha=0.05):
    """配对检验: paired t-test + Wilcoxon signed-rank test

    Returns: dict with t_stat, t_pvalue, w_stat, w_pvalue, significant
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(a)
    assert len(b) == n, f"长度不一致: {len(a)} vs {len(b)}"

    result = {'n_seeds': n, 'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
              'diff_mean': float(np.mean(a - b)), 'diff_std': float(np.std(a - b, ddof=1)) if n > 1 else 0.0}

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
    result['significant'] = result['w_pvalue'] < alpha if not np.isnan(result['w_pvalue']) else False
    return result


def run_layer3_tests(df_l3):
    """Layer 3 内部对比: label vs none, label vs mask/pool/combined"""
    results = []
    tasks = df_l3['task'].unique()
    models = df_l3['model'].unique()

    for task in tasks:
        for model in models:
            sub = df_l3[(df_l3['task'] == task) & (df_l3['model'] == model)]
            if len(sub) < 2:
                continue

            # 获取各编码的 test_r2 (按 seed 对齐)
            encodings = sub['encoding'].unique()
            pivot = sub.pivot_table(index='seed', columns='encoding', values='test_r2', aggfunc='first')

            # 以 label 为基准, 对比其他编码
            if 'label' not in pivot.columns:
                continue
            label_vals = pivot['label'].dropna()
            if len(label_vals) < 3:
                continue

            for enc in ['none', 'mask', 'pool', 'combined']:
                if enc not in pivot.columns:
                    continue
                enc_vals = pivot[enc].dropna()
                # 对齐 seed
                common_seeds = label_vals.index.intersection(enc_vals.index)
                if len(common_seeds) < 3:
                    continue
                a = label_vals.loc[common_seeds].values
                b = enc_vals.loc[common_seeds].values
                res = paired_test(a, b)
                res.update({'test_type': 'layer3_within', 'task': task, 'model': model,
                            'baseline': 'label', 'compare': enc,
                            'metric': 'test_r2'})
                results.append(res)

    return results


def run_cross_layer_tests(df_l1, df_l2, df_l3):
    """跨层对比: 每个任务的最优模型"""
    results = []

    for task in df_l1['task'].unique():
        # Layer 1 最优 (按 mean test_r2)
        l1_sub = df_l1[df_l1['task'] == task]
        l1_best_model = l1_sub.groupby('model')['test_r2'].mean().idxmax()
        l1_best = l1_sub[l1_sub['model'] == l1_best_model].sort_values('seed')['test_r2'].values

        # Layer 2 最优
        l2_sub = df_l2[df_l2['task'] == task]
        l2_best_model = l2_sub.groupby('model')['test_r2'].mean().idxmax()
        l2_best = l2_sub[l2_sub['model'] == l2_best_model].sort_values('seed')['test_r2'].values

        # Layer 3 最优 (按 model+encoding)
        l3_sub = df_l3[df_l3['task'] == task]
        l3_best_combo = l3_sub.groupby(['model', 'encoding'])['test_r2'].mean().idxmax()
        l3_best_model, l3_best_enc = l3_best_combo
        l3_best = l3_sub[(l3_sub['model'] == l3_best_model) &
                         (l3_sub['encoding'] == l3_best_enc)].sort_values('seed')['test_r2'].values

        # 配对检验 (需相同种子数)
        n = min(len(l1_best), len(l2_best), len(l3_best))
        if n < 3:
            print(f"  {task}: 种子数不足 ({n}), 跳过跨层检验")
            continue

        # L1 vs L2
        res = paired_test(l1_best[:n], l2_best[:n])
        res.update({'test_type': 'cross_layer', 'task': task,
                    'baseline': f'L1({l1_best_model})', 'compare': f'L2({l2_best_model})',
                    'metric': 'test_r2'})
        results.append(res)

        # L2 vs L3
        res = paired_test(l2_best[:n], l3_best[:n])
        res.update({'test_type': 'cross_layer', 'task': task,
                    'baseline': f'L2({l2_best_model})', 'compare': f'L3({l3_best_model}/{l3_best_enc})',
                    'metric': 'test_r2'})
        results.append(res)

        # L1 vs L3
        res = paired_test(l1_best[:n], l3_best[:n])
        res.update({'test_type': 'cross_layer', 'task': task,
                    'baseline': f'L1({l1_best_model})', 'compare': f'L3({l3_best_model}/{l3_best_enc})',
                    'metric': 'test_r2'})
        results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(description='M4: 统计显著性检验')
    parser.add_argument('--results_root', type=str, default=RESULTS_ROOT)
    parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平')
    args = parser.parse_args()

    # 加载数据
    l1_csv = os.path.join(args.results_root, 'layer1_ml', 'per_seed_results.csv')
    l2_csv = os.path.join(args.results_root, 'layer2_gnn', 'per_seed_results.csv')
    l3_csv = os.path.join(args.results_root, 'layer3_ring', 'per_seed_results.csv')

    if not os.path.exists(l3_csv):
        print(f"错误: 未找到 {l3_csv}")
        return

    df_l3 = pd.read_csv(l3_csv)
    df_l1 = pd.read_csv(l1_csv) if os.path.exists(l1_csv) else pd.DataFrame()
    df_l2 = pd.read_csv(l2_csv) if os.path.exists(l2_csv) else pd.DataFrame()

    all_results = []

    # Layer 3 内部对比
    print("=" * 60)
    print("Layer 3 内部对比 (label vs none/mask/pool/combined)")
    print("=" * 60)
    l3_tests = run_layer3_tests(df_l3)
    all_results.extend(l3_tests)
    for r in l3_tests:
        sig = "***" if r.get('significant') else ""
        print(f"  {r['task']:10s} | {r['model']:12s} | label vs {r['compare']:10s} | "
              f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
              f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # 跨层对比
    if not df_l1.empty and not df_l2.empty:
        print("\n" + "=" * 60)
        print("跨层对比 (L1最优 vs L2最优 vs L3最优)")
        print("=" * 60)
        cross_tests = run_cross_layer_tests(df_l1, df_l2, df_l3)
        all_results.extend(cross_tests)
        for r in cross_tests:
            sig = "***" if r.get('significant') else ""
            print(f"  {r['task']:10s} | {r['baseline']:20s} vs {r['compare']:30s} | "
                  f"diff={r['diff_mean']:+.4f}±{r['diff_std']:.4f} | "
                  f"t_p={r['t_pvalue']:.4f} w_p={r['w_pvalue']:.4f} {sig}")

    # 保存
    if all_results:
        df_out = pd.DataFrame(all_results)
        out_csv = os.path.join(args.results_root, 'significance_tests.csv')
        df_out.to_csv(out_csv, index=False)
        print(f"\n结果保存: {out_csv} ({len(df_out)} 条)")

        # 汇总
        sig_count = df_out['significant'].sum()
        total = len(df_out)
        print(f"显著性汇总: {sig_count}/{total} 个对比在 alpha={args.alpha} 下显著")


if __name__ == '__main__':
    main()
