"""从各方法子目录的 summary.csv 重建 layer4_summary.csv 和对比表"""
import os
import glob
import pandas as pd
import numpy as np


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

RESULTS_ROOT = '_PROJ_ROOT + "/code_end"/results/layer4_substituent'
LAYER2_CSV = '_PROJ_ROOT + "/code_end"/results/layer2_gnn/per_seed_results.csv'

# 收集所有 summary.csv
rows = []
for summary_path in sorted(glob.glob(os.path.join(RESULTS_ROOT, '*', '*', 'summary.csv'))):
    # 路径形如 .../layer4_substituent/HOMA/m1/summary.csv
    parts = summary_path.split(os.sep)
    task = parts[-3]
    method = parts[-2]
    df = pd.read_csv(summary_path)
    d = dict(zip(df['metric'], df['value']))
    row = {
        'task': task,
        'method': method,
        'n': int(d.get('n', 0)),
        'cv_r2': d.get('cv_r2_mean', np.nan),
        'cv_r2_std': d.get('cv_r2_std', np.nan),
        'cv_mae': d.get('cv_mae', np.nan),
        'cv_rmse': d.get('cv_rmse', np.nan),
        'train_r2': d.get('train_r2', np.nan),
        'train_mae': d.get('train_mae', np.nan),
        'train_rmse': d.get('train_rmse', np.nan),
        'test_r2': d.get('test_r2', np.nan),
        'test_mae': d.get('test_mae', np.nan),
        'test_rmse': d.get('test_rmse', np.nan),
        'lunci6_r2': d.get('lunci6_r2', np.nan),
        'lunci6_mae': d.get('lunci6_mae', np.nan),
        'lunci6_rmse': d.get('lunci6_rmse', np.nan),
        'train_time_sec': d.get('train_time_sec', np.nan),
    }
    rows.append(row)

df_all = pd.DataFrame(rows)
# 按 task, method 排序
method_order = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
task_order = ['HOMA', 'NICS_1zz', 'MBCO']
df_all['task'] = pd.Categorical(df_all['task'], categories=task_order, ordered=True)
df_all['method'] = pd.Categorical(df_all['method'], categories=method_order, ordered=True)
df_all = df_all.sort_values(['task', 'method']).reset_index(drop=True)
df_all['task'] = df_all['task'].astype(str)
df_all['method'] = df_all['method'].astype(str)

out_csv = os.path.join(RESULTS_ROOT, 'layer4_summary.csv')
df_all.to_csv(out_csv, index=False)
print(f"已重建: {out_csv} ({len(df_all)} 行)")
print(df_all[['task', 'method', 'cv_r2', 'test_r2', 'lunci6_r2']].to_string(index=False))

# ============== 生成对比表 (vs Layer 2 GNN) ==============
if os.path.exists(LAYER2_CSV):
    df_l2 = pd.read_csv(LAYER2_CSV)
    # per_seed_results.csv 有 test_r2 列; all_layer2_gnn_summary.csv 用 test_r2_mean
    test_col = 'test_r2' if 'test_r2' in df_l2.columns else 'test_r2_mean'
    # 取 seed=42 的 GNN 作为 baseline (与 Layer 4 一致)
    if 'seed' in df_l2.columns:
        df_l2_gnn = df_l2[(df_l2['model'] == 'GNN') & (df_l2['seed'] == 42)].set_index('task')
    else:
        df_l2_gnn = df_l2[df_l2['model'] == 'GNN'].set_index('task')
    baseline_map = df_l2_gnn[test_col].to_dict()
    print(f"\nLayer 2 GNN baseline {test_col}: {baseline_map}")

    cmp_rows = []
    for _, r in df_all.iterrows():
        baseline = baseline_map.get(r['task'], np.nan)
        test_r2_val = float(r['test_r2'])
        cmp_rows.append({
            'task': r['task'],
            'method': r['method'],
            'test_r2': test_r2_val,
            'baseline_test_r2': float(baseline) if not pd.isna(baseline) else np.nan,
            'delta_test_r2': test_r2_val - float(baseline) if not pd.isna(baseline) else np.nan,
            'lunci6_r2': float(r['lunci6_r2']),
            'cv_r2': float(r['cv_r2']),
        })
    df_cmp = pd.DataFrame(cmp_rows)
    cmp_csv = os.path.join(RESULTS_ROOT, 'layer4_vs_layer2_comparison.csv')
    df_cmp.to_csv(cmp_csv, index=False)
    print(f"\n对比表已保存: {cmp_csv}")
    for task in task_order:
        sub = df_cmp[df_cmp['task'] == task].sort_values('test_r2', ascending=False)
        print(f"\n--- {task} ---")
        print(sub[['method', 'test_r2', 'baseline_test_r2', 'delta_test_r2', 'lunci6_r2']].to_string(index=False))
else:
    print(f"警告: 未找到 Layer 2 结果文件 {LAYER2_CSV}")
