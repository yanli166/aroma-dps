
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
生成 Experiment A vs B 对比图表

Exp A: Progressive residual OOD (测试集随 fraction 变化)
Exp B: Fixed OOD benchmark (测试集固定为 20 种 ring types)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'ood_difficulty_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 Exp A (learning curve)
df_a = pd.read_csv(os.path.join(RESULTS_DIR, 'learning_curve', 'learning_curve_full.csv'))

# 加载 Exp B (fixed OOD)
df_b = pd.read_csv(os.path.join(RESULTS_DIR, 'fixed_ood_benchmark', 'fixed_ood_benchmark.csv'))

tasks = ['HOMA', 'NICS_1zz', 'MBCO']
task_labels = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS(1)zz', 'MBCO': 'MBCO'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=120)

for i, task in enumerate(tasks):
    ax = axes[i]
    sub_a = df_a[df_a['task'] == task].sort_values('fraction')
    sub_b = df_b[df_b['task'] == task].sort_values('fraction')

    ax.plot(sub_a['fraction'] * 100, sub_a['test_r2'], 'o-',
            linewidth=2.5, markersize=9, color='steelblue', label='Exp A: Residual OOD')
    ax.plot(sub_b['fraction'] * 100, sub_b['test_r2'], 's--',
            linewidth=2.5, markersize=9, color='coral', label='Exp B: Fixed OOD')

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Fraction of Ring Types in Training (%)', fontsize=11)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_title(task_labels[task], fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.0, 1.0)

    # 标注 70% 点
    for _, row in sub_a.iterrows():
        if row['fraction'] == 0.7:
            ax.annotate(f"{row['test_r2']:.3f}",
                        (70, row['test_r2']),
                        textcoords="offset points", xytext=(10, -15),
                        fontsize=9, color='steelblue', fontweight='bold')
    for _, row in sub_b.iterrows():
        if row['fraction'] == 0.7:
            ax.annotate(f"{row['test_r2']:.3f}",
                        (70, row['test_r2']),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=9, color='coral', fontweight='bold')

plt.suptitle('Experiment A vs B: Ring-Type Coverage vs Generalization',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'exp_a_vs_b_comparison.png')
plt.savefig(fig_path, bbox_inches='tight')
plt.close()
print(f"保存: {fig_path}")

# 打印对比表
print("\n" + "=" * 80)
print("Experiment A vs B 对比")
print("=" * 80)
for task in tasks:
    print(f"\n--- {task_labels[task]} ---")
    sub_a = df_a[df_a['task'] == task].sort_values('fraction')
    sub_b = df_b[df_b['task'] == task].sort_values('fraction')
    print(f"{'Fraction':>10} {'Exp A R²':>10} {'Exp B R²':>10} {'Diff (B-A)':>12}")
    for frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        ra = sub_a[sub_a['fraction'] == frac]['test_r2'].values
        rb = sub_b[sub_b['fraction'] == frac]['test_r2'].values
        if len(ra) > 0 and len(rb) > 0:
            diff = rb[0] - ra[0]
            print(f"{frac*100:>9.0f}% {ra[0]:>10.4f} {rb[0]:>10.4f} {diff:>+12.4f}")
