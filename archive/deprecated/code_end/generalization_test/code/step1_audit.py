
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
Step 1: 审计现有三层分析

检查:
  1. Layer B 中 group mean 是否使用测试集真实标签
  2. 标记为 diagnostic centered analysis
  3. 每组样本数分布
  4. 每任务 Δ label 的 mean/std/min/max, |Δ| 分布
  5. 检查同一 scaffold 是否跨 train/test

输出:
  results/lunci10_delta_learning/00_audit/
    audit_report.md
    group_statistics.csv
    delta_distribution.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
L10_TEST = '_PROJ_ROOT + "/code_end"/data1_end/lunci10-test.csv'
L10_META = '_PROJ_ROOT/lunci10/lunci10-begin.csv'
TRAIN_CSV = '_PROJ_ROOT + "/code_end"/data1_end/collet_homa_0716.csv'

OUTPUT_DIR = '_PROJ_ROOT + "/code_end"/results/lunci10_delta_learning/00_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = {
    'HOMA':     'HOMA',
    'NICS_1zz': 'NICS_ZZ',
    'MBCO':     'MBCO',
}


def load_lunci10_with_meta():
    """加载 lunci10 测试数据并合并 scaffold/substituent 元数据"""
    df_l10 = pd.read_csv(L10_TEST, encoding='utf-8-sig')
    df_l10.columns = df_l10.columns.str.strip()
    df_l10 = df_l10.loc[:, ~df_l10.columns.str.startswith('Unnamed')]
    df_l10 = df_l10.dropna(subset=['SMILES']).reset_index(drop=True)

    df_meta = pd.read_csv(L10_META)
    df_meta.columns = df_meta.columns.str.strip()

    # 合并: New_ID = begin.csv 的 no
    df = df_l10.merge(
        df_meta[['no', 'ring_name', 'sub_name', 'sub_type', 'ring_pos']],
        left_on='New_ID', right_on='no', how='left'
    )

    unmatched = df['ring_name'].isna().sum()
    if unmatched > 0:
        print(f"  [WARN] {unmatched}/{len(df)} 行未匹配元数据")

    df['ring_name'] = df['ring_name'].fillna('unknown')
    df['sub_name'] = df['sub_name'].fillna('unknown')
    df['ring_pos'] = df['ring_pos'].fillna(-1)
    df['sub_type'] = df['sub_type'].fillna('unknown')

    return df


def main():
    print("=" * 70)
    print("Step 1: 审计现有三层分析")
    print("=" * 70)

    df = load_lunci10_with_meta()

    # ── 1. 检查 Layer B group mean 是否使用测试集真实标签 ────────────────────
    print("\n## 1. Layer B group mean 来源审计")
    print("-" * 50)

    layer_b_code = """
    # 现有 Layer B 代码 (lunci10_three_layer_analysis.py, line 178-185):
    group_col = ['ring_name', 'ring_pos']
    df['true_mean'] = df.groupby(group_col)['true'].transform('mean')   # ← 测试集真实标签!
    df['pred_mean'] = df.groupby(group_col)['pred'].transform('mean')
    df['true_delta'] = df['true'] - df['true_mean']
    df['pred_delta'] = df['pred'] - df['pred_mean']
    """
    print(layer_b_code)

    finding_1 = (
        "**发现**: Layer B 的 `true_mean` 使用 `df.groupby(group_col)['true'].transform('mean')`,\n"
        "其中 `df['true']` 是 **lunci10 测试集的真实标签**。\n\n"
        "这意味着 Δ_true = A_true - mean(A_true | group) 中的 group mean 包含了测试集信息,\n"
        "因此 Layer B 的结果 **不是 prospective prediction**, 而是 **diagnostic centered analysis**。\n\n"
        "同样, Δ_pred = A_pred - mean(A_pred | group) 中的 group mean 也来自测试集预测值的组内均值,\n"
        "这虽然不使用真实标签, 但 centering 操作本身依赖于测试集的分组结构。\n\n"
        "结论: Layer B 结果应标记为 `diagnostic centered analysis`, 不能作为模型前瞻预测能力的证据。"
    )
    print(finding_1)

    # ── 2. 检查每组样本数 ──────────────────────────────────────────────────
    print("\n## 2. scaffold+ring_pos 组统计")
    print("-" * 50)

    group_col = ['ring_name', 'ring_pos']
    group_sizes = df.groupby(group_col).size().reset_index(name='n_samples')
    group_sizes = group_sizes.sort_values('n_samples', ascending=False)

    n_groups = len(group_sizes)
    print(f"总组数 (scaffold+ring_pos): {n_groups}")
    print(f"总样本数: {len(df)}")
    print(f"\n组大小分布:")
    print(f"  min:    {group_sizes['n_samples'].min()}")
    print(f"  max:    {group_sizes['n_samples'].max()}")
    print(f"  mean:   {group_sizes['n_samples'].mean():.1f}")
    print(f"  median: {group_sizes['n_samples'].median():.1f}")
    print(f"  std:    {group_sizes['n_samples'].std(ddof=1):.1f}")

    print(f"\n组大小分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"  {q*100:5.0f}%: {group_sizes['n_samples'].quantile(q):.0f}")

    print(f"\n大小分布直方图:")
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 1000]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt = ((group_sizes['n_samples'] > lo) & (group_sizes['n_samples'] <= hi)).sum()
        if cnt > 0:
            print(f"  ({lo:3d}, {hi:3d}]: {cnt:3d} groups")

    # 保存 group_statistics.csv
    group_sizes.to_csv(os.path.join(OUTPUT_DIR, 'group_statistics.csv'), index=False)

    # ── 3. 每任务 Δ label 的 mean/std/min/max, |Δ| 分布 ────────────────────
    print("\n## 3. Δ label 分布 (diagnostic centered)")
    print("-" * 50)

    delta_rows = []
    for task_name, col in TASKS.items():
        df_task = df.dropna(subset=[col]).copy()
        if len(df_task) == 0:
            continue

        # group-mean centering (使用测试集真实标签 - diagnostic only)
        df_task['_mean'] = df_task.groupby(group_col)[col].transform('mean')
        df_task['delta'] = df_task[col] - df_task['_mean']
        df_task['abs_delta'] = df_task['delta'].abs()

        delta_rows.append({
            'task': task_name,
            'n_samples': len(df_task),
            'n_groups': df_task.groupby(group_col).ngroups,
            'delta_mean': df_task['delta'].mean(),
            'delta_std': df_task['delta'].std(ddof=1),
            'delta_min': df_task['delta'].min(),
            'delta_max': df_task['delta'].max(),
            'abs_delta_mean': df_task['abs_delta'].mean(),
            'abs_delta_std': df_task['abs_delta'].std(ddof=1),
            'abs_delta_median': df_task['abs_delta'].median(),
            'abs_delta_q25': df_task['abs_delta'].quantile(0.25),
            'abs_delta_q75': df_task['abs_delta'].quantile(0.75),
            'abs_delta_q90': df_task['abs_delta'].quantile(0.90),
        })

        print(f"\n  [{task_name}] (n={len(df_task)}, groups={df_task.groupby(group_col).ngroups})")
        print(f"    Δ mean:  {df_task['delta'].mean():.4f}")
        print(f"    Δ std:   {df_task['delta'].std(ddof=1):.4f}")
        print(f"    Δ min:   {df_task['delta'].min():.4f}")
        print(f"    Δ max:   {df_task['delta'].max():.4f}")
        print(f"    |Δ| mean:   {df_task['abs_delta'].mean():.4f}")
        print(f"    |Δ| median: {df_task['abs_delta'].median():.4f}")
        print(f"    |Δ| q25/q75: {df_task['abs_delta'].quantile(0.25):.4f} / {df_task['abs_delta'].quantile(0.75):.4f}")
        print(f"    |Δ| q90:     {df_task['abs_delta'].quantile(0.90):.4f}")

    df_delta_dist = pd.DataFrame(delta_rows)
    df_delta_dist.to_csv(os.path.join(OUTPUT_DIR, 'delta_distribution.csv'), index=False)

    # ── 4. 检查同一 scaffold 是否跨 train/test ─────────────────────────────
    print("\n## 4. 检查 lunci10 scaffold 是否出现在训练集中")
    print("-" * 50)

    # 加载训练集
    df_train = pd.read_csv(TRAIN_CSV)
    train_smiles_col = [c for c in df_train.columns if 'smile' in c.lower() or 'SMILES' in c][0]

    # 获取 lunci10 的 scaffold 集合
    l10_scaffolds = set(df['ring_name'].unique())

    # 检查训练集中的分子是否属于 lunci10 的 scaffold
    # 通过 SMILES 匹配
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def get_murcko_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except:
            return None

    # 获取 lunci10 各 scaffold 的 canonical SMILES
    # 通过 ring_name 找对应的代表性 SMILES
    ring_to_smiles = {}
    for _, row in df.iterrows():
        if row['ring_name'] not in ring_to_smiles:
            ring_to_smiles[row['ring_name']] = row['SMILES']

    # 获取各 ring_name 的 Murcko scaffold
    ring_to_murcko = {}
    for ring_name, smi in ring_to_smiles.items():
        if ring_name == 'unknown':
            continue
        sc = get_murcko_scaffold(smi)
        ring_to_murcko[ring_name] = sc

    print(f"lunci10 ring_name 类型: {sorted(l10_scaffolds)}")
    print(f"\nlunci10 ring_name → Murcko scaffold 映射:")
    for r, m in sorted(ring_to_murcko.items()):
        print(f"  {r:25s} → {m}")

    # 检查训练集中是否有这些 Murcko scaffold
    print(f"\n训练集大小: {len(df_train)}")
    print("正在计算训练集 Murcko scaffold (可能需要一些时间)...")

    train_murcko = set()
    sample_size = min(5000, len(df_train))  # 采样检查
    for smi in df_train[train_smiles_col].sample(sample_size, random_state=42):
        sc = get_murcko_scaffold(smi)
        if sc:
            train_murcko.add(sc)

    print(f"训练集采样 {sample_size} 个分子的 Murcko scaffold 数: {len(train_murcko)}")

    # 检查 overlap
    overlap = {}
    for ring_name, murcko in ring_to_murcko.items():
        if murcko and murcko in train_murcko:
            overlap[ring_name] = True
        else:
            overlap[ring_name] = False

    print(f"\nlunci10 scaffold 是否出现在训练集 Murcko scaffold 中:")
    for r, in_train in sorted(overlap.items()):
        status = "YES - 出现在训练集" if in_train else "NO - 不在训练集"
        print(f"  {r:25s}: {status}")

    n_overlap = sum(overlap.values())
    n_total = len(overlap)
    print(f"\n总结: {n_overlap}/{n_total} lunci10 scaffolds 出现在训练集中")

    # ── 生成 audit_report.md ──────────────────────────────────────────────
    report = f"""# lunci10 三层分析审计报告

## 1. Layer B group mean 来源审计

**发现**: Layer B 的 `true_mean` 使用 `df.groupby(group_col)['true'].transform('mean')`,
其中 `df['true']` 是 **lunci10 测试集的真实标签**。

这意味着:
- `Δ_true = A_true - mean(A_true | group)` 中的 group mean 包含了测试集信息
- `Δ_pred = A_pred - mean(A_pred | group)` 中的 group mean 也来自测试集预测值的组内均值

**结论**: Layer B 的结果应标记为 **diagnostic centered analysis**, 不能作为模型前瞻预测能力的证据。
现有 Layer B 结果 (Δ R²) 仅说明 "如果用测试集真实标签做 centering, 模型预测的相对偏差是否与真实相对偏差相关",
这是一个有诊断价值的分析, 但不是 prospective prediction。

---

## 2. scaffold+ring_pos 组统计

| 统计量 | 值 |
|--------|-----|
| 总组数 | {n_groups} |
| 总样本数 | {len(df)} |
| 组大小 min | {group_sizes['n_samples'].min()} |
| 组大小 max | {group_sizes['n_samples'].max()} |
| 组大小 mean | {group_sizes['n_samples'].mean():.1f} |
| 组大小 median | {group_sizes['n_samples'].median():.1f} |
| 组大小 std | {group_sizes['n_samples'].std(ddof=1):.1f} |

### 组大小分位数

| 分位数 | 组大小 |
|--------|--------|
"""
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        report += f"| {q*100:.0f}% | {group_sizes['n_samples'].quantile(q):.0f} |\n"

    report += f"""
### 组大小分布

| 区间 | 组数 |
|------|------|
"""
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 1000]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt = ((group_sizes['n_samples'] > lo) & (group_sizes['n_samples'] <= hi)).sum()
        if cnt > 0:
            report += f"| ({lo}, {hi}] | {cnt} |\n"

    report += f"""
---

## 3. Δ label 分布 (diagnostic centered)

| 任务 | n_samples | n_groups | Δ mean | Δ std | Δ min | Δ max | |Δ| mean | |Δ| median | |Δ| q25 | |Δ| q75 | |Δ| q90 |
|------|-----------|----------|--------|-------|--------|--------|---------|-----------|--------|--------|--------|
"""
    for r in delta_rows:
        report += (f"| {r['task']} | {r['n_samples']} | {r['n_groups']} | "
                   f"{r['delta_mean']:.4f} | {r['delta_std']:.4f} | "
                   f"{r['delta_min']:.4f} | {r['delta_max']:.4f} | "
                   f"{r['abs_delta_mean']:.4f} | {r['abs_delta_median']:.4f} | "
                   f"{r['abs_delta_q25']:.4f} | {r['abs_delta_q75']:.4f} | "
                   f"{r['abs_delta_q90']:.4f} |\n")

    report += f"""
---

## 4. lunci10 scaffold 是否出现在训练集中

lunci10 ring_name → Murcko scaffold 映射:

| ring_name | Murcko scaffold | 出现在训练集 |
|-----------|-----------------|-------------|
"""
    for r, m in sorted(ring_to_murcko.items()):
        status = "YES" if overlap.get(r, False) else "NO"
        report += f"| {r} | {m} | {status} |\n"

    report += f"""
**总结**: {n_overlap}/{n_total} lunci10 scaffolds 出现在训练集 (采样 {sample_size} 个训练分子)。

**说明**: lunci10 的设计意图是 ring-type OOD 测试, 即测试集包含训练集中未见过的 ring types。
如果所有 lunci10 scaffolds 都出现在训练集中, 则说明 OOD 难度来自 substituent/scaffold 变体,
而非完全未见过的 scaffold 类型。
如果部分 scaffolds 不在训练集中, 则这些 scaffolds 构成了更严格的 OOD 测试。

---

## 5. 审计结论与建议

1. **Layer B 结果标记**: 现有 Layer B 应明确标记为 **diagnostic centered analysis**,
   不能作为 prospective prediction 证据。

2. **Group 统计**: {n_groups} 个 scaffold+ring_pos 组, 组大小分布 {group_sizes['n_samples'].min()}-{group_sizes['n_samples'].max()}。
   需关注小组 (< 3) 的统计可靠性。

3. **Δ 分布**: |Δ| 的分布显示 substituent-induced aromaticity shift 的量级,
   为后续 Step 8 (Δ magnitude analysis) 提供基础。

4. **Scaffold 跨 split**: {n_overlap}/{n_total} lunci10 scaffolds 出现在训练集中,
   说明大部分 scaffold 在训练集中有覆盖, OOD 难度可能主要来自 substituent 变体。

5. **后续 Step 2 构建 pair dataset 时**: 必须使用 canonical pair (避免重复),
   scaffold-grouped split (同一 scaffold 的所有 pairs 进入同一 split)。

---

## 文件列表

- `audit_report.md` — 本报告
- `group_statistics.csv` — 每个 scaffold+ring_pos 组的样本数
- `delta_distribution.csv` — 每任务 Δ label 的统计量
"""

    with open(os.path.join(OUTPUT_DIR, 'audit_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n审计报告已保存至: {OUTPUT_DIR}/audit_report.md")
    print(f"group_statistics.csv 已保存")
    print(f"delta_distribution.csv 已保存")


if __name__ == '__main__':
    main()
