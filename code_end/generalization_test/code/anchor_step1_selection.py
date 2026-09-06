
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
Anchor-based Δ-Learning Step 1: Anchor 选择

预注册选择规则 (在任何模型结果之前确定):

候选池: 所有 30 个取代基 (覆盖率均 ≥ 80%).

排序规则 (按优先级):
  1. 结构简单度 (atom count, 越小越优先)
  2. 电子中性 (排除极端 EDG/EWG)
  3. 覆盖率 (>= 80% 即可, 不强求 100%)

排除列表 (极端电子或空间效应):
  - 强 EDG: NMe2, NH2, OH (强供电子, H-bond donor)
  - 强 EWG: NO2, CN, SO2CF3, SMeO2, CF3, OCF3
  - 大体积: SiMe3, B(OH)2 (大空间位阻, 非典型取代基)
  - 杂原子特殊: SH (S-H, 不稳定), NCO, NHCHO, NHCOMe (酰胺类, 多构象)
  - 可变基团: OAc (酯, 可水解), COOH (可解离), COOMe (酯)
  - 反应活性: CCH (炔基, 可参与 click), vinyl (可聚合), aziridinyl (三元环, 应变)
  - CONH2 (酰胺, 可互变异构), COMe (酮, 可烯醇化)
  - SMe (硫醚, 可氧化), CH2OH (伯醇, 可氧化)
  - Et, I (Et 体积偏大; I 重原子, 极化率高)

候选池中"电子中性 + 结构简单"的剩余取代基:
  - F: 1 个原子, 最小, 近 H 等排, 弱 EDG/EWG, σ_p = +0.06 (近似中性)
  - Cl: 1 个原子, 小, 弱 EWG, σ_p = +0.23
  - OMe: 含 O, 中等大小, 弱 EDG, σ_p = -0.27

预注册选择:
  - Primary anchor: F (氟)
  - Backup anchor 1: Cl (氯) — 仍是小体积卤素, 电子效应略偏 EWG
  - Backup anchor 2: OMe (甲氧基) — 中等大小, 弱 EDG, 提供 N→O 对照

锚点组合: [F, Cl, OMe]
分别构建 anchor pair dataset 并报告各自结果.

选择理由 (pre-registered):
  - F 是最小取代基, 近 H 等排体, 电子扰动最小, 作为"准 reference state"
  - Cl 提供同族 (卤素) 但更大的对照, 测试 anchor 选择敏感性
  - OMe 提供不同杂原子 (O vs F) 的对照, 测试供电子/吸电子方向敏感性
"""
import os
import sys
import pandas as pd
import numpy as np

PROJ_ROOT = '_PROJ_ROOT'
LUNCI10_BEGIN = os.path.join(PROJ_ROOT, 'lunci10/lunci10-begin.csv')
LUNCI10_TEST = os.path.join(PROJ_ROOT, 'code_end/data1_end/lunci10-test.csv')

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'code_end/results/lunci10_anchor_delta_final')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预注册的 anchor 选择 (在任何模型结果之前确定)
ANCHORS = ['F', 'Cl', 'OMe']

# 排除的取代基 (极端电子/空间效应)
EXCLUDED = [
    'NMe2', 'NH2', 'OH',           # 强 EDG
    'NO2', 'CN', 'SO2CF3', 'SMeO2', 'CF3', 'OCF3',  # 强 EWG
    'SiMe3', 'B(OH)2',              # 大体积
    'SH', 'NCO', 'NHCHO', 'NHCOMe', # 杂原子特殊
    'OAc', 'COOH', 'COOMe',         # 可变基团
    'CCH', 'vinyl', 'aziridinyl',   # 反应活性
    'CONH2', 'COMe', 'SMe', 'CH2OH', 'Et', 'I',  # 其他
]


def analyze_coverage():
    """分析所有取代基的覆盖率"""
    df = pd.read_csv(LUNCI10_BEGIN, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    group_cols = ['ring_name', 'ring_pos']
    n_total_groups = df.groupby(group_cols).ngroups

    rows = []
    for sub in sorted(df['sub_name'].unique()):
        sub_df = df[df['sub_name'] == sub]
        covered = sub_df.groupby(group_cols).ngroups
        rows.append({
            'sub_name': sub,
            'n_groups_covered': covered,
            'coverage_ratio': covered / n_total_groups,
            'n_total_molecules': len(sub_df),
            'is_excluded': sub in EXCLUDED,
            'exclusion_reason': 'extreme electronics/size/reactivity' if sub in EXCLUDED else '',
        })

    df_cov = pd.DataFrame(rows).sort_values(['n_groups_covered', 'sub_name'], ascending=[False, True])
    df_cov['is_anchor'] = df_cov['sub_name'].isin(ANCHORS)
    df_cov.to_csv(os.path.join(OUTPUT_DIR, 'anchor_selection.csv'), index=False)

    print("=" * 70)
    print("Anchor 选择分析")
    print("=" * 70)
    print(f"Total (ring_name, ring_pos) groups: {n_total_groups}")
    print(f"Total substituents: {len(df_cov)}")
    print()
    print(f"Pre-registered anchors: {ANCHORS}")
    print(f"Excluded substituents ({len(EXCLUDED)}): {EXCLUDED}")
    print()
    print("Candidate pool (not excluded):")
    candidates = df_cov[~df_cov['is_excluded']].sort_values('sub_name')
    print(candidates[['sub_name', 'n_groups_covered', 'coverage_ratio', 'n_total_molecules']].to_string(index=False))
    print()
    print("Selected anchors:")
    for a in ANCHORS:
        row = df_cov[df_cov['sub_name'] == a].iloc[0]
        print(f"  {a}: coverage={row['n_groups_covered']}/{n_total_groups} ({row['coverage_ratio']:.0%}), n_molecules={row['n_total_molecules']}")
    return df_cov


if __name__ == '__main__':
    analyze_coverage()
    print()
    print(f"Output: {os.path.join(OUTPUT_DIR, 'anchor_selection.csv')}")
