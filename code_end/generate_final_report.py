"""
最终综合对比报告生成器

汇总所有层级实验结果, 生成 Markdown 报告:
  - Layer 1: 传统ML基线
  - Layer 2: GNN基线
  - Layer 3: 环编码范式消融 (修复后)
  - Layer 4: 取代基效应建模 (8种方法)
  - 芳香性 vs 非芳香拆分
  - Optuna 超参数优化
  - 泛化测试 (4种out-of-set)
"""
import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

warnings.filterwarnings('ignore')

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
RESULTS = os.path.join(PROJ_ROOT, 'results')
OUTPUT_MD = os.path.join(PROJ_ROOT, 'FINAL_REPORT.md')

TASKS = ['HOMA', 'NICS_1zz', 'MBCO']
TASK_DISPLAY = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS(1)zz', 'MBCO': 'MBCO'}


def safe_read_csv(path, **kw):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f'  WARN: failed to read {path}: {e}')
        return None


def fmt(v, dec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    try:
        return f'{float(v):.{dec}f}'
    except Exception:
        return str(v)


def fmt_pm(mean, std, dec=4):
    if (isinstance(mean, float) and np.isnan(mean)) or mean is None:
        return 'N/A'
    if (isinstance(std, float) and np.isnan(std)) or std is None:
        return fmt(mean, dec)
    return f'{float(mean):.{dec}f}±{float(std):.{dec}f}'


# ============== Layer 1 ==============
def collect_layer1():
    df = safe_read_csv(os.path.join(RESULTS, 'layer1_ml', 'all_layer1_ml_summary.csv'))
    if df is None:
        return None
    return df


def render_layer1(df):
    if df is None:
        return '## Layer 1: 传统ML基线\n\n数据未找到。\n'
    out = ['## Layer 1: 传统ML基线 (MACCS + Morgan指纹)\n']
    out.append('评估指标: 5折CV + 最终测试集 R²/MAE/RMSE\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('test_r2_mean', ascending=False)
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 排名 | 模型 | CV R² | Test R² | Test MAE | Test RMSE | 训练时间(s) |')
        out.append('|------|------|-------|---------|----------|-----------|------------|')
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            out.append(f'| {i} | {r["model"]} | {fmt_pm(r["cv_r2_mean"], r.get("cv_r2_std"))} | '
                       f'{fmt(r.get("test_r2_mean"))} | {fmt(r.get("test_mae_mean"))} | '
                       f'{fmt(r.get("test_rmse_mean"))} | {fmt(r.get("train_time_sec_mean"), 1)} |')
    return '\n'.join(out) + '\n'


# ============== Layer 2 ==============
def collect_layer2():
    df = safe_read_csv(os.path.join(RESULTS, 'layer2_gnn', 'all_layer2_gnn_summary.csv'))
    return df


def render_layer2(df):
    if df is None:
        return '## Layer 2: GNN基线\n\n数据未找到。\n'
    out = ['## Layer 2: GNN基线 (Ring Labeling, ring_flag=10)\n']
    out.append('评估指标: 5折CV + 最终测试集, 5个随机种子 (42,123,456,789,2024) 平均\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('test_r2_mean', ascending=False)
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 排名 | 模型 | CV R² | Test R² | Test MAE | Test RMSE | 训练时间(s) |')
        out.append('|------|------|-------|---------|----------|-----------|------------|')
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            out.append(f'| {i} | {r["model"]} | {fmt(r.get("cv_r2_mean"))} | '
                       f'{fmt(r.get("test_r2_mean"))} | {fmt(r.get("test_mae_mean"))} | '
                       f'{fmt(r.get("test_rmse_mean"))} | {fmt(r.get("train_time_sec_mean"), 1)} |')
    return '\n'.join(out) + '\n'


# ============== Layer 3 (refixed) ==============
def collect_layer3():
    """从 seed_*/TASK/MODEL_ENC/summary.csv 收集"""
    rows = []
    pattern = os.path.join(RESULTS, 'layer3_ring_fixed', 'seed_*', '*', '*', 'summary.csv')
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        # .../layer3_ring_fixed/seed_42/HOMA/GNN_none/summary.csv
        seed_str = parts[-4].replace('seed_', '')
        task = parts[-3]
        model_enc = parts[-2]
        if '_' in model_enc:
            model, enc = model_enc.rsplit('_', 1)
        else:
            model, enc = model_enc, 'unknown'
        df = safe_read_csv(path)
        if df is None:
            continue
        d = dict(zip(df['metric'], df['value']))
        # 强制转为 float (summary.csv 中 value 列为字符串)
        def _f(k):
            v = d.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')
        rows.append({
            'seed': int(seed_str), 'task': task, 'model': model, 'encoding': enc,
            'cv_r2': _f('cv_r2_mean'), 'test_r2': _f('test_r2'),
            'test_mae': _f('test_mae'), 'test_rmse': _f('test_rmse'),
            'train_time_sec': _f('train_time_sec'),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def render_layer3(df):
    if df is None:
        return '## Layer 3: 环编码范式消融 (修复初始化bug后)\n\n数据未找到或仍在运行。\n'
    out = ['## Layer 3: 环编码范式消融 (修复初始化bug后)\n']
    seeds = sorted(df['seed'].unique())
    out.append(f'已完成种子: {seeds} (共{len(seeds)}个)\n')
    out.append('编码类型: none(对照) / label(输入层) / mask(传播层) / pool(输出层) / combined(RA-GCN)\n')

    # 按 task/model/encoding 聚合
    for task in TASKS:
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 模型 | 编码 | CV R² (mean±std) | Test R² (mean±std) | n_seeds |')
        out.append('|------|------|------------------|--------------------|---------|')
        sub = df[df['task'] == task]
        for model in ['GNN', 'GIN', 'GAT', 'MPNN', 'GraphSAGE']:
            for enc in ['none', 'label', 'mask', 'pool', 'combined']:
                s = sub[(sub['model'] == model) & (sub['encoding'] == enc)]
                if s.empty:
                    continue
                cv_pm = fmt_pm(s['cv_r2'].mean(), s['cv_r2'].std(ddof=1) if len(s) > 1 else 0)
                te_pm = fmt_pm(s['test_r2'].mean(), s['test_r2'].std(ddof=1) if len(s) > 1 else 0)
                out.append(f'| {model} | {enc} | {cv_pm} | {te_pm} | {len(s)} |')
    # Bug fix 验证
    out.append('\n### 初始化Bug修复验证\n')
    out.append('对比 GNN/label (Layer 3) 与 GNN (Layer 2):')
    out.append('\n| 任务 | Layer 2 GNN Test R² | Layer 3 GNN/label Test R² | 差值 |')
    out.append('|------|---------------------|---------------------------|------|')
    l2 = safe_read_csv(os.path.join(RESULTS, 'layer2_gnn', 'all_layer2_gnn_summary.csv'))
    for task in TASKS:
        l3sub = df[(df['task'] == task) & (df['model'] == 'GNN') & (df['encoding'] == 'label')]
        if l3sub.empty:
            continue
        l3val = l3sub['test_r2'].mean()
        l2val = float('nan')
        if l2 is not None:
            l2r = l2[(l2['task'] == task) & (l2['model'] == 'GNN')]
            if not l2r.empty:
                l2val = l2r['test_r2_mean'].values[0]
        delta = l3val - l2val if not np.isnan(l2val) else float('nan')
        out.append(f'| {TASK_DISPLAY.get(task, task)} | {fmt(l2val)} | {fmt(l3val)} | {fmt(delta)} |')
    return '\n'.join(out) + '\n'


# ============== Layer 4 ==============
def collect_layer4():
    df = safe_read_csv(os.path.join(RESULTS, 'layer4_substituent', 'layer4_summary.csv'))
    return df


def render_layer4(df):
    if df is None:
        return '## Layer 4: 取代基效应建模\n\n数据未找到。\n'
    method_names = {
        'm1': 'M1 Hammett嵌入', 'm2': 'M2 单调性约束', 'm3': 'M3 层次注意力',
        'm4': 'M4 双通道消息', 'm5': 'M5 替换预测预训练', 'm6': 'M6 位置编码',
        'm7': 'M7 扰动学习', 'm8': 'M8 多尺度池化',
    }
    out = ['## Layer 4: 取代基效应建模 (8种方法, lunci6外部验证)\n']
    out.append('所有方法基于 Layer 2 GNN backbone, 加入取代基效应模块, 在 lunci6 上做外部验证\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('test_r2', ascending=False)
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 排名 | 方法 | CV R² | Test R² | lunci6 R² | ΔTest R² vs L2 | 训练时间(s) |')
        out.append('|------|------|-------|---------|-----------|---------------|------------|')
        # Layer 2 baseline
        l2 = safe_read_csv(os.path.join(RESULTS, 'layer2_gnn', 'all_layer2_gnn_summary.csv'))
        baseline = float('nan')
        if l2 is not None:
            l2r = l2[(l2['task'] == task) & (l2['model'] == 'GNN')]
            if not l2r.empty:
                baseline = l2r['test_r2_mean'].values[0]
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            delta = float(r['test_r2']) - baseline if not np.isnan(baseline) else float('nan')
            out.append(f'| {i} | {method_names.get(r["method"], r["method"])} | '
                       f'{fmt(r.get("cv_r2"))} | {fmt(r.get("test_r2"))} | '
                       f'{fmt(r.get("lunci6_r2"), 2)} | {fmt(delta, 4)} | '
                       f'{fmt(r.get("train_time_sec"), 1)} |')
    # 跨任务最佳
    out.append('\n### 跨任务最佳方法\n')
    best_per_task = {}
    for task in TASKS:
        sub = df[df['task'] == task]
        if sub.empty:
            continue
        best_test = sub.loc[sub['test_r2'].idxmax()]
        best_lunci6 = sub.loc[sub['lunci6_r2'].idxmax()]
        best_per_task[task] = (best_test['method'], best_test['test_r2'],
                                best_lunci6['method'], best_lunci6['lunci6_r2'])
    out.append('| 任务 | 最佳Test方法 | Test R² | 最佳lunci6方法 | lunci6 R² |')
    out.append('|------|-------------|---------|---------------|-----------|')
    for task, (bm, br, lm, lr) in best_per_task.items():
        out.append(f'| {TASK_DISPLAY.get(task, task)} | {method_names.get(bm, bm)} | '
                   f'{fmt(br)} | {method_names.get(lm, lm)} | {fmt(lr, 2)} |')
    return '\n'.join(out) + '\n'


# ============== Aromatic Split ==============
def collect_aromatic():
    df = safe_read_csv(os.path.join(PROJ_ROOT, 'aromatic_split', 'results', 'aromatic_split_summary.csv'))
    return df


def render_aromatic(df):
    if df is None:
        return '## 芳香性 vs 非芳香拆分\n\n数据未找到。\n'
    out = ['## 芳香性 vs 非芳香拆分测试\n']
    out.append('将数据集按 RDKit GetIsAromatic 分组, 分别训练+测试, 与全部数据对照\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 模式 | 模型 | n | CV R² | Test R² | Test MAE |')
        out.append('|------|------|---|-------|---------|----------|')
        for mode in ['all', 'aromatic', 'non_aromatic']:
            for model in ['GNN', 'MPNN']:
                r = sub[(sub['mode'] == mode) & (sub['model'] == model)]
                if r.empty:
                    out.append(f'| {mode} | {model} | - | - | (失败) | - |')
                else:
                    rr = r.iloc[0]
                    out.append(f'| {mode} | {model} | {int(rr["n"])} | {fmt(rr.get("cv_r2"))} | '
                               f'{fmt(rr.get("test_r2"))} | {fmt(rr.get("test_mae"))} |')
        # 分析
        all_mpnn = sub[(sub['mode'] == 'all') & (sub['model'] == 'MPNN')]
        aro_mpnn = sub[(sub['mode'] == 'aromatic') & (sub['model'] == 'MPNN')]
        non_mpnn = sub[(sub['mode'] == 'non_aromatic') & (sub['model'] == 'MPNN')]
        out.append('')
        if not all_mpnn.empty and not aro_mpnn.empty:
            out.append(f'- **观察**: all({fmt(all_mpnn.iloc[0]["test_r2"])}) > '
                       f'aromatic({fmt(aro_mpnn.iloc[0]["test_r2"])})')
            if not non_mpnn.empty:
                out.append(f'  > non_aromatic({fmt(non_mpnn.iloc[0]["test_r2"])})')
            elif len(sub[sub['mode'] == 'non_aromatic']) == 0:
                out.append('  > non_aromatic(失败-BatchNorm)')
    return '\n'.join(out) + '\n'


# ============== Optuna ==============
def collect_optuna():
    rows = []
    od = os.path.join(PROJ_ROOT, 'optuna_search', 'results')
    for f in sorted(glob.glob(os.path.join(od, 'best_params_*.json'))):
        name = os.path.basename(f).replace('best_params_', '').replace('.json', '')
        task, model = name.rsplit('_', 1)
        with open(f) as fh:
            d = json.load(fh)
        p = d['params']
        rows.append({
            'task': task, 'model': model, 'cv_r2': d['cv_r2'], **p
        })
    # optuna_summary.csv (最终汇总)
    summary = safe_read_csv(os.path.join(od, 'optuna_summary.csv'))
    return pd.DataFrame(rows) if rows else None, summary


def render_optuna(df, summary):
    if df is None or df.empty:
        return '## Optuna 超参数优化\n\n搜索仍在进行中或数据未找到。\n'
    out = ['## Optuna 超参数优化 (TPE贝叶斯, 30 trials × 3模型 × 3任务)\n']
    out.append('搜索空间: hidden_dim[64,128,256], n_conv[2-4], n_hidden[1-3], lr[1e-4,1e-2], '
               'dropout[0,0.5], batch[32,64,128], weight_decay[1e-6,1e-3]\n')
    out.append('优化目标: 5折CV平均R², n_epochs=150, patience=20\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('cv_r2', ascending=False)
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 排名 | 模型 | CV R² | hidden | conv | hidden_layers | lr | dropout | batch | weight_decay |')
        out.append('|------|------|-------|--------|------|---------------|-----|---------|-------|--------------|')
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            out.append(f'| {i} | {r["model"]} | {fmt(r["cv_r2"])} | {int(r["hidden_dim"])} | '
                       f'{int(r["n_conv_layers"])} | {int(r["n_hidden_layers"])} | '
                       f'{fmt(r["lr"], 5)} | {fmt(r["dropout"], 3)} | {int(r["batch_size"])} | '
                       f'{float(r["weight_decay"]):.2e} |')
    # 完成状态
    n_done = len(df)
    n_total = 9  # 3 tasks × 3 models
    out.append(f'\n**已完成**: {n_done}/{n_total} 组合\n')
    if n_done < n_total:
        out.append('⚠️ 部分搜索仍在进行, 完成后重新运行本脚本以获取完整结果\n')
    return '\n'.join(out) + '\n'


# ============== Generalization ==============
def collect_generalization():
    df = safe_read_csv(os.path.join(RESULTS, 'generalization', 'generalization_summary.csv'))
    return df


def render_generalization(df):
    if df is None:
        return '## 泛化测试\n\n数据未找到。\n'
    out = ['## 泛化测试 (4种out-of-set)\n']
    out.append('- **scaffold**: DeepChem骨架拆分 (385骨架)')
    out.append('- **ring_type**: 环类型留出 (苯/吡啶/吲哚/苯并噻吩/呋喃)')
    out.append('- **lunci6**: 6轮外部数据集')
    out.append('- **lunci78**: 7-8轮外部数据集\n')
    for task in TASKS:
        sub = df[df['task'] == task].copy()
        if sub.empty:
            continue
        out.append(f'\n### {TASK_DISPLAY.get(task, task)}\n')
        out.append('| 测试类型 | 留出组 | 模型 | n_train | n_test | Test R² | Test MAE |')
        out.append('|---------|--------|------|---------|--------|---------|----------|')
        for tt in ['scaffold', 'ring_type', 'lunci6', 'lunci78']:
            tt_sub = sub[sub['test_type'] == tt].sort_values('test_r2', ascending=False)
            for _, r in tt_sub.iterrows():
                out.append(f'| {tt} | {r["holdout"]} | {r["model"]} | {int(r["n_train"])} | '
                           f'{int(r["n_test"])} | {fmt(r["test_r2"], 4)} | {fmt(r["test_mae"], 4)} |')
    return '\n'.join(out) + '\n'


# ============== Main ==============
def main():
    print('=== 收集数据 ===')
    l1 = collect_layer1();     print(f'  Layer 1: {"OK" if l1 is not None else "MISSING"}')
    l2 = collect_layer2();     print(f'  Layer 2: {"OK" if l2 is not None else "MISSING"}')
    l3 = collect_layer3();     print(f'  Layer 3: {"OK" if l3 is not None else "MISSING"} '
                                     f'({len(l3) if l3 is not None else 0} rows)')
    l4 = collect_layer4();     print(f'  Layer 4: {"OK" if l4 is not None else "MISSING"}')
    aro = collect_aromatic();  print(f'  Aromatic split: {"OK" if aro is not None else "MISSING"}')
    opt_df, opt_sum = collect_optuna()
    print(f'  Optuna: {"OK" if opt_df is not None else "MISSING"} '
          f'({len(opt_df) if opt_df is not None else 0} combos)')
    gen = collect_generalization(); print(f'  Generalization: {"OK" if gen is not None else "MISSING"}')

    print('\n=== 生成报告 ===')
    sections = [
        f'# 芳香性预测综合实验报告\n',
        f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n',
        f'**数据集**: collet_nics_0716 / collet_homa_0716 / collet_mbco_0716\n',
        f'**任务**: HOMA, NICS(1)zz, MBCO\n',
        f'**5种随机种子**: 42, 123, 456, 789, 2024\n',
        '\n---\n',
        render_layer1(l1),
        render_layer2(l2),
        render_layer3(l3),
        render_layer4(l4),
        render_aromatic(aro),
        render_optuna(opt_df, opt_sum),
        render_generalization(gen),
    ]

    # 关键发现
    sections.append('## 关键发现总结\n')
    findings = []
    if l2 is not None:
        for task in TASKS:
            sub = l2[l2['task'] == task]
            if not sub.empty:
                best = sub.loc[sub['test_r2_mean'].idxmax()]
                findings.append(f'- **{TASK_DISPLAY.get(task, task)} Layer 2最佳**: {best["model"]} '
                                f'(Test R²={fmt(best["test_r2_mean"])})')
    if l3 is not None:
        for task in TASKS:
            sub = l3[(l3['task'] == task) & (l3['encoding'] == 'label')]
            if not sub.empty:
                best = sub.loc[sub['test_r2'].idxmax()]
                findings.append(f'- **{TASK_DISPLAY.get(task, task)} Layer 3最佳环编码(label)**: '
                                f'{best["model"]} (Test R²={fmt(best["test_r2"])})')
    if l4 is not None:
        for task in TASKS:
            sub = l4[l4['task'] == task]
            if not sub.empty:
                best_test = sub.loc[sub['test_r2'].idxmax()]
                best_lunci6 = sub.loc[sub['lunci6_r2'].idxmax()]
                findings.append(f'- **{TASK_DISPLAY.get(task, task)} Layer 4**: '
                                f'最佳Test={best_test["method"]}({fmt(best_test["test_r2"])}), '
                                f'最佳lunci6={best_lunci6["method"]}({fmt(best_lunci6["lunci6_r2"], 2)})')
    if opt_df is not None and not opt_df.empty:
        for task in TASKS:
            sub = opt_df[opt_df['task'] == task]
            if not sub.empty:
                best = sub.loc[sub['cv_r2'].idxmax()]
                findings.append(f'- **{TASK_DISPLAY.get(task, task)} Optuna最佳**: '
                                f'{best["model"]} (CV R²={fmt(best["cv_r2"])})')
    sections.append('\n'.join(findings) + '\n')

    md = '\n'.join(sections)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md)
    print(f'\n报告已保存: {OUTPUT_MD}')
    print(f'总行数: {len(md.splitlines())}')


if __name__ == '__main__':
    main()
