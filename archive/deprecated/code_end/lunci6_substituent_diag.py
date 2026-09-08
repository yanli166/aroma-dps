
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lunci6 取代基泛化诊断工具包 (适配现有代码库)
=====================================================================
目的: 区分"仅取代基变化"测试集上预测失败的三种机制
  机制A: 训练集对 lunci6 目标值区间的覆盖不足 (标签密度问题)
  机制B: lunci6 的取代基类型在训练中没见过 (取代基分布偏移)
  机制C: 模型学会了"忽略取代基" (敏感度坍缩, 预测≈核心环常数)

适配:
  - 从 data1_end/lunci6-test.csv 获取 SMILES
  - 合并到 Layer 2/4 的预测文件中
  - 用 RDKit Murcko Scaffold 做同核心分组
  - 用 RDKit ReplaceCore 提取取代基

输出:
  - 控制台详细诊断
  - lunci6_diagnostic_report.md 完整报告
=====================================================================
"""
import os
import sys
import itertools
import json
import numpy as np
import pandas as pd
from datetime import datetime

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
os.chdir(PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)

# ---------------- 配置 ----------------
TASKS = ['HOMA', 'NICS_1zz', 'MBCO']
TASK_DISPLAY = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS(1)zz', 'MBCO': 'MBCO'}

# 训练数据路径和目标列名
TRAIN_CSV = {
    'HOMA':     ('data1_end/collet_homa_0716.csv', 'homa_value'),
    'NICS_1zz': ('data1_end/collet_nics_0716.csv', 'NICS_value'),
    'MBCO':     ('data1_end/collet_mbco_0716.csv', 'mbco_value'),
}
LUNCI6_CSV = 'data1_end/lunci6-test.csv'
LUNCI6_TARGET_COL = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}

# Layer 4 预测
L4_PRED_TEMPLATE = 'results/layer4_substituent/{task}/{method}/lunci6_predictions.csv'
L4_METHODS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
L4_NAMES = {
    'm1': 'Hammett嵌入', 'm2': '单调性约束', 'm3': '层次注意力', 'm4': '双通道消息',
    'm5': '替换预测预训练', 'm6': '位置编码', 'm7': '扰动学习', 'm8': '多尺度池化',
}

# Layer 2 预测 (generalization test)
L2_PRED_TEMPLATE = 'results/generalization/{task}/lunci6/{model}/test_predictions.csv'
L2_MODELS = ['GNN', 'MPNN', 'GraphSAGE']

# Hammett 常数 (sigma_para, sigma_meta)
HAMMETT = {
    'N(CH3)2': (-0.83, -0.15), 'NMe2': (-0.83, -0.15), 'NH2': (-0.66, -0.16),
    'OH': (-0.37, 0.12), 'OCH3': (-0.27, 0.12), 'OMe': (-0.27, 0.12), 'OC': (-0.27, 0.12),
    'CH3': (-0.17, -0.07), 'Me': (-0.17, -0.07), 'C': (-0.17, -0.07),
    'C(CH3)3': (-0.20, -0.10), 'H': (0.00, 0.00), 'F': (0.06, 0.34),
    'I': (0.18, 0.35), 'Cl': (0.23, 0.37), 'Br': (0.23, 0.39),
    'CHO': (0.42, 0.36), 'COOH': (0.45, 0.37), 'COOCH3': (0.45, 0.37),
    'CF3': (0.54, 0.43), 'CN': (0.66, 0.56), 'NO2': (0.78, 0.71),
    'S': (0.00, 0.15), 'SH': (0.00, 0.15), 'SMe': (0.00, 0.15), 'CSC': (0.00, 0.15),
}

OUTPUT_MD = os.path.join(PROJ_ROOT, 'lunci6_diagnostic_report.md')

# ---------------- MD 报告收集器 ----------------
class MDReport:
    def __init__(self):
        self.sections = []

    def add(self, text):
        self.sections.append(text)

    def save(self, path):
        md = '\n'.join(self.sections)
        with open(path, 'w') as f:
            f.write(md)
        print(f'\n报告已保存: {path}')

    def h1(self, t): self.add(f'\n# {t}\n')
    def h2(self, t): self.add(f'\n## {t}\n')
    def h3(self, t): self.add(f'\n### {t}\n')
    def text(self, t): self.add(t)
    def table(self, header, rows):
        self.add(f'\n| {" | ".join(header)} |')
        self.add(f'|{"---|" * len(header)}')
        for r in rows:
            self.add(f'| {" | ".join(str(x) for x in r)} |')
        self.add('')


MD = MDReport()


# ---------------- RDKit 工具 ----------------
def get_scaffold(smi):
    """Murcko 骨架"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(str(smi))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None
    except Exception:
        return None


def get_substituent(smi, scaffold_smi=None):
    """提取取代基 SMILES (非骨架部分)"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        if scaffold_smi is None:
            scaffold_smi = get_scaffold(smi)
        if scaffold_smi is None or scaffold_smi == '':
            return 'H'  # 无骨架 = 纯环
        scaffold = Chem.MolFromSmiles(scaffold_smi)
        if scaffold is None:
            return None
        # 用 ReplaceCore 提取侧链
        side_chains = Chem.ReplaceCore(mol, scaffold)
        if side_chains is not None and side_chains.GetNumAtoms() > 0:
            sc_smi = Chem.MolToSmiles(side_chains)
            # 清理 dummy atoms [*], 保留取代基主体
            sc_smi = sc_smi.replace('[*]', '').replace('[1*]', '').replace('[2*]', '')
            # 去掉前导连字符
            sc_smi = sc_smi.lstrip('-').strip()
            return sc_smi if sc_smi else 'H'
        return 'H'  # 无侧链 = 未取代
    except Exception:
        return None


def hammett_sigma(sub_smi):
    """根据取代基 SMILES 查找 Hammett sigma_para"""
    if sub_smi is None or sub_smi == 'H' or sub_smi == '':
        return 0.0
    # 直接查找
    if sub_smi in HAMMETT:
        return HAMMETT[sub_smi][0]
    # 模糊匹配
    for key, (sp, sm) in HAMMETT.items():
        if key.lower() in sub_smi.lower() or sub_smi.lower() in key.lower():
            return sp
    return np.nan


# ---------------- 数据加载 ----------------
def load_lunci6_meta():
    """加载 lunci6 测试集元数据 (SMILES, Ring_ID 等)"""
    df = pd.read_csv(LUNCI6_CSV)
    return df


def load_train(task):
    """加载训练数据"""
    path, tcol = TRAIN_CSV[task]
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 清理 NICS 的 Unnamed 列
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    smi_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    df['_y_'] = pd.to_numeric(df[tcol], errors='coerce')
    df = df.dropna(subset=['_y_', smi_col]).reset_index(drop=True)
    return df, smi_col


def load_preds_with_smiles(task, method, layer='L4'):
    """加载预测文件并合并 SMILES"""
    if layer == 'L4':
        path = L4_PRED_TEMPLATE.format(task=task, method=method)
    else:
        path = L2_PRED_TEMPLATE.format(task=task, model=method)
    if not os.path.exists(path):
        return None
    pred = pd.read_csv(path)
    if not {'true', 'pred'} <= set(pred.columns):
        return None
    # 合并 lunci6 元数据
    meta = load_lunci6_meta()
    if len(pred) == len(meta):
        pred['SMILES'] = meta['SMILES'].values
        pred['New_ID'] = meta['New_ID'].values
        pred['Ring_ID'] = meta['Ring_ID'].values
    else:
        # 尝试按 true 值对齐
        tcol = LUNCI6_TARGET_COL[task]
        meta_sorted = meta.sort_values(tcol).reset_index(drop=True)
        pred_sorted = pred.sort_values('true').reset_index(drop=True)
        if len(pred_sorted) == len(meta_sorted):
            pred_sorted['SMILES'] = meta_sorted['SMILES'].values
            pred_sorted['New_ID'] = meta_sorted['New_ID'].values
            pred_sorted['Ring_ID'] = meta_sorted['Ring_ID'].values
            pred = pred_sorted
        else:
            print(f'  WARN: {task}/{method} 预测({len(pred)})与lunci6({len(meta)})行数不匹配')
            return None
    # 计算 scaffold 和取代基
    pred['_scaffold_'] = pred['SMILES'].map(get_scaffold)
    pred['_sub_'] = pred.apply(lambda r: get_substituent(r['SMILES'], r['_scaffold_']), axis=1)
    return pred


def any_preds(task, layer='L4'):
    """获取任一方法的预测 (用于 true 值)"""
    methods = L4_METHODS if layer == 'L4' else L2_MODELS
    for m in methods:
        d = load_preds_with_smiles(task, m, layer)
        if d is not None:
            return d
    return None


# ============ Part 1: 训练集对 lunci6 目标区间的覆盖 (机制A) ============
def part1():
    print('\n' + '=' * 72)
    print('Part 1  标签覆盖率检验  ->  验证"这个范围的样本非常之多"是否成立')
    print('-' * 72)
    print('range 宽 ≠ 密度够: 训练 range 8.54 可能大部分堆在负值区, 正值区稀疏\n')

    MD.h2('Part 1: 标签覆盖率检验 (机制A)')
    MD.text('验证训练集对 lunci6 目标值区间的**密度覆盖**(不是 range 覆盖)。\n')
    MD.text('> range 宽 ≠ 密度够: 训练 range 8.54 可能大部分堆在负值区, 正值区稀疏\n')

    header = ['任务', 'lunci6区间', '覆盖率', '放宽±0.5σ', '训练n', '均值偏移z']
    rows = []
    print(f'{"任务":10s} {"lunci6区间":>20s} {"覆盖率":>8s} {"放宽±0.5σ":>10s} {"训练n":>7s} {"均值偏移z":>9s}')

    for task in TASKS:
        tr = load_train(task)
        df_te = any_preds(task)
        if tr is None or df_te is None:
            continue
        df_tr, smi_col = tr
        y_tr = df_tr['_y_'].values
        y_te = df_te['true'].values
        lo, hi = float(y_te.min()), float(y_te.max())
        m = 0.5 * float(y_te.std())
        cov = np.mean((y_tr >= lo) & (y_tr <= hi))
        cov_m = np.mean((y_tr >= lo - m) & (y_tr <= hi + m))
        z = (y_te.mean() - y_tr.mean()) / (y_tr.std() + 1e-12)

        print(f'{task:10s} [{lo:8.3f},{hi:8.3f}] {cov * 100:7.1f}% {cov_m * 100:9.1f}% '
              f'{len(y_tr):7d} {z:9.2f}')
        rows.append([TASK_DISPLAY[task], f'[{lo:.3f}, {hi:.3f}]',
                     f'{cov*100:.1f}%', f'{cov_m*100:.1f}%', len(y_tr), f'{z:.2f}'])

    MD.table(header, rows)
    MD.text("""
**解读**:
- 覆盖率高(>5%) 且 |z|<2 → 机制A排除, 问题在取代基(机制B/C)
- 覆盖率很低(<1%) 或 |z|>3 → 虽然 range 覆盖, 但密度不足, 模型在该区域仍是外推
""")


# ============ Part 2: 取代基敏感度比 + Δ配对指标 (机制C, 核心!) ============
def group_metrics(df):
    """同核心分组 -> 敏感度比, Δ配对指标, 收缩λ扫描"""
    sst = ssp = n_all = 0.0
    pairs, shrink_rows = [], []
    for _, g in df.groupby('_scaffold_'):
        if len(g) < 2:
            continue
        t, p = g['true'].values, g['pred'].values
        sst += t.var() * len(g)
        ssp += p.var() * len(g)
        n_all += len(g)
        gm = p.mean()
        row = {'n': len(g)}
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            row[lam] = np.mean(np.abs((gm + lam * (p - gm)) - t))
        shrink_rows.append(row)
        for i, j in itertools.combinations(range(len(g)), 2):
            pairs.append((t[i] - t[j], p[i] - p[j]))
    if not pairs:
        return None
    dt = np.array([a for a, _ in pairs])
    dp = np.array([b for _, b in pairs])
    from scipy.stats import spearmanr
    sr = pd.DataFrame(shrink_rows) if shrink_rows else None
    best_lam, best_mae, base_mae = None, None, None
    if sr is not None and len(sr):
        maes = {lam: np.average(sr[lam], weights=sr['n']) for lam in [0.0, 0.25, 0.5, 0.75, 1.0]}
        best_lam = min(maes, key=maes.get)
        best_mae = maes[best_lam]
        base_mae = maes[1.0]
    return dict(
        sens=np.sqrt(ssp / sst) if sst > 0 else np.nan,
        mae_d=np.mean(np.abs(dp - dt)),
        sign=np.mean(np.sign(dp) == np.sign(dt)),
        rho=spearmanr(dt, dp).statistic if len(dt) > 2 else np.nan,
        npairs=len(pairs),
        best_lam=best_lam, best_mae=best_mae, base_mae=base_mae,
    )


def part2():
    print('\n' + '=' * 72)
    print('Part 2  取代基敏感度比 + Δ配对指标  ->  机制C(模型忽略取代基)实锤检验')
    print('-' * 72)

    MD.h2('Part 2: 取代基敏感度比 + Δ配对指标 (机制C检验)')
    MD.text('**核心检验**: 模型是否学会了"忽略取代基", 预测≈核心环常数?\n')
    MD.text('- **敏感度比** = 组内pred_std / 组内true_std (≈0 说明坍缩)')
    MD.text('- **λ\*** = 取代基响应最优保留比例 (λ=0 表示完全忽略取代基)')
    MD.text('- **Δ符号acc** = 配对预测差值符号正确率 (50%=随机)\n')

    for task in TASKS:
        all_rows = []
        all_layer_info = []
        for layer, methods, name_prefix in [('L4', L4_METHODS, ''), ('L2', L2_MODELS, 'L2_')]:
            for m in methods:
                df = load_preds_with_smiles(task, m, layer)
                if df is None:
                    continue
                df = df.dropna(subset=['_scaffold_'])
                r = group_metrics(df)
                if r is None:
                    continue
                label = f'{name_prefix}{m}' if layer == 'L2' else m
                all_rows.append((label, r, layer))
                all_layer_info.append((label, r))

        if not all_rows:
            continue

        print(f'\n[{task}]  敏感度比=组内pred_std/组内true_std; λ*: 取代基响应最优保留比例')
        print(f'{"方法":10s} {"敏感度比":>8s} {"Δ-MAE":>8s} {"Δ符号acc":>8s} '
              f'{"Δ-ρ":>6s} {"配对数":>6s} {"λ*":>5s} {"MAE(λ=0)":>9s} {"MAE(λ=1)":>9s}')

        MD.h3(f'{TASK_DISPLAY[task]}')
        header = ['方法', '敏感度比', 'Δ-MAE', 'Δ符号acc', 'Δ-ρ(Spearman)', '配对数', 'λ*', 'MAE(λ=0)', 'MAE(λ=1)']
        md_rows = []
        for label, r, layer in all_rows:
            lam0_mae = r['best_mae'] if r['best_lam'] == 0.0 else float('nan')
            print(f'{label:10s} {r["sens"]:8.2f} {r["mae_d"]:8.4f} {r["sign"]:8.2%} '
                  f'{r["rho"]:6.2f} {r["npairs"]:6d} {str(r["best_lam"]):>5s} '
                  f'{lam0_mae:9.4f} {r["base_mae"]:9.4f}')
            md_rows.append([
                label, f'{r["sens"]:.2f}', f'{r["mae_d"]:.4f}', f'{r["sign"]:.1%}',
                f'{r["rho"]:.2f}', r['npairs'], str(r['best_lam']),
                f'{lam0_mae:.4f}' if not np.isnan(lam0_mae) else '-',
                f'{r["base_mae"]:.4f}',
            ])
        MD.table(header, md_rows)

    MD.text("""
**解读**:
- 敏感度比 ≈ 0 → 机制C实锤: 模型输出几乎不随取代基变化, 坍缩成核心环常数
- λ\* = 0 或 0.25 → 模型的取代基响应是噪声/有害的, "加权取代基"会更糟
- λ\* = 1 但 Δ-MAE大 → 响应方向大体对但精度差, 数据/特征问题
- Δ符号acc ≈ 50% → 模型连取代基影响的方向都没学到
""")


# ============ Part 3: 取代基 seen/unseen + Hammett σ 分布 (机制B) ============
def part3():
    print('\n' + '=' * 72)
    print('Part 3  取代基类型分布对比  ->  机制B(取代基本身没见过)检验')
    print('-' * 72)

    MD.h2('Part 3: 取代基类型分布对比 (机制B检验)')
    MD.text('检验 lunci6 的取代基是否在训练集中出现过, 以及 Hammett σ 分布是否偏移。\n')

    for task in TASKS:
        tr = load_train(task)
        df_te = any_preds(task)
        if tr is None or df_te is None:
            continue
        df_tr, smi_col = tr

        # 提取训练集取代基
        df_tr['_scaffold_'] = df_tr[smi_col].map(get_scaffold)
        df_tr['_sub_'] = df_tr.apply(lambda r: get_substituent(r[smi_col], r['_scaffold_']), axis=1)
        df_te = df_te.dropna(subset=['_sub_'])

        tr_subs = set(df_tr['_sub_'].dropna().unique())
        te_subs = df_te['_sub_'].astype(str)
        unseen = ~te_subs.isin(tr_subs)
        err = np.abs(df_te['pred'].values - df_te['true'].values)

        print(f'\n[{task}] lunci6取代基种类={te_subs.nunique()}, '
              f'未见过分子占比={unseen.mean() * 100:.1f}%')
        print(f'  训练集取代基种类: {len(tr_subs)}')
        print(f'  lunci6取代基: {sorted(te_subs.unique())}')

        MD.h3(f'{TASK_DISPLAY[task]}')
        MD.text(f'- lunci6 取代基种类: {te_subs.nunique()}')
        MD.text(f'- 训练集取代基种类: {len(tr_subs)}')
        MD.text(f'- lunci6 取代基列表: {sorted(te_subs.unique())}')
        MD.text(f'- 未见过分子占比: {unseen.mean()*100:.1f}%\n')

        if unseen.any() and (~unseen).any():
            seen_mae = err[~unseen.values].mean()
            unseen_mae = err[unseen.values].mean()
            print(f'  seen 取代基分子: n={(~unseen).sum()}, MAE={seen_mae:.4f}')
            print(f'  unseen取代基分子: n={unseen.sum()},  MAE={unseen_mae:.4f}')

            MD.table(['类型', 'n', 'MAE'], [
                ['seen (训练集见过)', (~unseen).sum(), f'{seen_mae:.4f}'],
                ['unseen (训练集没见过)', unseen.sum(), f'{unseen_mae:.4f}'],
            ])

            if unseen_mae > seen_mae * 1.3:
                MD.text('→ unseen 的 MAE 显著更高: **机制B成立**, 需要定向补数据\n')
            else:
                MD.text('→ seen 的 MAE 同样很高: 不是"没见过", 是模型没用取代基信息(**机制C**)\n')
        elif unseen.all():
            print(f'  全部 unseen! n={unseen.sum()}, MAE={err.mean():.4f}')
            MD.text(f'- 全部 {unseen.sum()} 个分子取代基均为 unseen\n')
        else:
            print(f'  全部 seen! n={(~unseen).sum()}, MAE={err.mean():.4f}')
            MD.text(f'- 全部 {(~unseen).sum()} 个分子取代基均为 seen\n')

        # Hammett σ 分布
        sp_tr = pd.Series([hammett_sigma(s) for s in df_tr['_sub_'].dropna()]).dropna()
        sp_te = pd.Series([hammett_sigma(s) for s in te_subs]).dropna()
        if len(sp_tr) and len(sp_te):
            print(f'  Hammett σ_p: 训练 mean={sp_tr.mean():+.2f} std={sp_tr.std():.2f} | '
                  f'lunci6 mean={sp_te.mean():+.2f} std={sp_te.std():.2f}')
            MD.text(f'- Hammett σ_p: 训练 mean={sp_tr.mean():+.2f} std={sp_tr.std():.2f} | '
                    f'lunci6 mean={sp_te.mean():+.2f} std={sp_te.std():.2f}')
            if abs(sp_te.mean() - sp_tr.mean()) > 0.3:
                MD.text(f'  → lunci6 的 σ 分布系统性偏向一侧, 可解释单向系统偏差\n')


# ============ Part 4: 母体环基线 ============
def part4():
    print('\n' + '=' * 72)
    print('Part 4  母体环/组均值基线  ->  "模型 vs 忽略取代基的常数"公平对决')
    print('-' * 72)

    MD.h2('Part 4: 母体环/组均值基线')
    MD.text('将模型预测与"忽略取代基的组均值 oracle"对比。\n')
    MD.text('- 模型 MAE 接近或差于组均值 oracle → 取代基分支没有净贡献')
    MD.text('- 两者之差 = "取代基信息"当前的真实价值, 也是锚定Δ架构要抢回来的空间\n')

    # 对 Layer 4 的 m5 (最佳) 和 m7 (思路最对) + Layer 2 的 MPNN
    check_methods = [('L4', 'm5'), ('L4', 'm7'), ('L2', 'MPNN')]

    header = ['任务', '方法', '模型MAE', '组均值oracle MAE', '组内真实std', '取代基净贡献']
    md_rows = []

    for task in TASKS:
        for layer, m in check_methods:
            df = load_preds_with_smiles(task, m, layer)
            if df is None:
                continue
            df = df.dropna(subset=['_scaffold_'])
            mae_model = np.mean(np.abs(df['pred'] - df['true']))
            g = df.groupby('_scaffold_')['true'].transform('mean')
            mask = df.groupby('_scaffold_')['true'].transform('size') >= 2
            if mask.sum() == 0:
                continue
            mae_oracle = np.mean(np.abs(g[mask] - df.loc[mask, 'true']))
            group_std = df.loc[mask].groupby('_scaffold_')['true'].std().mean()
            net = mae_oracle - mae_model  # 正=模型优于oracle, 负=模型不如oracle

            label = f'{layer}_{m}'
            print(f'[{task}][{label}] 模型MAE={mae_model:.4f} | '
                  f'组均值(oracle)MAE={mae_oracle:.4f} | '
                  f'组内真实std={group_std:.4f} | 净贡献={net:+.4f}')

            md_rows.append([TASK_DISPLAY[task], label, f'{mae_model:.4f}',
                            f'{mae_oracle:.4f}', f'{group_std:.4f}', f'{net:+.4f}'])

    MD.table(header, md_rows)
    MD.text("""
**解读**:
- 模型 MAE ≥ 组均值 oracle MAE → 取代基分支没有净贡献 (甚至有害)
- 净贡献 ≈ 0 → 模型等效于"忽略取代基的常数预测器"
- 净贡献 < 0 → 模型的取代基响应是噪声, 比不响应更糟
""")


# ============ Part 5: 解法速查 ============
def part5():
    print('\n' + '=' * 72)
    print('Part 5  解法速查 (按诊断结果对号入座)')
    print('-' * 72)

    MD.h2('Part 5: 解法速查 (按诊断结果对号入座)')

    MD.h3('机制C: 敏感度比≈0 / λ*≈0 (最可能)')
    MD.text("""按优先级:
1. **锚定Δ架构** (m7升级): ŷ = f_ring(核心) + [g(完整分子) − g(同骨架全H母体)]
   - g(H)≡0 由构造保证, 未取代环预测天然精确
   - 所有取代基学习压进Δ分支, 误差被严格界定
2. **配对Δ监督**: 同核心任意两分子(i,j)构成监督对
   - loss = SmoothL1(ŷi−ŷj, yi−yj) + 0.3·softplus(−(ŷi−ŷj)(yi−yj))
   - 比逐点回归强得多地榨取有限的同核心对比信号
3. **定向数据生成**: RDKit反应枚举 [cH:1]>>[c:1]-X
   - 对每个核心环×取代基库生成伪配对分子
   - 用廉价方法(xTB算HOMA/低级别DFT算NICS)打标, 混入微调
""")

    MD.h3('机制B: unseen取代基MAE显著更高')
    MD.text("""1. 检查原子特征化是否有"未知类型坍缩"(形式电荷/氧化态/Si/B/金属)
2. 对 unseen 化学型做定向数据生成(同上), 或至少在报告中按 seen/unseen 分组
""")

    MD.h3('机制A: 覆盖率低')
    MD.text('该区间定向采样/生成 + 重要性加权, 并检查标签质量\n')

    MD.h3('通用改进建议')
    MD.text("""- **m1升级**: Hammett σ 单参数 → Swain-Lupton F/R 双参数(分离诱导/共轭),
  低成本获得 m4"双通道"的大部分物理先验
- **m2升级**: 全局单调性约束 → 配对Δ符号软损失, 更省更稳
- **评估切换**: 对"仅取代基变化"测试集, 主指标改为 Δ-MAE、Δ符号acc、Δ-Spearman、敏感度比
""")


# ============ 主函数 ============
def main():
    print(f'lunci6 取代基泛化诊断工具包')
    print(f'运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'工作目录: {PROJ_ROOT}')

    MD.h1('lunci6 取代基泛化诊断报告')
    MD.text(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    MD.text('**目的**: 区分"仅取代基变化"测试集上预测失败的三种机制:')
    MD.text('- 机制A: 训练集对 lunci6 目标值区间的覆盖不足 (标签密度问题)')
    MD.text('- 机制B: lunci6 的取代基类型在训练中没见过 (取代基分布偏移)')
    MD.text('- 机制C: 模型学会了"忽略取代基" (敏感度坍缩, 预测≈核心环常数)\n')
    MD.text('**测试集**: lunci6-test.csv (312条, 全部为萘环naphthalene衍生物)')
    MD.text('**对比**: Layer 2 (GNN/MPNN/GraphSAGE) + Layer 4 (m1-m8 共8种取代基效应方法)\n')
    MD.text('---\n')

    # 检查数据可用性
    meta = load_lunci6_meta()
    print(f'\nlunci6 测试集: {len(meta)} 行, 列={list(meta.columns)}')
    print(f'分子数(去重SMILES): {meta["SMILES"].nunique()}')
    print(f'唯一骨架数: {meta["SMILES"].map(get_scaffold).nunique()}')

    MD.text(f'**lunci6 测试集统计**: {len(meta)} 行, '
            f'{meta["SMILES"].nunique()} 个唯一分子, '
            f'{meta["SMILES"].map(get_scaffold).nunique()} 个唯一骨架\n')

    part1()
    part2()
    part3()
    part4()
    part5()

    # 保存报告
    MD.save(OUTPUT_MD)
    print('\n诊断完成。')


if __name__ == '__main__':
    main()
