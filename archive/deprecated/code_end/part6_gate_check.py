
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# -*- coding: utf-8 -*-
"""
lunci6 补充诊断: 决定"模型侧修复够不够"的两个闸门数字
=====================================================================
闸门1: 训练集中"同核心、不同取代基"的配对资源有多少?
        -> 决定配对Δ监督(模型侧修复的核心)是否可行
闸门2: 标签噪声地板 (重复分子的标签差异) vs 取代基信号强度
        -> 决定是否存在信息论意义上的天花板(噪声>信号时只能先修数据)

附带检查: 训练集萘环覆盖 / lunci6重复分子 / 含[*]的异常取代基

适配说明 (相对 .txt 原版):
  - m5 预测文件只有 true/pred 两列, 从 lunci6-test.csv 注入 SMILES
  - 列名大小写兼容 (SMILES / smiles)
  - 输出同时写入 part6_gate_check_report.md
=====================================================================
"""
import os
import sys
import numpy as np
import pandas as pd

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
os.chdir(PROJ_ROOT)

TASKS = ['HOMA', 'NICS_1zz', 'MBCO']
TRAIN_CSV = {
    'HOMA':    'data1_end/collet_homa_0716.csv',
    'NICS_1zz': 'data1_end/collet_nics_0716.csv',
    'MBCO':    'data1_end/collet_mbco_0716.csv',
}
LUNCI6_CSV = 'data1_end/lunci6-test.csv'
PRED_M5 = 'results/layer4_substituent/{task}/m5/lunci6_predictions.csv'
SMI_CANDS  = ['smiles', 'SMILES', 'smi', 'mol_smiles', 'molecule']
CORE_CANDS = ['core_id', 'core', 'scaffold', 'parent', 'ring_core', 'series']

# lunci6 组内真实std (来自上一轮诊断报告 Part4)
LUNCI6_SIGNAL = {'HOMA': 0.0456, 'NICS_1zz': 2.3257, 'MBCO': 0.0136}

# 收集 MD 报告内容
MD_LINES = []


def md(line=''):
    print(line)
    MD_LINES.append(line)


def find_col(df, cands):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        return Chem, MurckoScaffold
    except ImportError:
        return None, None


def load_lunci6_smiles():
    """从 lunci6-test.csv 读取 SMILES 列表 (顺序与预测文件对齐)"""
    df = pd.read_csv(LUNCI6_CSV, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    smi_col = find_col(df, SMI_CANDS)
    if smi_col is None:
        return None
    return df[smi_col].astype(str).tolist()


def load_pred_with_smiles(task):
    """加载 m5 lunci6 预测文件, 并从 lunci6-test.csv 注入 SMILES 列"""
    p = PRED_M5.format(task=task)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if not {'true', 'pred'} <= set(df.columns):
        return None
    smiles_list = load_lunci6_smiles()
    if smiles_list is None or len(smiles_list) != len(df):
        return df  # 无法注入, 返回原始 df
    df['smiles'] = smiles_list
    return df


# ============ 闸门1: 训练集同核心配对资源 ============
def gate1():
    md('=' * 72)
    md('闸门1  训练集"同核心不同取代基"配对资源  ->  配对Δ监督可行性')
    md('-' * 72)
    Chem, Murcko = try_rdkit()
    if Chem is None:
        md('未安装 rdkit, 跳过。pip install rdkit 后重跑, 或提供 core_id 列。')
        return
    naph = Chem.MolFromSmarts('c1ccc2ccccc2c1')
    md('')
    md('| 任务 | 分子数 | 唯一核心 | ≥2分子核心 | 可用配对数 | 组内std中位 | 组内std均值 | 萘环分子数 | 萘环占比 |')
    md('|---|---|---|---|---|---|---|---|---|')
    for task, path in TRAIN_CSV.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        smi = find_col(df, SMI_CANDS)
        core_col = find_col(df, CORE_CANDS)
        if core_col:
            keys = df[core_col].astype(str)
        elif smi:
            keys = df[smi].map(
                lambda s: (lambda m: Murcko.MurckoScaffoldSmiles(mol=m) if m else None)(
                    Chem.MolFromSmiles(str(s))))
        else:
            md(f'[{task}] 无 SMILES/core 列, 跳过'); continue
        g = keys.value_counts()
        multi = g[g >= 2]
        pairs = int((multi * (multi - 1) / 2).sum())
        # 组内标签方差 = 取代基信号强度(训练侧)
        tcol = next((c for c in df.columns if task.split('_')[0].upper() in c.upper()),
                    df.columns[-1])
        y = pd.to_numeric(df[tcol], errors='coerce')
        tmp = pd.DataFrame({'k': keys.values, 'y': y.values}).dropna()
        wstd = tmp.groupby('k')['y'].agg(['std', 'count'])
        wstd = wstd[wstd['count'] >= 2]['std'].dropna()
        std_med = wstd.median() if len(wstd) else float('nan')
        std_mean = wstd.mean() if len(wstd) else float('nan')
        has_naph = 0
        naph_pct = 0.0
        if smi:
            has_naph = df[smi].map(
                lambda s: (lambda m: m is not None and m.HasSubstructMatch(naph))(
                    Chem.MolFromSmiles(str(s)))).sum()
            naph_pct = has_naph / len(df) * 100
        md(f'| {task} | {len(df)} | {len(g)} | {len(multi)} | {pairs} | '
           f'{std_med:.4f} | {std_mean:.4f} | {has_naph} | {naph_pct:.1f}% |')
    md('')
    md('**判定**:')
    md('- 配对数 ≥ 数千 → 闸门1通过: 模型侧修复(配对Δ监督)可行, 无需新数据')
    md('- 配对数 数百~数千 → 可行但偏紧: 配对监督 + 强结构约束(锚定/瓶颈/幅度封顶)')
    md('- 配对数 < 数百 → 闸门1不通过: 需要数据侧(伪配对生成+廉价打标, 复用m5的机器)')
    md('- 萘环占比 ≈ 0 → 核心偏差无法从训练学出, 只能靠 k-shot 校准或生成稠环数据')


# ============ 闸门2: 标签噪声地板 vs 信号 ============
def gate2():
    md('')
    md('=' * 72)
    md('闸门2  标签噪声地板 vs 取代基信号  ->  模型修复的天花板在哪')
    md('-' * 72)
    md('')
    md('| 任务 | 重复分子组数 | 噪声地板std | 取代基信号std | 信噪比SNR | Δ-MAE极限 |')
    md('|---|---|---|---|---|---|')
    for task in TASKS:
        df = load_pred_with_smiles(task)
        if df is None:
            md(f'| {task} | (无预测文件) | - | - | - | - |')
            continue
        smi = find_col(df, SMI_CANDS)
        if not smi:
            md(f'| {task} | (无SMILES列) | - | - | - | - |')
            continue
        # 重复分子标签差异 = 噪声地板估计 (同分子多次出现的标签波动)
        dup = df.groupby(df[smi].astype(str))['true'].agg(['std', 'count', 'mean'])
        dup = dup[dup['count'] >= 2]['std'].dropna()
        signal = LUNCI6_SIGNAL[task]
        if len(dup) == 0:
            md(f'| {task} | 0 | N/A | {signal:.4f} | N/A | N/A |')
            md(f'  [{task}] 无重复分子, 无法从lunci6估噪声; 若训练集有重复分子可做同样检查')
            continue
        noise = dup.median()
        snr = signal / max(noise, 1e-9)
        delta_mae_lim = noise * 1.13  # 同核心对的Δ含√2倍噪声
        md(f'| {task} | {len(dup)} | {noise:.4f} | {signal:.4f} | {snr:.2f} | {delta_mae_lim:.4f} |')
    md('')
    md('**判定** (信噪比 = 信号std / 噪声std):')
    md('- SNR > 2 → 闸门2通过: 配对Δ监督大有可为, 模型侧能拿回大部分精度')
    md('- SNR 1~2 → 边际: 需要大量配对平均掉噪声, 或配合数据侧降噪')
    md('- SNR < 1 → 闸门2不通过: 任何算法都无法从现有标签提取Δ信号(信息论极限),')
    md('  唯一出路是数据侧: 统一标签计算协议(几何来源/DFT级别/构象规则)降噪')


# ============ 附带: 异常取代基检查 ============
def extra():
    md('')
    md('=' * 72)
    md('附带检查  含 [*] dummy 原子的取代基条目')
    md('-' * 72)
    df = load_pred_with_smiles('HOMA')
    if df is None:
        md('无 HOMA m5 预测文件, 跳过')
        return
    sub_col = find_col(df, ['substituent', 'substituents', 'sub', 'R', 'sub_smiles'])
    smi_col = find_col(df, SMI_CANDS)
    if smi_col:
        # 通过 SMILES 检测含 [*] 的 dummy 原子
        bad_mask = df[smi_col].astype(str).str.contains(r'\*', regex=True)
        bad = df[bad_mask]
        all_err = (df['pred'] - df['true']).abs()
        md(f'含[*]的分子行数: {len(bad)} / {len(df)}')
        if len(bad):
            err = (bad['pred'] - bad['true']).abs()
            md(f'这些分子的MAE={err.mean():.4f} vs 全体MAE={all_err.mean():.4f}')
            md('若显著更高: 图特征化对dummy原子处理有bug, 修复白捡的精度')
        else:
            md('无含[*]的分子, 图特征化无 dummy 原子问题')
    elif sub_col:
        bad = df[df[sub_col].astype(str).str.contains(r'\*', regex=True)]
        all_err = (df['pred'] - df['true']).abs()
        md(f'含[*]的分子行数: {len(bad)} / {len(df)}')
        if len(bad):
            err = (bad['pred'] - bad['true']).abs()
            md(f'这些分子的MAE={err.mean():.4f} vs 全体MAE={all_err.mean():.4f}')
    else:
        md('无 substituent/SMILES 列, 跳过')


if __name__ == '__main__':
    gate1()
    gate2()
    extra()
    md('')
    md('把输出贴回来, 我据此给出最终的"模型侧/数据侧"路线图。')

    # 写 MD 报告
    out_md = os.path.join(PROJ_ROOT, 'part6_gate_check_report.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# lunci6 补充诊断: 配对资源与噪声地板\n\n')
        f.write('**生成时间**: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
        f.write('**目的**: 通过两个闸门数字, 判断"模型侧修复(配对Δ监督)够不够"还是"必须走数据侧"\n\n')
        f.write('---\n\n')
        f.write('\n'.join(MD_LINES))
    print(f'\n报告已保存: {out_md}', flush=True)
