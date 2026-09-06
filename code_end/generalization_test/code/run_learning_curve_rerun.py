
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
Learning Curve 补跑脚本: 补跑缺失的 fraction (50%) 和修复 MBCO 70%

缺失项:
  1. HOMA 50%, NICS 50%, MBCO 50% (数据文件不存在)
  2. MBCO 70% (BatchNorm batch=1 崩溃, 已修复 train_eval.py)
  3. NICS 70% (R²=-0.113, 需检查离群值)

逻辑:
  - 从 lunci10-expanded-test.csv 按 Ring_ID (骨架) 分组
  - 按 fraction 将部分骨架的样本加入主训练集, 剩余作为测试集
  - 用最优模型 (HOMA:MPNN, NICS:MPNN, MBCO:GraphSAGE) 训练并评估
"""
import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import torch

# 路径设置
PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
LAST_END_ROOT = '_PROJ_ROOT/last_end_code'
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_MODELS_ROOT)
sys.path.insert(0, LAST_END_ROOT)

from common.constants import ORIG_MODELS_ROOT
from common.tasks import TASKS, DEFAULT_SEED, compute_metrics, clean_dataset_csv
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, train_model, eval_model, set_full_seed

# 常量
NVL = 60
MAX_ATOMS = 75
EXT_MAX_ATOMS = 85
RING_FLAG_VALUE = 10
SEED = DEFAULT_SEED

L10_CSV = '_PROJ_ROOT/lunci10/lunci10-expanded-test.csv'
TMP_DIR = '/tmp/lc_rerun'
RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/learning_curve'

# 最优模型 (来自 generalization_lunci10 summary)
BEST_MODELS = {
    'HOMA': 'MPNN',
    'NICS_1zz': 'MPNN',
    'MBCO': 'GraphSAGE',
}

# 任务映射
TASK_MAP = {t['name']: t for t in TASKS}
TARGET_MAP = {
    'HOMA': 'homa_value',
    'NICS_1zz': 'NICS_value',
    'MBCO': 'mbco_value',
}

L10_COL_MAP = {
    'HOMA': 'HOMA',
    'NICS_1zz': 'NICS_ZZ',
    'MBCO': 'MBCO',
}


def prepare_l10_data(task_name, fraction):
    """准备 lunci10 数据: 按 identify_ring_type (芳香环类型) 分组

    67 种芳香环类型 (benzene, pyrrole, indole, ...), 按 fraction 划分训练/测试。
    测试集 = 未在训练集中出现过的环类型 (scaffold-level OOD)。
    """
    from generalization_test.code.ring_utils import identify_ring_type

    df_l10 = pd.read_csv(L10_CSV)
    target_col_l10 = L10_COL_MAP[task_name]

    # 剔除 NaN
    df_l10 = df_l10.dropna(subset=[target_col_l10]).reset_index(drop=True)

    # 为每个 SMILES 计算芳香环类型
    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}

    # 按环类型分组
    rt_to_smiles = {}
    for smi, rt in smi_to_rt.items():
        rt_to_smiles.setdefault(rt, []).append(smi)

    rt_list = sorted(rt_to_smiles.keys())
    n_rt = len(rt_list)
    rng = np.random.RandomState(SEED)
    rng.shuffle(rt_list)

    n_train_rt = int(n_rt * fraction)
    train_rts = set(rt_list[:n_train_rt])
    test_rts = set(rt_list[n_train_rt:])

    # 根据环类型划分 SMILES
    train_smiles = set()
    test_smiles = set()
    for rt in train_rts:
        train_smiles.update(rt_to_smiles[rt])
    for rt in test_rts:
        test_smiles.update(rt_to_smiles[rt])

    df_l10_train = df_l10[df_l10['SMILES'].isin(train_smiles)].reset_index(drop=True)
    df_l10_test = df_l10[df_l10['SMILES'].isin(test_smiles)].reset_index(drop=True)

    print(f"    train_rts={len(train_rts)}, test_rts={len(test_rts)}, "
          f"train_smiles={len(train_smiles)}, test_smiles={len(test_smiles)}, "
          f"l10_train={len(df_l10_train)}, l10_test={len(df_l10_test)}")

    # 打印测试集的环类型 (调试用)
    if fraction >= 0.5:
        print(f"    测试集环类型: {sorted(test_rts)}")

    return df_l10_train, df_l10_test, len(train_rts), len(test_rts)


def prepare_combined_csv(task_name, fraction, df_l10_train):
    """合并主训练数据 + lunci10 部分骨架"""
    task = TASK_MAP[task_name]
    target_col = TARGET_MAP[task_name]
    target_col_l10 = L10_COL_MAP[task_name]

    # 主训练数据
    clean_path = clean_dataset_csv(task['dataset_path'], target_col)
    df_main = pd.read_csv(clean_path)

    # 准备 lunci10 训练数据 (重命名列)
    df_l10 = df_l10_train.copy()
    df_l10 = df_l10.rename(columns={
        'SMILES': 'smiles',
        'Ring_Atoms': 'atom_on_ring',  # 关键: 重命名 Ring_Atoms → atom_on_ring
        target_col_l10: target_col,
    })
    # 确保列一致
    keep_cols = ['smiles', 'Ring_ID', 'Ring_Size', 'atom_on_ring', target_col]
    df_l10 = df_l10[[c for c in keep_cols if c in df_l10.columns]]

    # 合并
    df_combined = pd.concat([df_main, df_l10], ignore_index=True)
    return df_combined


def save_temp_csv(df, prefix, task_name, fraction):
    """保存临时 CSV"""
    os.makedirs(TMP_DIR, exist_ok=True)
    fname = f'{prefix}_{task_name}_{fraction}.csv'
    path = os.path.join(TMP_DIR, fname)
    df.to_csv(path, index=False)
    return path


def save_l10_test_csv(df_l10_test, task_name, fraction):
    """保存 lunci10 测试 CSV"""
    target_col = TARGET_MAP[task_name]
    target_col_l10 = L10_COL_MAP[task_name]

    df = df_l10_test.copy()
    df = df.rename(columns={
        'SMILES': 'smiles',
        'Ring_Atoms': 'atom_on_ring',  # 关键: 重命名 Ring_Atoms → atom_on_ring
        target_col_l10: target_col,
    })
    keep_cols = ['smiles', 'Ring_ID', 'Ring_Size', 'atom_on_ring', target_col]
    df = df[[c for c in keep_cols if c in df.columns]]
    return save_temp_csv(df, 'l10_test', task_name, fraction)


def run_single(task_name, model_name, fraction, device, params):
    """运行单个 fraction 的训练和评估"""
    print(f"\n  Fraction={int(fraction*100)}%")

    # 准备数据
    df_l10_train, df_l10_test, n_train_rts, n_test_rts = prepare_l10_data(task_name, fraction)
    combined_df = prepare_combined_csv(task_name, fraction, df_l10_train)
    combined_csv = save_temp_csv(combined_df, 'combined', task_name, fraction)
    test_csv = save_l10_test_csv(df_l10_test, task_name, fraction)

    target_col = TARGET_MAP[task_name]

    # 加载数据
    print(f"    Loading data...")
    train_data = load_adj_format(combined_csv, target_col, NVL, MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_data = load_adj_format(test_csv, target_col, NVL, EXT_MAX_ATOMS,
                               ring_flag_value=RING_FLAG_VALUE, device=device)

    n = train_data['n']
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = max(int(0.125 * n), 2)
    val_idx = torch.tensor(perm[:n_val], device=device)
    train_idx = torch.tensor(perm[n_val:], device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    # 训练
    t0 = time.time()
    model = train_model(model_name, params, train_data, train_idx, val_idx,
                       device, params['n_epochs'], params['patience'], SEED)
    train_time = time.time() - t0

    # 评估
    tr_r2, tr_mae, tr_rmse, _, _ = eval_model(model, train_data, train_idx,
                                              params['batch_size'], device)
    te_r2, te_mae, te_rmse, te_pred, te_true = eval_model(model, test_data, test_idx,
                                                           params['batch_size'], device)

    print(f"    Train R2={tr_r2:.4f} | Test R2={te_r2:.4f} MAE={te_mae:.4f} ({train_time:.0f}s)")

    # 保存结果
    frac_dir = os.path.join(RESULTS_DIR, task_name, f'frac_{int(fraction*100)}')
    os.makedirs(frac_dir, exist_ok=True)
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(frac_dir, 'predictions.csv'), index=False)

    import csv
    with open(os.path.join(frac_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['model', model_name])
        w.writerow(['task', task_name])
        w.writerow(['fraction', fraction])
        w.writerow(['n_train_rts', n_train_rts])
        w.writerow(['n_test_rts', n_test_rts])
        w.writerow(['train_r2', tr_r2])
        w.writerow(['train_mae', tr_mae])
        w.writerow(['test_r2', te_r2])
        w.writerow(['test_mae', te_mae])
        w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'task': task_name, 'fraction': fraction,
        'test_r2': te_r2, 'test_mae': te_mae,
        'n_train_rts': n_train_rts, 'n_test_rts': n_test_rts,
        'train_r2': tr_r2, 'train_time': train_time,
    }


def update_summary_csv(new_results):
    """更新 learning_curve_full.csv"""
    csv_path = os.path.join(RESULTS_DIR, 'learning_curve_full.csv')
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame(columns=['task', 'fraction', 'test_r2', 'test_mae',
                                       'n_train_rts', 'n_test_rts'])

    # 合并新结果 (去重: 同 task+fraction 只保留新结果)
    df_new = pd.DataFrame(new_results)
    cols = ['task', 'fraction', 'test_r2', 'test_mae', 'n_train_rts', 'n_test_rts']
    df_new = df_new[cols]

    # 去除旧结果中被新结果覆盖的行
    for _, row in df_new.iterrows():
        mask = (df_old['task'] == row['task']) & (df_old['fraction'] == row['fraction'])
        df_old = df_old[~mask]

    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(['task', 'fraction']).reset_index(drop=True)
    df_merged.to_csv(csv_path, index=False)
    print(f"\n更新汇总: {csv_path}")
    print(df_merged.to_string(index=False))
    return df_merged


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Learning Curve 补跑')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='HOMA,NICS_1zz,MBCO',
                        help='逗号分隔任务名')
    parser.add_argument('--fractions', type=str, default='0.5,0.7',
                        help='逗号分隔 fraction 列表')
    parser.add_argument('--rerun_all', action='store_true',
                        help='强制重跑所有指定 fraction (不跳过已有结果)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    params = dict(DEFAULT_PARAMS)
    tasks = args.tasks.split(',')
    fractions = [float(f) for f in args.fractions.split(',')]

    all_results = []
    for task_name in tasks:
        model_name = BEST_MODELS[task_name]
        print(f"\n{'='*60}")
        print(f"# {task_name} / {model_name}")
        print(f"{'='*60}")

        for frac in fractions:
            frac_pct = int(frac * 100)
            frac_dir = os.path.join(RESULTS_DIR, task_name, f'frac_{frac_pct}')
            summary_path = os.path.join(frac_dir, 'summary.csv')

            # 检查是否已有结果
            if os.path.exists(summary_path) and not args.rerun_all:
                print(f"\n  Fraction={frac_pct}%: 已有结果, 跳过 (用 --rerun_all 强制重跑)")
                # 读取已有结果
                df_s = pd.read_csv(summary_path)
                result = dict(zip(df_s['metric'], df_s['value']))
                all_results.append({
                    'task': task_name, 'fraction': frac,
                    'test_r2': float(result.get('test_r2', 0)),
                    'test_mae': float(result.get('test_mae', 0)),
                    'n_train_rts': int(result.get('n_train_rts', 0)),
                    'n_test_rts': int(result.get('n_test_rts', 0)),
                })
                continue

            try:
                result = run_single(task_name, model_name, frac, device, params)
                all_results.append(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [失败] {task_name} frac={frac}: {e}")

    # 更新汇总 CSV
    if all_results:
        df_merged = update_summary_csv(all_results)

        # 重新生成图表
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
            for i, task_name in enumerate(['HOMA', 'NICS_1zz', 'MBCO']):
                sub = df_merged[df_merged['task'] == task_name].sort_values('fraction')
                if len(sub) == 0:
                    continue
                ax = axes[i]
                ax.plot(sub['fraction'], sub['test_r2'], 'o-', linewidth=2, markersize=8)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Fraction of New Ring Families')
                ax.set_ylabel('Test R²')
                ax.set_title(task_name)
                ax.grid(True, alpha=0.3)
                # 标注数值
                for _, row in sub.iterrows():
                    ax.annotate(f"{row['test_r2']:.3f}",
                                (row['fraction'], row['test_r2']),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', fontsize=8)
            plt.tight_layout()
            fig_path = os.path.join(RESULTS_DIR, 'learning_curve.png')
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
            print(f"\n图表已保存: {fig_path}")
        except Exception as e:
            print(f"图表生成失败: {e}")


if __name__ == '__main__':
    main()
