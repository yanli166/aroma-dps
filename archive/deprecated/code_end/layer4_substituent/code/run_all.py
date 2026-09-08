"""
Layer 4 主入口: 运行所有8种方法 × 3个任务, 以lunci6为最终验证

用法:
  python run_all.py --gpu 0 --methods all
  python run_all.py --gpu 0 --methods m1,m6,m7
  python run_all.py --gpu 0 --methods m1,m2 --tasks HOMA,NICS_1zz

输出:
  - 每方法×任务: output_root/{task}/{method}/ (cv_results.csv, summary.csv,
    test_predictions.csv, lunci6_predictions.csv, parity_plot.png, best_model.pth)
  - layer4_summary.csv: 所有方法×任务汇总
  - layer4_vs_layer2_comparison.csv: 8种方法 vs Layer 2 GNN 基线对比表
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

sys.path.insert(0, '_PROJ_ROOT + "/code_end"')
sys.path.insert(0, '_PROJ_ROOT + "/unified_models"')

from common.tasks import TASKS, DEFAULT_SEED
from layer4_substituent.code.train_eval import DEFAULT_PARAMS, run_method

# 8 种方法名称映射: (中文名, 基础模型类型)
METHODS = {
    'm1': ('Hammett_σ嵌入', 'gnn'),
    'm2': ('单调性约束', 'gnn'),
    'm3': ('分层交叉注意力', 'gnn'),
    'm4': ('双通道消息传递', 'gnn'),
    'm5': ('SSL预训练', 'gnn'),
    'm6': ('位置编码', 'gnn'),
    'm7': ('扰动学习', 'gnn'),
    'm8': ('多尺度池化', 'gnn'),
}

# Layer 2 GNN 基线结果 (用于对比, Layer 4 全部基于 GNN)
LAYER2_BASELINE_CSV = '_PROJ_ROOT + "/code_end"/results/layer2_gnn/all_layer2_gnn_summary.csv'


def load_layer2_baseline():
    """加载 Layer 2 GNN 基线结果用于对比

    Layer 2 汇总 CSV 列名使用聚合形式 (test_r2_mean 等),
    本函数筛选 model=='GNN' 的行作为对比基线。
    """
    if not os.path.exists(LAYER2_BASELINE_CSV):
        print(f"  [警告] Layer 2 基线文件不存在: {LAYER2_BASELINE_CSV}")
        return None
    try:
        df = pd.read_csv(LAYER2_BASELINE_CSV)
        if 'model' in df.columns:
            df = df[df['model'] == 'GNN'].copy()
        return df
    except Exception as e:
        print(f"  [警告] 加载 Layer 2 基线失败: {e}")
        return None


def build_comparison_table(layer4_df, baseline_df):
    """构建 Layer 4 方法 vs Layer 2 GNN 基线对比表

    Args:
        layer4_df:   Layer 4 汇总结果 (run_method 返回的字典列表)
        baseline_df: Layer 2 GNN 基线 (load_layer2_baseline 返回)

    Returns:
        对比表 DataFrame, 每行一个 方法×任务, 包含:
            task, method, method_name,
            test_r2, test_mae, test_rmse, lunci6_r2, lunci6_mae, lunci6_rmse,
            baseline_test_r2, baseline_test_mae, baseline_test_rmse,
            delta_test_r2 (Layer4 - Layer2)
    """
    rows = []
    for _, r in layer4_df.iterrows():
        task = r['task']
        method = r['method']
        method_name = METHODS.get(method, (method, ''))[0]
        row = {
            'task': task,
            'method': method,
            'method_name': method_name,
            'test_r2': r['test_r2'],
            'test_mae': r['test_mae'],
            'test_rmse': r['test_rmse'],
            'lunci6_r2': r['lunci6_r2'],
            'lunci6_mae': r['lunci6_mae'],
            'lunci6_rmse': r['lunci6_rmse'],
        }
        # 匹配 Layer 2 GNN 基线 (同任务)
        if baseline_df is not None and 'task' in baseline_df.columns:
            base = baseline_df[baseline_df['task'] == task]
            if len(base) > 0:
                b = base.iloc[0]
                # Layer 2 汇总 CSV 使用 test_r2_mean 等聚合列名
                b_r2 = b.get('test_r2_mean', b.get('test_r2', np.nan))
                b_mae = b.get('test_mae_mean', b.get('test_mae', np.nan))
                b_rmse = b.get('test_rmse_mean', b.get('test_rmse', np.nan))
                row['baseline_test_r2'] = b_r2
                row['baseline_test_mae'] = b_mae
                row['baseline_test_rmse'] = b_rmse
                row['delta_test_r2'] = r['test_r2'] - b_r2 if not np.isnan(b_r2) else np.nan
            else:
                row['baseline_test_r2'] = np.nan
                row['baseline_test_mae'] = np.nan
                row['baseline_test_rmse'] = np.nan
                row['delta_test_r2'] = np.nan
        else:
            row['baseline_test_r2'] = np.nan
            row['baseline_test_mae'] = np.nan
            row['baseline_test_rmse'] = np.nan
            row['delta_test_r2'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Layer 4: 取代基效应建模 (8种方法 × 3任务, lunci6 集外验证)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--methods', type=str, default='all',
                        help='all 或 逗号分隔: m1,m2,...,m8')
    parser.add_argument('--tasks', type=str, default='all',
                        help='all 或 逗号分隔: HOMA,NICS_1zz,MBCO')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/results/layer4_substituent')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # cudnn 确定性 (与 Layer 2/3 一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    # 解析方法与任务
    method_list = (list(METHODS.keys()) if args.methods == 'all'
                   else [m.strip() for m in args.methods.split(',')])
    for m in method_list:
        if m not in METHODS:
            raise ValueError(f"未知方法: {m} (可选: {list(METHODS.keys())})")
    task_list = (TASKS if args.tasks == 'all'
                 else [next(t for t in TASKS if t['name'] == n.strip())
                       for n in args.tasks.split(',')])

    print(f"方法: {method_list}")
    print(f"任务: {[t['name'] for t in task_list]}")
    print(f"输出目录: {args.output_dir}")

    # 运行所有 方法 × 任务
    all_results = []
    for task in task_list:
        for method in method_list:
            try:
                res = run_method(method, task, params, device, args.output_dir,
                                 args.n_epochs, args.patience, args.seed)
                all_results.append(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{method}|{task['name']}] 失败: {e}")

    # 汇总结果
    if all_results:
        cols = ['task', 'method', 'n', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
                'test_r2', 'test_mae', 'test_rmse',
                'lunci6_r2', 'lunci6_mae', 'lunci6_rmse', 'train_time_sec']
        df_out = pd.DataFrame(all_results)[cols]
        print(f"\n{'='*60}\nLayer 4 取代基效应建模汇总\n{'='*60}")
        print(df_out.to_string(index=False))
        out_csv = os.path.join(args.output_dir, 'layer4_summary.csv')
        df_out.to_csv(out_csv, index=False)
        print(f"\n汇总结果已保存: {out_csv}")

        # 生成对比表 (8种方法 vs Layer 2 GNN 基线)
        baseline_df = load_layer2_baseline()
        comp = build_comparison_table(df_out, baseline_df)
        comp_csv = os.path.join(args.output_dir, 'layer4_vs_layer2_comparison.csv')
        comp.to_csv(comp_csv, index=False)
        print(f"对比表已保存: {comp_csv}")

        # 打印对比表 (按任务分组, 按 test_r2 降序)
        if 'delta_test_r2' in comp.columns:
            print(f"\n{'='*60}\nLayer 4 vs Layer 2 (GNN) 对比\n{'='*60}")
            for task in comp['task'].unique():
                sub = comp[comp['task'] == task].sort_values('test_r2', ascending=False)
                print(f"\n--- {task} ---")
                show_cols = ['method', 'test_r2', 'baseline_test_r2', 'delta_test_r2',
                             'lunci6_r2']
                print(sub[show_cols].to_string(index=False))
    else:
        print("\n无结果可汇总 (所有方法×任务均失败)")


if __name__ == '__main__':
    main()
