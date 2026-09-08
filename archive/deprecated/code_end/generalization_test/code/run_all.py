"""
泛化能力验证: 主入口 (适配 0716 新数据)

对三个任务 (HOMA / NICS_1zz / MBCO) 的 top3 模型执行四种泛化测试:
  1. scaffold: 骨架拆分 (DeepChem 风格)
  2. ring_type: 芳环类型拆分 (各主要环类型逐一留出)
  3. lunci6: lunci6 集外测试
  4. lunci78: lunci78 集外测试

Top3 模型 (基于此前实验):
  HOMA     : MPNN, GraphSAGE, GNN
  NICS_1zz : GNN, MPNN, GraphSAGE
  MBCO     : GraphSAGE, GAT, GIN
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

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))
from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)

from common.tasks import TASKS, DEFAULT_SEED
from common.graph_data import load_adj_format
from generalization_test.code.train_eval import DEFAULT_PARAMS, run_single_test
from generalization_test.code.splits import (
    scaffold_split, ring_type_split, prepare_external_test_csv, report_distributions)

TOP3_MODELS = {
    'HOMA':     ['MPNN', 'GraphSAGE', 'GNN'],
    'NICS_1zz': ['GNN', 'MPNN', 'GraphSAGE'],
    'MBCO':     ['GraphSAGE', 'GAT', 'GIN'],
}

NVL, MAX_ATOMS = 60, 75
EXT_MAX_ATOMS = 85  # 外部测试集允许更大的分子 (lunci78 最大 77 原子含 H)
RING_FLAG_VALUE = 10  # label 编码


def run_scaffold_test(task, models, params, device, output_root, n_epochs, patience, seed):
    """骨架拆分测试 (DeepChem)"""
    print(f"\n{'#'*60}\n# 骨架拆分测试 (DeepChem): {task['name']}\n{'#'*60}")

    from common.tasks import clean_dataset_csv
    clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
    data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                           ring_flag_value=RING_FLAG_VALUE, device=device)
    df = pd.read_csv(clean_path)
    report_distributions(df)

    result = scaffold_split(df, smiles_col='smiles', seed=seed, min_test_size=30)
    train_idx, val_idx, test_idx, holdout = result
    if train_idx is None:
        print(f"  骨架拆分失败, 跳过")
        return []

    split_info = {'data': data, 'train_idx': train_idx, 'val_idx': val_idx,
                  'test_idx': test_idx, 'holdout': holdout}

    results = []
    for model_name in models:
        try:
            res = run_single_test(model_name, task, params, device, 'scaffold',
                                  split_info, output_root, n_epochs, patience, seed)
            results.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{model_name}/scaffold/{task['name']}] 失败: {e}")
    return results


def run_ring_type_test(task, models, params, device, output_root, n_epochs, patience, seed):
    """芳环类型拆分测试"""
    print(f"\n{'#'*60}\n# 芳环类型拆分测试: {task['name']}\n{'#'*60}")

    from common.tasks import clean_dataset_csv
    clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
    data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                           ring_flag_value=RING_FLAG_VALUE, device=device)
    df = pd.read_csv(clean_path)

    from generalization_test.code.ring_utils import get_ring_type_distribution
    rt_dist = get_ring_type_distribution(df['smiles'].tolist())
    candidates = [(rt, cnt) for rt, cnt in rt_dist.most_common(15)
                  if rt not in ('invalid', 'non_aromatic', 'unknown', '') and cnt >= 50]
    if not candidates:
        print(f"  无足够环类型, 跳过")
        return []

    candidates = candidates[:5]
    print(f"  将测试 {len(candidates)} 种环类型: {[rt for rt, _ in candidates]}")

    all_results = []
    for ring_type, cnt in candidates:
        print(f"\n  --- 留出环类型: {ring_type} ({cnt} 样本) ---")
        result = ring_type_split(df, smiles_col='smiles', holdout_ring_type=ring_type,
                                 val_ratio=0.125, seed=seed, min_test_size=30)
        train_idx, val_idx, test_idx, rt = result
        if train_idx is None:
            continue

        print(f"    train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        split_info = {'data': data, 'train_idx': train_idx, 'val_idx': val_idx,
                      'test_idx': test_idx, 'holdout': ring_type}
        for model_name in models:
            try:
                res = run_single_test(model_name, task, params, device, 'ring_type',
                                      split_info, output_root, n_epochs, patience, seed)
                all_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"    [{model_name}/ring_type:{ring_type}/{task['name']}] 失败: {e}")
    return all_results


def run_external_test(task, models, params, device, output_root, n_epochs, patience, seed,
                      test_source='lunci6'):
    """lunci6/lunci78 集外测试"""
    print(f"\n{'#'*60}\n# {test_source} 集外测试: {task['name']}\n{'#'*60}")

    # 主数据集 (训练+验证)
    train_data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n = train_data['n']

    # 87.5/12.5 train/val split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(0.125 * n)
    val_idx = torch.tensor(perm[:n_val], device=device)
    train_idx = torch.tensor(perm[n_val:], device=device)

    # 外部测试集 (使用更大的 max_atoms 容纳 lunci78 的大分子)
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source=test_source)
    test_data = load_adj_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    print(f"  主数据集: train={len(train_idx)}, val={len(val_idx)}")
    print(f"  {test_source} 测试集: {len(test_idx)} 样本")

    split_info = {'train_data': train_data, 'test_data': test_data,
                  'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
                  'holdout': f'{test_source}_external'}

    results = []
    for model_name in models:
        try:
            res = run_single_test(model_name, task, params, device, test_source,
                                  split_info, output_root, n_epochs, patience, seed)
            results.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{model_name}/{test_source}/{task['name']}] 失败: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description='泛化能力验证 (0716新数据)')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/results/generalization')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_types', type=str, default='scaffold,ring_type,lunci6,lunci78,lunci10',
                        help='逗号分隔: scaffold,ring_type,lunci6,lunci78,lunci10')
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]
    test_types = args.test_types.split(',')

    all_results = []
    for task in task_list:
        models = TOP3_MODELS[task['name']]
        print(f"\n{'='*60}\n任务: {task['name']} | Top3: {models}\n{'='*60}")

        if 'scaffold' in test_types:
            all_results.extend(run_scaffold_test(task, models, params, device,
                                                args.output_dir, args.n_epochs, args.patience, args.seed))
        if 'ring_type' in test_types:
            all_results.extend(run_ring_type_test(task, models, params, device,
                                                  args.output_dir, args.n_epochs, args.patience, args.seed))
        if 'lunci6' in test_types:
            all_results.extend(run_external_test(task, models, params, device,
                                                 args.output_dir, args.n_epochs, args.patience, args.seed,
                                                 test_source='lunci6'))
        if 'lunci78' in test_types:
            all_results.extend(run_external_test(task, models, params, device,
                                                 args.output_dir, args.n_epochs, args.patience, args.seed,
                                                 test_source='lunci78'))
        if 'lunci10' in test_types:
            all_results.extend(run_external_test(task, models, params, device,
                                                 args.output_dir, args.n_epochs, args.patience, args.seed,
                                                 test_source='lunci10'))

    if all_results:
        cols = ['task', 'test_type', 'holdout', 'model', 'n_train', 'n_test',
                'train_r2', 'test_r2', 'test_mae', 'test_rmse', 'train_time_sec']
        df_out = pd.DataFrame(all_results)[cols]
        print(f"\n{'='*60}\n泛化验证汇总\n{'='*60}")
        print(df_out.to_string(index=False))
        out_csv = os.path.join(args.output_dir, 'generalization_summary.csv')
        df_out.to_csv(out_csv, index=False)
        print(f"\n结果已保存: {out_csv}")


if __name__ == '__main__':
    main()
