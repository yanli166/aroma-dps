"""
多种子 × 5折CV 实验脚本 — 修复版

修复项:
  - M1: recover_from_summaries 按 layer 区分路径 pattern (Layer1/2 用 2 层, Layer3 用 3 层)
  - M2: recover_from_summaries 添加 cv_r2_mean -> cv_r2 列名映射
  - M3: Layer 3 增加 'none' 编码 (无环信息 ablation 对照)
  - m3: 统一 CSV 列名 (使用 common.constants.PER_SEED_COLUMNS)

用法:
  python run_multiseed.py --layer 1 --gpu 0 --seeds 42,123,456,789,2024
  python run_multiseed.py --layer 2 --gpu 0 --seeds 42,123,456,789,2024 --models GNN,GIN
  python run_multiseed.py --layer 3 --gpu 0 --seeds 42,123,456,789,2024
  python run_multiseed.py --aggregate_only
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))
from common.constants import ORIG_MODELS_ROOT, PER_SEED_COLUMNS, METRIC_COLS_FOR_AGG
sys.path.insert(0, ORIG_MODELS_ROOT)

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def _save_incremental(output_root, results, completed_seed=None):
    """增量保存逐种子结果"""
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    # m3: 统一列顺序
    df = pd.DataFrame(results)
    # 确保列存在
    for col in PER_SEED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in PER_SEED_COLUMNS if c in df.columns]]
    df.to_csv(csv_path, index=False)
    if completed_seed is not None:
        ckpt_path = os.path.join(output_root, '.completed_seeds.txt')
        with open(ckpt_path, 'a') as f:
            f.write(f"{completed_seed}\n")


def _load_completed_seeds(output_root):
    """从 checkpoint 文件加载已完成的种子"""
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    results = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                results = df.to_dict('records')
        except Exception:
            pass

    ckpt_path = os.path.join(output_root, '.completed_seeds.txt')
    done_seeds = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            done_seeds = set(int(line.strip()) for line in f if line.strip())

    if not done_seeds and results:
        done_seeds = set(df['seed'].unique()) if 'seed' in df.columns else set()

    return results, done_seeds


# ============== Layer 1: 传统 ML ==============
def run_layer1(seeds, tasks, gpu, output_root):
    """多种子 Layer 1 ML 实验"""
    from baseline_traditional_ml.code.ml_train_eval import build_models, FoldAwareFactory, run_one_model_seeded
    from common.tasks import TASKS, canonical_splits
    from common.features import build_fingerprint_matrix

    task_list = TASKS if tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in tasks.split(',')]

    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#'*60}\n# Layer 1 ML | Seed={seed}\n{'#'*60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            name = task['name']
            print(f"\n{'='*60}\n任务: {name} (seed={seed})\n{'='*60}", flush=True)
            from common.tasks import clean_dataset_csv
            clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
            df = pd.read_csv(clean_path)
            smiles_list = df['smiles'].tolist()
            y = df[task['target_col']].astype(float).to_numpy()
            n_total = len(smiles_list)

            # C1: 先在原始 n 上做划分
            splits = canonical_splits(n_total, seed=seed)

            # C1: raise_on_invalid=True (默认)
            X, valid = build_fingerprint_matrix(smiles_list, df=df)
            assert valid.all()

            models = build_models(seed=seed, gpu=gpu)
            for model_name, model in models.items():
                # C3: FoldAwareFactory
                fold_factory = FoldAwareFactory(model, seed)
                try:
                    res = run_one_model_seeded(model_name, fold_factory, X, y, splits, name,
                                               seed_dir, n_total, seed)
                    res['seed'] = seed
                    res['encoding'] = 'label'  # m3: 统一加 encoding 列
                    all_seed_results.append(res)
                except Exception as e:
                    print(f"  [{model_name}|seed={seed}|{name}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Layer 2: GNN ==============
def run_layer2(seeds, tasks, models, gpu, output_root, n_epochs, patience):
    """多种子 Layer 2 GNN 实验"""
    from baseline_gnn.code.gnn_train_eval import run_model_on_task, DEFAULT_PARAMS, CUSTOM_MODELS, PYG_MODELS
    from common.tasks import TASKS

    task_list = TASKS if tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in tasks.split(',')]
    all_models = list(CUSTOM_MODELS.keys()) + PYG_MODELS
    model_list = all_models if models == 'all' else models.split(',')

    import torch
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = n_epochs
    params['patience'] = patience

    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#'*60}\n# Layer 2 GNN | Seed={seed} | GPU={gpu}\n{'#'*60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            for m in model_list:
                try:
                    res = run_model_on_task(m, task, params, device, seed_dir,
                                            n_epochs, patience, seed)
                    res['seed'] = seed
                    res['encoding'] = 'label'  # m3: 统一加 encoding 列
                    all_seed_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{m}|seed={seed}|{task['name']}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Layer 3: 环编码 ==============
def run_layer3(seeds, tasks, models, gpu, output_root, n_epochs, patience):
    """多种子 Layer 3 环编码实验"""
    from ring_encoding_ablation.code.ring_train_eval import run_experiment, DEFAULT_PARAMS, CUSTOM_MODELS, ENCODINGS
    from common.tasks import TASKS

    task_list = TASKS if tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in tasks.split(',')]
    model_list = list(CUSTOM_MODELS.keys()) if models == 'all' else models.split(',')

    import torch
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = n_epochs
    params['patience'] = patience

    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#'*60}\n# Layer 3 Ring Encoding | Seed={seed} | GPU={gpu}\n{'#'*60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for model_name in model_list:
            # M3: ENCODINGS 包含 'none', 加上 'combined'
            for encoding in ENCODINGS + ['combined']:
                for task in task_list:
                    try:
                        res = run_experiment(model_name, encoding, task, params, device,
                                             seed_dir, n_epochs, patience, seed)
                        res['seed'] = seed
                        all_seed_results.append(res)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  [{model_name}|{encoding}|seed={seed}|{task['name']}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== 聚合 ==============
def aggregate_results(output_root, layer):
    """聚合多种子结果: mean ± std"""
    per_seed_csv = os.path.join(output_root, 'per_seed_results.csv')
    if not os.path.exists(per_seed_csv):
        print(f"未找到 {per_seed_csv}, 请先运行实验")
        return None

    df = pd.read_csv(per_seed_csv)
    print(f"\n{'='*60}\n{layer} 多种子聚合 (seeds={sorted(df['seed'].unique())})\n{'='*60}")

    # m3: 统一聚合列
    metric_cols = [c for c in METRIC_COLS_FOR_AGG if c in df.columns]

    # 分组键
    if layer == 'layer3_ring':
        group_keys = ['task', 'model', 'encoding']
    else:
        group_keys = ['task', 'model']

    agg = df.groupby(group_keys)[metric_cols].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]

    summary_csv = os.path.join(output_root, f'all_{layer}_summary.csv')
    agg.to_csv(summary_csv, index=False)
    print(f"聚合结果保存: {summary_csv}")

    display_cols = [c for c in ['task', 'model', 'encoding', 'test_r2_mean', 'test_r2_std',
                                'test_mae_mean', 'test_rmse_mean', 'train_time_sec_mean'] if c in agg.columns]
    print(agg[display_cols].to_string(index=False))
    return agg


def recover_from_summaries(output_root, layer):
    """M1+M2 修复: 从 individual summary.csv 重建 per_seed_results.csv

    M1: 按 layer 区分路径 pattern
        - Layer 1/2: seed_X/TASK/MODEL/summary.csv (2 层子目录)
        - Layer 3:   seed_X/TASK/MODEL_ENCODING/summary.csv (3 层子目录)
    M2: cv_r2_mean -> cv_r2 列名映射
    """
    import glob

    is_layer3 = 'layer3' in layer or 'ring' in layer

    # M1: 按 layer 区分 pattern
    if is_layer3:
        pattern = os.path.join(output_root, 'seed_*', '*', '*', 'summary.csv')
    else:
        pattern = os.path.join(output_root, 'seed_*', '*', 'summary.csv')

    files = glob.glob(pattern)
    if not files:
        print(f"未找到 summary.csv 文件 in {output_root} (pattern: {pattern})")
        return None

    results = []
    for f in files:
        try:
            parts = f.replace(output_root, '').strip('/').split('/')
            seed = int(parts[0].replace('seed_', ''))
            task = parts[1]
            model_or_tag = parts[2]

            df = pd.read_csv(f, header=None, names=['metric', 'value'])
            d = dict(zip(df['metric'], df['value']))
            res = {'seed': seed, 'task': task}

            if is_layer3:
                # model_encoding 格式 (如 GAT_label, GraphSAGE_pool, GNN_none)
                if '_' in model_or_tag:
                    model, encoding = model_or_tag.rsplit('_', 1)
                    res['model'] = model
                    res['encoding'] = encoding
                else:
                    res['model'] = model_or_tag
                    res['encoding'] = 'unknown'
            else:
                res['model'] = model_or_tag
                res['encoding'] = 'label'  # m3: 统一加 encoding 列

            # M2: cv_r2_mean -> cv_r2 映射
            if 'cv_r2_mean' in d:
                res['cv_r2'] = float(d['cv_r2_mean'])
            elif 'cv_r2' in d:
                res['cv_r2'] = float(d['cv_r2'])

            for k in ['n', 'cv_r2_std', 'cv_mae', 'cv_rmse',
                       'train_r2', 'train_mae', 'train_rmse',
                       'val_r2', 'val_mae', 'val_rmse',
                       'test_r2', 'test_mae', 'test_rmse',
                       'train_time_sec']:
                if k in d:
                    try:
                        res[k] = float(d[k])
                    except (ValueError, TypeError):
                        pass
            results.append(res)
        except Exception as e:
            print(f"  跳过 {f}: {e}")

    df = pd.DataFrame(results)
    # m3: 统一列顺序
    for col in PER_SEED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in PER_SEED_COLUMNS if c in df.columns]]
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"恢复 {len(df)} 条结果 -> {csv_path}")
    return df


def _merge_parts_and_aggregate(results_root=None):
    """合并 part 目录的 per_seed_results.csv, 然后聚合所有层"""
    if results_root is None:
        results_root = os.path.join(PROJ_ROOT, 'code_end', 'results')

    # Layer 2: 合并 part1/2/3/4 -> layer2_gnn
    layer2_main = os.path.join(results_root, 'layer2_gnn')
    layer2_parts = [f'layer2_gnn_part{i}' for i in range(1, 5)]
    _merge_parts(results_root, layer2_main, layer2_parts, 'layer2_gnn')

    # Layer 3: 合并 part1/2/3/4 -> layer3_ring
    layer3_main = os.path.join(results_root, 'layer3_ring')
    layer3_parts = [f'layer3_ring_part{i}' for i in range(1, 5)]
    _merge_parts(results_root, layer3_main, layer3_parts, 'layer3_ring')

    # 聚合所有层
    for name in ['layer1_ml', 'layer2_gnn', 'layer3_ring']:
        out_root = os.path.join(results_root, name)
        csv_path = os.path.join(out_root, 'per_seed_results.csv')
        if os.path.exists(csv_path):
            aggregate_results(out_root, name)
        else:
            print(f"\n{name}: 无 per_seed_results.csv, 尝试从 summary.csv 恢复...")
            df = recover_from_summaries(out_root, name)
            if df is not None and len(df) > 0:
                aggregate_results(out_root, name)


def _merge_parts(results_root, main_dir, part_names, layer_name):
    """合并多个 part 目录的 per_seed_results.csv 到主目录"""
    os.makedirs(main_dir, exist_ok=True)
    main_csv = os.path.join(main_dir, 'per_seed_results.csv')

    all_dfs = []
    for part in part_names:
        part_dir = os.path.join(results_root, part)
        part_csv = os.path.join(part_dir, 'per_seed_results.csv')
        if os.path.exists(part_csv):
            df = pd.read_csv(part_csv)
            print(f"  {part}: {len(df)} 行 (per_seed_results.csv)")
            all_dfs.append(df)
        else:
            df = recover_from_summaries(part_dir, layer_name)
            if df is not None and len(df) > 0:
                print(f"  {part}: {len(df)} 行 (从 summary.csv 恢复)")
                all_dfs.append(df)
            else:
                print(f"  {part}: 无结果")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        # 去重: 同 (seed, task, model, encoding) 保留最后
        dedup_cols = ['seed', 'task', 'model']
        if 'encoding' in merged.columns:
            dedup_cols.append('encoding')
        merged = merged.drop_duplicates(subset=dedup_cols, keep='last')
        # m3: 统一列顺序
        for col in PER_SEED_COLUMNS:
            if col not in merged.columns:
                merged[col] = pd.NA
        merged = merged[[c for c in PER_SEED_COLUMNS if c in merged.columns]]
        merged.to_csv(main_csv, index=False)
        print(f"  -> 合并去重后 {len(merged)} 行 -> {main_csv}")
    else:
        print(f"  -> 无可合并数据")


# ============== 主入口 ==============
def main():
    parser = argparse.ArgumentParser(description='多种子 × 5折CV 实验 (修复版)')
    parser.add_argument('--layer', type=int, choices=[1, 2, 3], default=None)
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--models', type=str, default='all')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--aggregate_only', action='store_true')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    print(f"种子: {seeds}")

    if args.recover:
        layer_map = {1: 'layer1_ml', 2: 'layer2_gnn', 3: 'layer3_ring'}
        if args.layer:
            dirs = [args.output_root or os.path.join(PROJ_ROOT, 'code_end', 'results', layer_map[args.layer])]
        else:
            import glob
            dirs = glob.glob(os.path.join(PROJ_ROOT, 'code_end', 'results', 'layer*'))
        for d in sorted(dirs):
            if os.path.isdir(d):
                layer_name = os.path.basename(d)
                print(f"\n恢复 {layer_name}...")
                recover_from_summaries(d, layer_name)
        return

    if args.aggregate_only:
        _merge_parts_and_aggregate()
        return

    if args.layer is None:
        print("请指定 --layer (1/2/3) 或 --aggregate_only")
        return

    layer_map = {1: ('layer1_ml', run_layer1), 2: ('layer2_gnn', run_layer2), 3: ('layer3_ring', run_layer3)}
    layer_name, run_fn = layer_map[args.layer]
    output_root = args.output_root or os.path.join(PROJ_ROOT, 'code_end', 'results', layer_name)
    os.makedirs(output_root, exist_ok=True)

    t0 = time.time()
    if args.layer == 1:
        run_fn(seeds, args.tasks, args.gpu, output_root)
    elif args.layer == 2:
        run_fn(seeds, args.tasks, args.models, args.gpu, output_root, args.n_epochs, args.patience)
    elif args.layer == 3:
        run_fn(seeds, args.tasks, args.models, args.gpu, output_root, args.n_epochs, args.patience)

    elapsed = time.time() - t0
    print(f"\n{layer_name} 完成, 耗时 {elapsed/60:.1f} 分钟")

    aggregate_results(output_root, layer_name)


if __name__ == '__main__':
    main()
