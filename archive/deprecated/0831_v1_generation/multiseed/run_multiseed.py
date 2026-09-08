"""
多种子 × 5折CV 实验脚本 — 四阶段版

支持 4 个阶段的多种子运行,统一 CSV 列名 (PER_SEED_COLUMNS, 'config' 列替代 'encoding')。

种子列表: [42, 123, 456, 789, 2024]

阶段:
  - 1ml:  Stage1 传统ML (调用 stage1_representation_comparison/code/traditional_ml.py)
  - 1gnn: Stage1 GNN基线 (调用 stage1_representation_comparison/code/gnn_baseline.py)
  - 2:    Stage2 环条件化消融 (调用 stage2_ring_conditioning/code/ring_conditioning_ablation.py)
  - 3:    Stage3 Ring Masking 预训练 (调用 stage3_mask_pretraining/code/run_pretrain_eval.py)
  - 4:    Stage4 跨架构验证 (调用 stage4_cross_architecture/code/cross_arch_eval.py)

输出目录结构:
  - Stage1 ML:   results/stage1_traditional_ml/seed_X/TASK/MODEL/
  - Stage1 GNN:  results/stage1_gnn/seed_X/TASK/MODEL/
  - Stage2:      results/stage2_ring_conditioning/seed_X/TASK/BACKBONE_CONFIG/
  - Stage3:      results/stage3_mask_pretraining/seed_X/TASK/TRAINING_MODE/
  - Stage4:      results/stage4_cross_arch/seed_X/TASK/BACKBONE_CONFIG/

用法:
  python multiseed/run_multiseed.py --stage 1ml --gpu 0
  python multiseed/run_multiseed.py --stage 1gnn --gpu 0
  python multiseed/run_multiseed.py --stage 2 --gpu 0
  python multiseed/run_multiseed.py --stage 3 --gpu 0 --pretrain_epochs 100
  python multiseed/run_multiseed.py --stage 4 --gpu 0
  python multiseed/run_multiseed.py --aggregate_only
  python multiseed/run_multiseed.py --recover --stage 2
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd

# 将 last_end_code 根目录加入 sys.path, 以便 import common / models / stageX_*
LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import PER_SEED_COLUMNS, METRIC_COLS_FOR_AGG

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

# 各阶段默认输出子目录名
STAGE_DIRS = {
    '1ml':  'stage1_traditional_ml',
    '1gnn': 'stage1_gnn',
    '2':    'stage2_ring_conditioning',
    '3':    'stage3_mask_pretraining',
    '4':    'stage4_cross_arch',
}


def _default_output_root(stage):
    """获取某阶段的默认输出目录"""
    return os.path.join(LAST_END_ROOT, 'results', STAGE_DIRS[stage])


def _parse_tasks(tasks):
    """解析任务列表: 'all' -> 全部, 否则按逗号分隔"""
    from common.tasks import TASKS
    if tasks == 'all':
        return list(TASKS)
    return [next(t for t in TASKS if t['name'] == n) for n in tasks.split(',')]


def _save_incremental(output_root, results, completed_seed=None):
    """增量保存逐种子结果 (统一列顺序 PER_SEED_COLUMNS)"""
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    df = pd.DataFrame(results)
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
    """从 checkpoint 文件加载已完成的种子 (断点续跑)"""
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


def _find_stage1_gnn_results(stage2_output_root):
    """定位 Stage1 GNN 结果 CSV (供 Stage2 选最佳 backbone)"""
    results_root = os.path.dirname(stage2_output_root)
    for name in ['per_seed_results.csv', 'all_stage1_gnn_summary.csv']:
        p = os.path.join(results_root, 'stage1_gnn', name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"未找到 Stage1 GNN 结果 CSV (需先运行 --stage 1gnn), 已尝试: "
        f"{results_root}/stage1_gnn/{{per_seed_results.csv,all_stage1_gnn_summary.csv}}")


def _find_stage2_results(stage3_output_root):
    """定位 Stage2 结果 CSV (供 Stage3 选最佳配置)"""
    results_root = os.path.dirname(stage3_output_root)
    for name in ['per_seed_results.csv', 'all_stage2_summary.csv']:
        p = os.path.join(results_root, 'stage2_ring_conditioning', name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"未找到 Stage2 结果 CSV (需先运行 --stage 2), 已尝试: "
        f"{results_root}/stage2_ring_conditioning/{{per_seed_results.csv,all_stage2_summary.csv}}")


# ============== Stage 1: 传统 ML ==============
def run_stage1_ml(seeds, tasks, gpu, output_root):
    """多种子 Stage1 传统ML 实验

    调用 stage1_representation_comparison/code/traditional_ml.py 的核心函数。
    输出目录: results/stage1_traditional_ml/seed_X/TASK/MODEL/
    """
    from stage1_representation_comparison.code.traditional_ml import (
        build_models, FoldAwareFactory, run_one_model_seeded)
    from common.tasks import canonical_splits, clean_dataset_csv
    from common.features import build_fingerprint_matrix

    task_list = _parse_tasks(tasks)
    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#' * 60}\n# Stage1 ML | Seed={seed}\n{'#' * 60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            name = task['name']
            print(f"\n{'=' * 60}\n任务: {name} (seed={seed})\n{'=' * 60}", flush=True)
            clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
            df = pd.read_csv(clean_path)
            smiles_list = df['smiles'].tolist()
            y = df[task['target_col']].astype(float).to_numpy()
            n_total = len(smiles_list)

            # C1: 先在原始 n 上做划分, 保证与 GNN 索引空间一致
            # parent-molecule-level split: 同一分子的多个环级样本进入同一 split
            splits = canonical_splits(n_total, seed=seed, groups=df['smiles'].values)
            X, valid = build_fingerprint_matrix(smiles_list, df=df)
            assert valid.all(), "无效SMILES应已在build_fingerprint_matrix中raise"

            models = build_models(seed=seed, gpu=gpu)
            for model_name, model in models.items():
                fold_factory = FoldAwareFactory(model, seed)
                try:
                    res = run_one_model_seeded(model_name, fold_factory, X, y, splits,
                                               name, seed_dir, n_total, seed)
                    res['seed'] = seed
                    res['config'] = 'fingerprint'  # 统一 config 列 (传统ML固定表示)
                    all_seed_results.append(res)
                except Exception as e:
                    print(f"  [{model_name}|seed={seed}|{name}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Stage 1: GNN 基线 ==============
def run_stage1_gnn(seeds, tasks, models, gpu, output_root, n_epochs, patience):
    """多种子 Stage1 GNN 基线实验

    调用 stage1_representation_comparison/code/gnn_baseline.py 的核心函数。
    输出目录: results/stage1_gnn/seed_X/TASK/MODEL/
    """
    from stage1_representation_comparison.code.gnn_baseline import (
        run_model_on_task, DEFAULT_PARAMS, CUSTOM_MODELS, PYG_MODELS, ALL_MODELS)
    from common.tasks import TASKS
    import torch

    task_list = TASKS if tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in tasks.split(',')]
    all_models = ALL_MODELS if models == 'all' else models.split(',')

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
        print(f"\n{'#' * 60}\n# Stage1 GNN | Seed={seed} | GPU={gpu}\n{'#' * 60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            for m in all_models:
                try:
                    res = run_model_on_task(m, task, params, device, seed_dir,
                                            n_epochs, patience, seed)
                    res['seed'] = seed
                    # run_model_on_task 已通过 save_results 写入 config='base'
                    all_seed_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{m}|seed={seed}|{task['name']}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Stage 2: 环条件化消融 ==============
def run_stage2(seeds, tasks, gpu, output_root, n_epochs, patience):
    """多种子 Stage2 环条件化消融实验

    调用 stage2_ring_conditioning/code/ring_conditioning_ablation.py 的核心函数。
    自动从 Stage1 GNN 结果选最佳 backbone。
    输出目录: results/stage2_ring_conditioning/seed_X/TASK/BACKBONE_CONFIG/
    """
    from stage2_ring_conditioning.code.ring_conditioning_ablation import (
        run_one_config, CONFIGS, DEFAULT_PARAMS, select_best_backbone)
    import torch

    task_list = _parse_tasks(tasks)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = n_epochs
    params['patience'] = patience

    # 从 Stage1 GNN 结果选最佳 backbone (跨种子/任务取均值, 仅考虑 build_model 支持的 backbone)
    stage1_csv = _find_stage1_gnn_results(output_root)
    print(f"从 Stage1 结果自动选最佳 backbone: {stage1_csv}")
    backbone = select_best_backbone(stage1_csv)

    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#' * 60}\n# Stage2 Ring Conditioning | Seed={seed} | backbone={backbone} | GPU={gpu}\n{'#' * 60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            for cfg_name in CONFIGS:
                try:
                    res = run_one_config(cfg_name, backbone, task, params, seed_dir,
                                         device, n_epochs, patience, seed)
                    res['seed'] = seed
                    # run_one_config 已通过 save_results 写入 config=cfg_name
                    all_seed_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{backbone}_{cfg_name}|seed={seed}|{task['name']}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Stage 3: Ring Masking 预训练 ==============
def run_stage3(seeds, tasks, gpu, output_root, n_epochs, patience, pretrain_epochs):
    """多种子 Stage3 Ring Masking 预训练实验

    调用 stage3_mask_pretraining/code/run_pretrain_eval.py 的核心函数。
    自动从 Stage2 结果选最佳配置 (backbone + ring_flag + readout_mode)。
    输出目录: results/stage3_mask_pretraining/seed_X/TASK/TRAINING_MODE/
    """
    from stage3_mask_pretraining.code.run_pretrain_eval import (
        run_one_mode, select_best_stage2_config, DEFAULT_PARAMS,
        TRAINING_MODES, NODE_VEC_LEN, MAX_ATOMS)
    from common.tasks import canonical_splits
    from common.graph_data import load_adj_format
    import torch

    task_list = _parse_tasks(tasks)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = n_epochs
    params['patience'] = patience

    # 从 Stage2 结果为每个任务选最佳配置 (backbone + ring_flag + readout_mode)
    stage2_csv = _find_stage2_results(output_root)
    print(f"从 Stage2 结果选最佳配置: {stage2_csv}")

    # 预计算每个任务的 best_cfg + data (与 seed 无关, 避免重复 RDKit 特征化)
    task_data = {}
    for task in task_list:
        best_cfg = select_best_stage2_config(stage2_csv, task['name'])
        print(f"  [{task['name']}] Stage2 最佳配置: backbone={best_cfg['backbone']}, "
              f"ring_flag={best_cfg['ring_flag']}, readout={best_cfg['readout_mode']}")
        data = load_adj_format(task['dataset_path'], task['target_col'],
                               NODE_VEC_LEN, MAX_ATOMS,
                               ring_flag_value=best_cfg['ring_flag'], device=device)
        task_data[task['name']] = (best_cfg, data)

    all_seed_results, done_seeds = _load_completed_seeds(output_root)
    if done_seeds:
        print(f"断点续跑: 已完成种子 {sorted(done_seeds)}, 跳过")

    for seed in seeds:
        if seed in done_seeds:
            continue
        print(f"\n{'#' * 60}\n# Stage3 Mask Pretraining | Seed={seed} | GPU={gpu}\n{'#' * 60}")
        # run_one_mode 内部创建 output_root/seed_X/TASK/TRAINING_MODE/, 故传 output_root
        os.makedirs(os.path.join(output_root, f'seed_{seed}'), exist_ok=True)

        for task in task_list:
            best_cfg, data = task_data[task['name']]
            # 规范划分 (与各阶段一致, seed 控制划分)
            # parent-molecule-level split: 同一分子的多个环级样本进入同一 split
            splits = canonical_splits(data['n'], seed=seed, groups=data['smiles'])
            test_t = torch.tensor(splits[0], dtype=torch.long, device=device)
            ftrain_t = torch.tensor(splits[2], dtype=torch.long, device=device)
            fval_t = torch.tensor(splits[3], dtype=torch.long, device=device)
            cv_t = [(torch.tensor(tr, dtype=torch.long, device=device),
                     torch.tensor(va, dtype=torch.long, device=device))
                    for tr, va in splits[1]]
            splits_t = (test_t, cv_t, ftrain_t, fval_t)

            for mode in TRAINING_MODES:
                try:
                    res = run_one_mode(task, mode, best_cfg, params, data, splits_t,
                                       device, output_root, seed, pretrain_epochs)
                    res['seed'] = seed
                    # run_one_mode 已通过 save_results 写入 config=training_mode
                    all_seed_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{mode}|seed={seed}|{task['name']}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== Stage 4: 跨架构验证 ==============
def run_stage4(seeds, tasks, backbones, gpu, output_root, n_epochs, patience):
    """多种子 Stage4 跨架构验证实验

    调用 stage4_cross_architecture/code/cross_arch_eval.py 的核心函数。
    输出目录: results/stage4_cross_arch/seed_X/TASK/BACKBONE_CONFIG/
    """
    from stage4_cross_architecture.code.cross_arch_eval import (
        run_custom_backbone, run_pyg_backbone, CONFIGS, DEFAULT_PARAMS,
        BACKBONES_CUSTOM, BACKBONES_PYG)
    from common.train_eval import save_results
    import torch

    task_list = _parse_tasks(tasks)
    valid_backbones = BACKBONES_CUSTOM + BACKBONES_PYG
    if backbones == 'all':
        backbone_list = list(valid_backbones)
    else:
        backbone_list = [b.strip() for b in backbones.split(',') if b.strip()]

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
        print(f"\n{'#' * 60}\n# Stage4 Cross-Arch | Seed={seed} | GPU={gpu}\n{'#' * 60}")
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for task in task_list:
            task_name = task['name']
            for backbone in backbone_list:
                if backbone in BACKBONES_PYG:
                    runner = run_pyg_backbone
                    bb_type = 'pyg'
                elif backbone in BACKBONES_CUSTOM:
                    runner = run_custom_backbone
                    bb_type = 'custom'
                else:
                    print(f"  [警告] 未知 backbone: {backbone} (可选: {valid_backbones}), 跳过")
                    continue
                for config_name in CONFIGS:
                    out_dir = os.path.join(seed_dir, task_name, f'{backbone}_{config_name}')
                    os.makedirs(out_dir, exist_ok=True)
                    try:
                        res = runner(backbone, config_name, task, params, seed,
                                     n_epochs, patience, device)
                        summary = save_results(
                            out_dir=out_dir, model_name=backbone, config_name=config_name,
                            task_name=task_name, n_total=res['n_total'],
                            cv_rows=res['cv_rows'], train_metrics=res['train_metrics'],
                            test_metrics=res['test_metrics'], te_pred=res['te_pred'],
                            te_true=res['te_true'], train_time_sec=res['train_time_sec'],
                            extra={'backbone_type': bb_type,
                                   'ring_flag': CONFIGS[config_name]['ring_flag'],
                                   'readout_mode': CONFIGS[config_name]['readout'],
                                   'seed': seed})
                        summary['seed'] = seed
                        all_seed_results.append(summary)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  [{backbone}_{config_name}|seed={seed}|{task_name}] 失败: {e}", flush=True)

        _save_incremental(output_root, all_seed_results, completed_seed=seed)
        print(f"  [Seed={seed}] 完成, 已增量保存", flush=True)

    return pd.DataFrame(all_seed_results)


# ============== 聚合 ==============
def aggregate_results(output_root, stage_name):
    """聚合多种子结果: mean ± std

    分组键统一为 ['task', 'model', 'config'] (使用 'config' 列)。
    输出 all_{stage_name}_summary.csv
    """
    per_seed_csv = os.path.join(output_root, 'per_seed_results.csv')
    if not os.path.exists(per_seed_csv):
        print(f"未找到 {per_seed_csv}, 请先运行实验或使用 --recover 恢复")
        return None

    df = pd.read_csv(per_seed_csv)
    if df.empty:
        print(f"{per_seed_csv} 为空, 跳过聚合")
        return None

    print(f"\n{'=' * 60}\n{stage_name} 多种子聚合 (seeds={sorted(df['seed'].unique())})\n{'=' * 60}")

    metric_cols = [c for c in METRIC_COLS_FOR_AGG if c in df.columns]
    group_keys = ['task', 'model', 'config']

    agg = df.groupby(group_keys)[metric_cols].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]

    summary_csv = os.path.join(output_root, f'all_{stage_name}_summary.csv')
    agg.to_csv(summary_csv, index=False)
    print(f"聚合结果保存: {summary_csv}")

    display_cols = [c for c in ['task', 'model', 'config',
                                'test_r2_mean', 'test_r2_std',
                                'test_mae_mean', 'test_rmse_mean', 'train_time_sec_mean']
                    if c in agg.columns]
    print(agg[display_cols].to_string(index=False))
    return agg


def recover_from_summaries(output_root, stage_name):
    """从各 seed 的 summary.csv 重建 per_seed_results.csv

    所有阶段的 summary.csv 均位于 seed_X/TASK/SUBDIR/summary.csv (2 层子目录):
      - Stage1 ML:   seed_X/TASK/MODEL/summary.csv
      - Stage1 GNN:  seed_X/TASK/MODEL/summary.csv
      - Stage2:      seed_X/TASK/BACKBONE_CONFIG/summary.csv
      - Stage3:      seed_X/TASK/TRAINING_MODE/summary.csv
      - Stage4:      seed_X/TASK/BACKBONE_CONFIG/summary.csv

    model / config 优先从 summary.csv 读取 (save_results 写入); 缺失时:
      - config 默认 'fingerprint' (仅 Stage1 ML 的 summary.csv 无 config 行)
      - model 取目录名
    cv_r2_mean -> cv_r2 列名映射 (M2 修复)。
    """
    pattern = os.path.join(output_root, 'seed_*', '*', '*', 'summary.csv')
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
            subdir = parts[2]  # MODEL 或 BACKBONE_CONFIG 或 TRAINING_MODE

            df = pd.read_csv(f, header=None, names=['metric', 'value'])
            d = dict(zip(df['metric'], df['value']))
            res = {'seed': seed, 'task': task}

            # model / config 优先从 summary.csv 读取
            res['model'] = str(d.get('model', subdir))
            res['config'] = str(d.get('config', 'fingerprint'))

            # M2: cv_r2_mean -> cv_r2 映射
            if 'cv_r2_mean' in d:
                res['cv_r2'] = float(d['cv_r2_mean'])
            elif 'cv_r2' in d:
                res['cv_r2'] = float(d['cv_r2'])

            for k in ['n', 'cv_r2_std', 'cv_mae', 'cv_rmse',
                      'train_r2', 'train_mae', 'train_rmse',
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
    for col in PER_SEED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in PER_SEED_COLUMNS if c in df.columns]]
    csv_path = os.path.join(output_root, 'per_seed_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"恢复 {len(df)} 条结果 -> {csv_path}")
    return df


# ============== 主入口 ==============
def main():
    parser = argparse.ArgumentParser(description='多种子 × 5折CV 实验 (四阶段版)')
    parser.add_argument('--stage', type=str, choices=['1ml', '1gnn', '2', '3', '4'],
                        default=None, help='运行阶段 (1ml/1gnn/2/3/4)')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)),
                        help='逗号分隔的种子列表')
    parser.add_argument('--tasks', type=str, default='all', help='逗号分隔任务名, 或 all')
    parser.add_argument('--models', type=str, default='all',
                        help='Stage1 GNN 模型列表, 逗号分隔, 或 all')
    parser.add_argument('--backbones', type=str, default='MPNN,DMPNN,GIN,GAT',
                        help='Stage4 backbone 列表, 逗号分隔, 或 all')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--n_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Stage3 预训练轮数')
    parser.add_argument('--output_root', type=str, default=None,
                        help='输出目录 (默认各阶段 results/stageX_*/)')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='仅聚合所有阶段结果')
    parser.add_argument('--recover', action='store_true',
                        help='从 summary.csv 重建 per_seed_results.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    print(f"种子: {seeds}")

    # ----- 恢复模式 -----
    if args.recover:
        if args.stage:
            dirs = [args.output_root or _default_output_root(args.stage)]
            stage_names = [STAGE_DIRS[args.stage]]
        else:
            dirs = [_default_output_root(s) for s in STAGE_DIRS]
            stage_names = [STAGE_DIRS[s] for s in STAGE_DIRS]
        for d, name in zip(dirs, stage_names):
            if os.path.isdir(d):
                print(f"\n恢复 {name}...")
                recover_from_summaries(d, name)
        return

    # ----- 仅聚合模式 -----
    if args.aggregate_only:
        for s in STAGE_DIRS:
            out_root = _default_output_root(s)
            name = STAGE_DIRS[s]
            csv_path = os.path.join(out_root, 'per_seed_results.csv')
            if os.path.exists(csv_path):
                aggregate_results(out_root, name)
            else:
                print(f"\n{name}: 无 per_seed_results.csv, 尝试从 summary.csv 恢复...")
                df = recover_from_summaries(out_root, name)
                if df is not None and len(df) > 0:
                    aggregate_results(out_root, name)
        return

    # ----- 单阶段运行 -----
    if args.stage is None:
        print("请指定 --stage (1ml/1gnn/2/3/4) 或 --aggregate_only / --recover")
        return

    output_root = args.output_root or _default_output_root(args.stage)
    os.makedirs(output_root, exist_ok=True)
    stage_name = STAGE_DIRS[args.stage]

    t0 = time.time()
    if args.stage == '1ml':
        run_stage1_ml(seeds, args.tasks, args.gpu, output_root)
    elif args.stage == '1gnn':
        run_stage1_gnn(seeds, args.tasks, args.models, args.gpu, output_root,
                       args.n_epochs, args.patience)
    elif args.stage == '2':
        run_stage2(seeds, args.tasks, args.gpu, output_root,
                   args.n_epochs, args.patience)
    elif args.stage == '3':
        run_stage3(seeds, args.tasks, args.gpu, output_root,
                   args.n_epochs, args.patience, args.pretrain_epochs)
    elif args.stage == '4':
        run_stage4(seeds, args.tasks, args.backbones, args.gpu, output_root,
                   args.n_epochs, args.patience)

    elapsed = time.time() - t0
    print(f"\n{stage_name} 完成, 耗时 {elapsed / 60:.1f} 分钟")

    aggregate_results(output_root, stage_name)


if __name__ == '__main__':
    main()
