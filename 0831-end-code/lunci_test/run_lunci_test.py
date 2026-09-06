#!/usr/bin/env python3
"""
集外测试脚本: 在 lunci6 / lunci78 测试集上评估各阶段最优模型

对每个任务 (HOMA, NICS_1zz, MBCO), 从各阶段 per_seed_results.csv 中按 test_r2 均值
(跨种子) 选出最优配置, 用 5 个种子独立训练:
  1. canonical_splits(n, seed, groups=smiles) 生成 final_train / final_val 划分
     (parent-molecule-level split, 同一分子的多个环级样本进入同一 split)
  2. 在 final_train 上训练模型, 用 final_val 做 early stopping
  3. 加载 lunci6 / lunci78 测试数据 (用 LUNCI_COL_MAP 重命名列为小写后保存为临时 CSV)
  4. 在 lunci 数据上评估 (R2, MAE, RMSE)
  5. 每个 seed 保存预测值, 最终报告 mean ± std (ddof=1)

模型类型处理:
  - 传统 ML:        build_fingerprint_matrix 提取特征 + StandardScaler 标准化
  - GNN (自定义):   load_adj_format + build_model + train_custom_model / eval_custom_model
  - GNN (PyG/DMPNN): load_pyg_format + build_pyg_model + train_pyg_model / eval_pyg_model
  - Stage3 预训练:  若 training_mode != direct_supervised, 先在 final_train 上掩码预训练
                    (MaskedAutoencoder + pretrain), 再加载权重微调

输出目录结构:
  results/lunci_test/TEST_SET/MODEL_ID/
    predictions_seed_X.csv   每个 seed 的 (true, pred)
    test_predictions.csv      所有 seed 合并 (true, pred, seed)
    per_seed_metrics.csv      每个 seed 的 R2/MAE/RMSE
    summary.csv               mean ± std 汇总 (key-value 格式)
  results/lunci_test/lunci_summary.csv   所有组合的最终汇总

用法:
  python lunci_test/run_lunci_test.py --gpu 0
  python lunci_test/run_lunci_test.py --gpu 0 --tasks HOMA --test_sets lunci6 --stages 2,3
"""
import os
import sys
import csv
import time
import tempfile
import argparse
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# 将 last_end_code 根目录加入 sys.path, 以便 import common / models / stageX_*
LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.tasks import (
    TASKS, canonical_splits, compute_metrics, clean_dataset_csv,
    LUNCI_COL_MAP, EXTERNAL_TEST_FILES, get_task,
)
from common.graph_data import load_adj_format, load_pyg_format
from common.features import build_fingerprint_matrix
from common.train_eval import (
    set_full_seed, train_custom_model, eval_custom_model,
    train_pyg_model, eval_pyg_model,
)
from models.ring_conditioned_gnn import build_model
from models.pyg_models import build_pyg_model
from stage3_mask_pretraining.code.mask_pretrain import MaskedAutoencoder, pretrain
from stage3_mask_pretraining.code.run_pretrain_eval import select_best_stage2_config
from stage1_representation_comparison.code.traditional_ml import build_models


# ============== 默认训练参数 (与各阶段一致) ==============
DEFAULT_PARAMS = {
    'hidden_dim': 128,
    'n_conv_layers': 3,
    'n_hidden_layers': 2,
    'learning_rate': 0.001,
    'p_dropout': 0.2,
    'batch_size': 64,
    'weight_decay': 1e-5,
    'n_epochs': 200,
    'patience': 30,
}
NODE_VEC_LEN = 60
MAX_ATOMS = 85  # 增大以容纳 lunci78 中含氢后达 77 原子的分子
SEEDS = [42, 123, 456, 789, 2024]
PRETRAIN_EPOCHS = 100

# Stage2 4 组消融配置 (config 名 -> ring_flag / readout / 展示名)
STAGE2_CONFIGS = {
    'base':              {'ring_flag': 0,  'readout': 'fixed_avg', 'display': 'Base'},
    'membership':        {'ring_flag': 10, 'readout': 'fixed_avg', 'display': 'Membership'},
    'learnable_readout': {'ring_flag': 0,  'readout': 'attention', 'display': 'LearnableReadout'},
    'joint':             {'ring_flag': 10, 'readout': 'attention', 'display': 'Joint'},
}
# Stage4 2 组配置
STAGE4_CONFIGS = {
    'base':             {'ring_flag': 0,  'readout': 'fixed_avg',  'display': 'base'},
    'ring_conditioned': {'ring_flag': 10, 'readout': 'attention',  'display': 'ring_conditioned'},
}
# 使用 PyG 模型的 backbone
PYG_BACKBONES = ['DMPNN']

# 各阶段结果 CSV 候选路径 (按优先级, 取第一个存在)
RESULTS_ROOT = os.path.join(LAST_END_ROOT, 'results')
STAGE_CSV_PATHS = {
    '1ml':  [os.path.join(RESULTS_ROOT, 'stage1_traditional_ml', 'per_seed_results.csv')],
    '1gnn': [os.path.join(RESULTS_ROOT, 'stage1_gnn', 'per_seed_results.csv')],
    '2':    [os.path.join(RESULTS_ROOT, 'stage2_ring_conditioning', 'per_seed_results.csv')],
    '3':    [os.path.join(RESULTS_ROOT, 'stage3_mask_pretraining', 'per_seed_results.csv')],
    '4':    [os.path.join(RESULTS_ROOT, 'stage4_cross_arch', 'per_seed_results.csv'),
             os.path.join(RESULTS_ROOT, 'stage4_g3', 'per_seed_results.csv'),
             os.path.join(RESULTS_ROOT, 'stage4_g0_parallel', 'per_seed_results.csv')],
}

# lunci 临时 CSV 缓存 (避免重复重命名)
_LUNCI_CSV_CACHE = {}


# ============== 工具函数 ==============
def find_stage_csv(stage):
    """按候选路径查找阶段结果 CSV, 返回第一个存在的路径, 否则 None"""
    for p in STAGE_CSV_PATHS.get(stage, []):
        if os.path.exists(p):
            return p
    return None


def _agg_test_r2(df, task_name, group_keys):
    """按 group_keys 分组, 取 test_r2 跨种子均值, 返回最优组的 key dict (含 test_r2_mean)

    Returns:
        dict 或 None (无数据时). 例如 group_keys=['model','config'] 时返回
        {'model': 'MPNN', 'config': 'joint', 'test_r2_mean': 0.985}
    """
    sub = df[df['task'] == task_name].copy()
    if len(sub) == 0:
        return None
    if 'test_r2' not in sub.columns:
        return None
    agg = sub.groupby(group_keys)['test_r2'].mean().sort_values(ascending=False)
    best_keys = agg.index[0]
    result = dict(zip(group_keys, best_keys if isinstance(best_keys, tuple) else (best_keys,)))
    result['test_r2_mean'] = float(agg.iloc[0])
    return result


def select_best(stage, task_name):
    """从指定阶段结果 CSV 选最优配置

    Returns:
        dict 描述最优配置, 包含 'kind' 字段标识模型类型:
          - 'ml'          传统ML
          - 'gnn_custom'  自定义 GNN (adj 格式)
          - 'gnn_pyg'     PyG GNN (DMPNN)
          - 'stage3'      Stage3 预训练 (基于自定义 GNN)
        若对应阶段 CSV 不存在, 返回 None
    """
    csv_path = find_stage_csv(stage)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)

    if stage == '1ml':
        best = _agg_test_r2(df, task_name, ['model'])
        if best is None:
            return None
        return {'kind': 'ml', 'model': str(best['model']),
                'test_r2_mean': best['test_r2_mean']}

    if stage == '1gnn':
        best = _agg_test_r2(df, task_name, ['model'])
        if best is None:
            return None
        backbone = str(best['model'])
        is_pyg = backbone in PYG_BACKBONES
        return {'kind': 'gnn_pyg' if is_pyg else 'gnn_custom',
                'backbone': backbone,
                'ring_flag': 0, 'readout': 'fixed_avg',  # Stage1 Base
                'test_r2_mean': best['test_r2_mean']}

    if stage == '2':
        best = _agg_test_r2(df, task_name, ['model', 'config'])
        if best is None:
            return None
        backbone, config = str(best['model']), str(best['config'])
        cfg = STAGE2_CONFIGS.get(config)
        if cfg is None:
            # 未知 config 名, 回退到 base 设置
            cfg = {'ring_flag': 0, 'readout': 'fixed_avg', 'display': config}
        return {'kind': 'gnn_custom', 'backbone': backbone,
                'ring_flag': cfg['ring_flag'], 'readout': cfg['readout'],
                'config': config, 'display': cfg['display'],
                'test_r2_mean': best['test_r2_mean']}

    if stage == '3':
        # Stage3 固定使用 Stage2 最佳配置 (backbone + ring_flag + readout),
        # 这里先从 Stage2 结果获取该配置, 再从 Stage3 CSV 选最优 training_mode
        stage2_csv = find_stage_csv('2')
        if stage2_csv is None:
            return None
        best_cfg = select_best_stage2_config(stage2_csv, task_name)
        best = _agg_test_r2(df, task_name, ['model', 'config'])
        if best is None:
            return None
        return {'kind': 'stage3',
                'backbone': str(best['model']),
                'training_mode': str(best['config']),
                'ring_flag': int(best_cfg['ring_flag']),
                'readout': str(best_cfg['readout_mode']),
                'test_r2_mean': best['test_r2_mean']}

    if stage == '4':
        best = _agg_test_r2(df, task_name, ['model', 'config'])
        if best is None:
            return None
        backbone, config = str(best['model']), str(best['config'])
        cfg = STAGE4_CONFIGS.get(config)
        if cfg is None:
            cfg = {'ring_flag': 0, 'readout': 'fixed_avg', 'display': config}
        is_pyg = backbone in PYG_BACKBONES
        return {'kind': 'gnn_pyg' if is_pyg else 'gnn_custom',
                'backbone': backbone,
                'ring_flag': cfg['ring_flag'], 'readout': cfg['readout'],
                'config': config, 'display': cfg['display'],
                'test_r2_mean': best['test_r2_mean']}

    return None


def build_model_id(stage, best):
    """根据阶段和最优配置生成 MODEL_ID (用作输出目录名)"""
    if stage == '1ml':
        return f"stage1ml_{best['model']}"
    if stage == '1gnn':
        return f"stage1gnn_{best['backbone']}"
    if stage == '2':
        return f"stage2_{best['display']}"
    if stage == '3':
        return f"stage3_{best['training_mode']}"
    if stage == '4':
        return f"stage4_{best['backbone']}_{best['display']}"
    return f"stage_{stage}"


def prepare_lunci_csv(test_set, target_col):
    """准备 lunci 测试集 CSV: 用 LUNCI_COL_MAP 重命名列为小写, 保存为临时文件

    lunci 原始 CSV 列名: New_ID,SMILES,Ring_ID,Ring_Size,Ring_Atoms,HOMA,MBCO,NICS_ZZ
    重命名后:           smiles, atom_on_ring, homa_value, NICS_value, mbco_value
                        (Ring_ID / Ring_Size 保持大写, 与训练 CSV 一致)

    Returns:
        临时 CSV 路径 (同 test_set/target_col 复用, 缓存)
    """
    key = (test_set, target_col)
    if key in _LUNCI_CSV_CACHE:
        return _LUNCI_CSV_CACHE[key]
    src = EXTERNAL_TEST_FILES[test_set]
    df = pd.read_csv(src)
    # 仅重命名存在的列
    rename_map = {k: v for k, v in LUNCI_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    # 剔除目标列为 NaN 的行 (与传统ML clean_dataset_csv 一致)
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        if before != len(df):
            print(f"  [prepare_lunci_csv] {test_set}/{target_col}: 剔除 {before - len(df)} 行 NaN")
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"lunci_{test_set}_{target_col}.csv")
    df.to_csv(tmp_path, index=False)
    _LUNCI_CSV_CACHE[key] = tmp_path
    return tmp_path


# ============== 训练 + 评估函数 (每种模型类型一个) ==============
def run_traditional_ml(task, model_name, seed, params, lunci_csv, device, gpu):
    """训练传统 ML 模型 (final_train) 并在 lunci 上评估

    流程与 stage1 traditional_ml.py 的 final 模型一致:
      - build_fingerprint_matrix 提取特征 (训练 + lunci)
      - StandardScaler 在 final_train 上 fit, transform final_val / lunci
      - 用 build_models 构造指定模型 (seed 控制随机性), fit final_train
    """
    target_col = task['target_col']
    # 加载训练数据
    clean_path = clean_dataset_csv(task['dataset_path'], target_col)
    df = pd.read_csv(clean_path)
    smiles_list = df['smiles'].tolist()
    y = df[target_col].astype(float).to_numpy()
    n_total = len(smiles_list)

    # parent-molecule-level split (仅用 final_train / final_val)
    _, _, final_train_idx, final_val_idx = canonical_splits(
        n_total, seed=seed, groups=smiles_list)

    # 特征矩阵 (训练数据)
    X, valid = build_fingerprint_matrix(smiles_list, df=df)
    assert valid.all(), "训练数据存在无效 SMILES"

    # lunci 特征
    df_lunci = pd.read_csv(lunci_csv)
    smiles_lunci = df_lunci['smiles'].tolist()
    X_lunci, valid_lu = build_fingerprint_matrix(smiles_lunci, df=df_lunci)
    assert valid_lu.all(), "lunci 数据存在无效 SMILES"
    y_lunci = df_lunci[target_col].astype(float).to_numpy()

    # 标准化 (在 final_train 上 fit)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[final_train_idx])
    X_lu = scaler.transform(X_lunci)

    # 构造并训练模型 (与 traditional_ml.py final 模型一致: 用 base_seed, fold=0)
    set_full_seed(seed)
    models = build_models(seed=seed, gpu=gpu)
    if model_name not in models:
        raise ValueError(f"模型 {model_name} 不在 build_models 输出中 "
                         f"(可用: {list(models.keys())}), 可能缺少依赖库")
    model = models[model_name]
    model.fit(X_tr, y[final_train_idx])

    # 在 lunci 上评估
    pred = model.predict(X_lu)
    metrics = compute_metrics(y_lunci, pred)
    return metrics, pred, y_lunci


def run_custom_gnn(task, backbone, ring_flag, readout_mode, seed, params,
                   lunci_csv, device, training_mode=None, pretrain_epochs=PRETRAIN_EPOCHS):
    """训练自定义 GNN (adj 格式) 并在 lunci 上评估

    流程:
      - load_adj_format 加载训练数据 (ring_flag_value 控制环标记注入)
      - canonical_splits 生成 final_train / final_val
      - (Stage3) 若 training_mode != direct_supervised, 先在 final_train 上掩码预训练
      - build_model 构造模型 (可选加载预训练权重), train_custom_model 训练
      - load_adj_format 加载 lunci 数据 (相同 ring_flag_value), eval_custom_model 评估
    """
    target_col = task['target_col']
    # 训练数据 (adj 格式, 直接放到 device)
    data = load_adj_format(
        task['dataset_path'], target_col,
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device=device)
    n = data['n']
    nvl = data['node_vec_len']

    _, _, final_train_idx, final_val_idx = canonical_splits(
        n, seed=seed, groups=data['smiles'])
    ftrain_t = torch.tensor(final_train_idx, dtype=torch.long, device=device)
    fval_t = torch.tensor(final_val_idx, dtype=torch.long, device=device)

    # Stage3 掩码预训练 (仅在 final_train 分子内, 防止信息泄漏)
    pretrained_state = None
    if training_mode is not None and training_mode != 'direct_supervised':
        mask_type = 'random' if training_mode == 'random_mask_pretrain' else 'ring'
        print(f"    [Stage3 预训练] mask={mask_type}, epochs={pretrain_epochs}, "
              f"samples={len(final_train_idx)}", flush=True)
        mae_model = MaskedAutoencoder(
            backbone=backbone, node_vec_len=nvl, hidden_dim=params['hidden_dim'],
            n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
            p_dropout=params['p_dropout'], readout_mode=readout_mode,
        ).to(device)
        mae_model = pretrain(
            mae_model, data, ftrain_t, device, mask_type=mask_type,
            n_epochs=pretrain_epochs, batch_size=params['batch_size'],
            lr=params['learning_rate'], weight_decay=params['weight_decay'], seed=seed)
        pretrained_state = {k: v.clone() for k, v in mae_model.encoder.state_dict().items()}
        del mae_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 构造并训练模型
    set_full_seed(seed)
    model = build_model(
        backbone, node_vec_len=NODE_VEC_LEN,
        hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'], p_dropout=params['p_dropout'],
        readout_mode=readout_mode,
    ).to(device)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state, strict=False)
    model = train_custom_model(
        model, params, data, ftrain_t, fval_t, device,
        n_epochs=params['n_epochs'], patience=params['patience'], seed=seed)

    # lunci 数据评估 (相同 ring_flag_value)
    lunci_data = load_adj_format(
        lunci_csv, target_col,
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device=device)
    n_lu = lunci_data['n']
    all_idx = torch.arange(n_lu, dtype=torch.long, device=device)
    metrics, pred, true = eval_custom_model(
        model, lunci_data, all_idx, params['batch_size'], device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, pred, true


def run_pyg_gnn(task, backbone, ring_flag, readout_mode, seed, params, lunci_csv, device):
    """训练 PyG GNN (DMPNN) 并在 lunci 上评估

    流程:
      - load_pyg_format 加载训练数据 (CPU 上, batch 时 to(device))
      - canonical_splits 生成 final_train / final_val
      - build_pyg_model 构造模型, train_pyg_model 训练
      - load_pyg_format 加载 lunci 数据, eval_pyg_model 评估
    """
    target_col = task['target_col']
    data_list, _ = load_pyg_format(
        task['dataset_path'], target_col,
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device='cpu')
    n = len(data_list)
    in_channels = data_list[0].x.size(-1)

    _, _, final_train_idx, final_val_idx = canonical_splits(
        n, seed=seed, groups=[d.smiles for d in data_list])

    set_full_seed(seed)
    model = build_pyg_model(
        backbone, in_channels=in_channels,
        hidden_dim=params['hidden_dim'], n_layers=params['n_conv_layers'],
        dropout=params['p_dropout'], edge_dim=4, readout_mode=readout_mode,
    ).to(device)
    model = train_pyg_model(
        model, params, data_list,
        np.asarray(final_train_idx), np.asarray(final_val_idx), device,
        n_epochs=params['n_epochs'], patience=params['patience'], seed=seed)

    # lunci 数据评估
    lunci_list, _ = load_pyg_format(
        lunci_csv, target_col,
        node_vec_len=NODE_VEC_LEN, max_atoms=MAX_ATOMS,
        ring_flag_value=ring_flag, device='cpu')
    n_lu = len(lunci_list)
    all_idx = np.arange(n_lu)
    metrics, pred, true = eval_pyg_model(
        model, lunci_list, all_idx, params['batch_size'], device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, pred, true


def dispatch_run(stage, best, task, seed, params, lunci_csv, device, gpu, pretrain_epochs):
    """根据阶段/配置分发到对应的训练评估函数

    Returns:
        (metrics, pred, true)  metrics=(r2, mae, rmse)
    """
    kind = best['kind']
    if stage == '1ml':
        return run_traditional_ml(
            task, best['model'], seed, params, lunci_csv, device, gpu)
    if kind == 'stage3':
        return run_custom_gnn(
            task, best['backbone'], best['ring_flag'], best['readout'],
            seed, params, lunci_csv, device,
            training_mode=best['training_mode'], pretrain_epochs=pretrain_epochs)
    if kind == 'gnn_custom':
        return run_custom_gnn(
            task, best['backbone'], best['ring_flag'], best['readout'],
            seed, params, lunci_csv, device, training_mode=None)
    if kind == 'gnn_pyg':
        return run_pyg_gnn(
            task, best['backbone'], best['ring_flag'], best['readout'],
            seed, params, lunci_csv, device)
    raise ValueError(f"未知 kind: {kind}")


# ============== 结果保存 ==============
def save_combo_results(out_dir, task_name, stage, model_id, test_set,
                       seed_metrics, preds_list):
    """保存单个组合 (TEST_SET/MODEL_ID) 的结果

    seed_metrics: list of dict {seed, r2, mae, rmse, time}
    preds_list:   list of (seed, true_array, pred_array)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 每个 seed 的预测
    for seed, true, pred in preds_list:
        pd.DataFrame({'true': true, 'pred': pred}).to_csv(
            os.path.join(out_dir, f'predictions_seed_{seed}.csv'), index=False)

    # 合并所有 seed 的预测 (任务要求 test_predictions.csv)
    if preds_list:
        all_rows = []
        for seed, true, pred in preds_list:
            for t, p in zip(true, pred):
                all_rows.append({'seed': seed, 'true': float(t), 'pred': float(p)})
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, 'test_predictions.csv'), index=False)

    # per-seed metrics
    df_sm = pd.DataFrame(seed_metrics)
    df_sm.to_csv(os.path.join(out_dir, 'per_seed_metrics.csv'), index=False)

    # mean ± std (ddof=1)
    def _ms(arr):
        arr = np.asarray(arr, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return m, s

    r2_m, r2_s = _ms(df_sm['r2'].values)
    mae_m, mae_s = _ms(df_sm['mae'].values)
    rmse_m, rmse_s = _ms(df_sm['rmse'].values)

    # summary.csv (key-value 格式, 与现有 save_results 风格一致)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['task', task_name])
        w.writerow(['stage', stage])
        w.writerow(['model_id', model_id])
        w.writerow(['test_set', test_set])
        w.writerow(['n_seeds', len(seed_metrics)])
        w.writerow(['r2_mean', r2_m])
        w.writerow(['r2_std', r2_s])
        w.writerow(['mae_mean', mae_m])
        w.writerow(['mae_std', mae_s])
        w.writerow(['rmse_mean', rmse_m])
        w.writerow(['rmse_std', rmse_s])

    return {
        'task': task_name, 'stage': stage, 'model_id': model_id,
        'test_set': test_set, 'n_seeds': len(seed_metrics),
        'r2_mean': r2_m, 'r2_std': r2_s,
        'mae_mean': mae_m, 'mae_std': mae_s,
        'rmse_mean': rmse_m, 'rmse_std': rmse_s,
    }


# ============== 主入口 ==============
def main():
    parser = argparse.ArgumentParser(
        description='集外测试: 在 lunci6/lunci78 上评估各阶段最优模型')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--tasks', type=str, default='all',
                        help='逗号分隔任务名 (HOMA,NICS_1zz,MBCO) 或 all')
    parser.add_argument('--test_sets', type=str, default='all',
                        help='逗号分隔测试集名 (lunci6,lunci78) 或 all')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)),
                        help='逗号分隔的种子列表')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'],
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'],
                        help='early stopping patience')
    parser.add_argument('--stages', type=str, default='all',
                        help='逗号分隔阶段列表 (1ml,1gnn,2,3,4) 或 all')
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS,
                        help='Stage3 掩码预训练轮数')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(RESULTS_ROOT, 'lunci_test'),
                        help='输出根目录 (默认 results/lunci_test)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"集外测试 | device={device}")
    print(f"output_dir={args.output_dir}")

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    task_names = ([t['name'] for t in TASKS] if args.tasks == 'all'
                  else [s.strip() for s in args.tasks.split(',') if s.strip()])
    test_sets = (['lunci6', 'lunci78'] if args.test_sets == 'all'
                 else [s.strip() for s in args.test_sets.split(',') if s.strip()])
    stages = (['1ml', '1gnn', '2', '3', '4'] if args.stages == 'all'
              else [s.strip() for s in args.stages.split(',') if s.strip()])

    params = dict(DEFAULT_PARAMS)
    params['n_epochs'] = args.n_epochs
    params['patience'] = args.patience

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = []

    for task_name in task_names:
        task = get_task(task_name)
        target_col = task['target_col']
        print(f"\n{'#' * 70}\n# 任务: {task_name} (target={target_col})\n{'#' * 70}")

        for stage in stages:
            best = select_best(stage, task_name)
            if best is None:
                print(f"\n[跳过] {task_name} / stage={stage}: 未找到阶段结果 CSV "
                      f"(候选: {STAGE_CSV_PATHS.get(stage, [])})")
                continue
            model_id = build_model_id(stage, best)
            print(f"\n=== stage={stage} | model_id={model_id} | best={best} ===")

            for test_set in test_sets:
                lunci_csv = prepare_lunci_csv(test_set, target_col)
                out_dir = os.path.join(args.output_dir, test_set, model_id)
                os.makedirs(out_dir, exist_ok=True)

                seed_metrics = []
                preds_list = []
                for seed in seeds:
                    print(f"  [{test_set}/{model_id}] seed={seed} 训练中...", flush=True)
                    t0 = time.time()
                    try:
                        metrics, pred, true = dispatch_run(
                            stage, best, task, seed, params,
                            lunci_csv, device, args.gpu, args.pretrain_epochs)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"    [失败] seed={seed}: {e}", flush=True)
                        continue
                    elapsed = time.time() - t0
                    r2, mae, rmse = metrics
                    seed_metrics.append({
                        'seed': seed, 'r2': r2, 'mae': mae, 'rmse': rmse,
                        'time': elapsed,
                    })
                    preds_list.append((seed, true, pred))
                    print(f"    seed={seed}: R2={r2:.4f} MAE={mae:.4f} "
                          f"RMSE={rmse:.4f} ({elapsed:.0f}s)", flush=True)

                if not seed_metrics:
                    print(f"  [{test_set}/{model_id}] 所有 seed 均失败, 跳过汇总",
                          flush=True)
                    continue

                summary = save_combo_results(
                    out_dir, task_name, stage, model_id, test_set,
                    seed_metrics, preds_list)
                all_summaries.append(summary)
                print(f"  [{test_set}/{model_id}] 汇总: "
                      f"R2={summary['r2_mean']:.4f}±{summary['r2_std']:.4f} | "
                      f"MAE={summary['mae_mean']:.4f}±{summary['mae_std']:.4f} | "
                      f"RMSE={summary['rmse_mean']:.4f}±{summary['rmse_std']:.4f}",
                      flush=True)

    # 最终汇总 CSV
    if all_summaries:
        df_all = pd.DataFrame(all_summaries)
        # 排列顺序: task, stage, test_set, model_id
        df_all = df_all.sort_values(
            by=['task', 'stage', 'test_set', 'model_id']).reset_index(drop=True)
        agg_path = os.path.join(args.output_dir, 'lunci_summary.csv')
        df_all.to_csv(agg_path, index=False)
        print(f"\n{'=' * 70}\n最终汇总 (mean±std over seeds, ddof=1)\n{'=' * 70}")
        print(df_all.to_string(index=False))
        print(f"\n汇总保存至: {agg_path}")
    else:
        print("\n无任何结果被保存 (请检查阶段结果 CSV 是否存在 / 任务-阶段组合是否有效)")

    print("\n集外测试完成。")


if __name__ == '__main__':
    main()
