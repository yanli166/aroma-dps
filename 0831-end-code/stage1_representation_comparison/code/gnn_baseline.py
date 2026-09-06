"""
Stage 1: 可学习分子图表示 (GNN) 基线 — [0831 重构]

对比固定分子表示 (传统ML) 与可学习分子图表示 (GNN) 在环级芳香性预测中的性能。

Stage1 Base 配置:
  - ring_flag_value=0: 消息传递阶段不引入环条件化, 不在输入特征中标记目标环
  - readout_mode='fixed_avg': 输出阶段固定目标环原子平均聚合 (使用 ring_indices)
  - dry-run backbone: MPNN (时间紧可只跑 1 backbone, PROTOCOL_SPEC)

协议要点:
  - 划分: common.protocol.get_final_splits (split_seed=2026), model_seed 不进 split。
  - 分组: make_group_ids(smiles) (canonical), 启动 assert_no_leak (dev/test, folds, final-train/val)。
  - 模型选择只看 cv_mae; final test 绝不用于选择。
  - 输出: cv_results.csv / final_test_results.csv / per_seed_results.csv (含 run_status)。
  - Reduced epochs: n_epochs≈30, patience≈8。
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, get_task
from common.graph_data import load_adj_format, load_pyg_format
from common.train_eval import (
    set_full_seed, train_custom_model, eval_custom_model,
    train_pyg_model, eval_pyg_model, save_results,
)
from common.protocol import get_final_splits, make_group_ids, assert_no_leak, \
    completeness_check, SPLIT_SEED, MODEL_SEEDS
from common.constants import RESULTS_V2_DIR
from models.ring_conditioned_gnn import build_model
from models.pyg_models import build_pyg_model

STAGE1_DIR = os.path.join(RESULTS_V2_DIR, 'stage1', 'gnn')
NODE_VEC_LEN, MAX_ATOMS = 60, 75
FINAL_SPLIT_VAL_RATIO = 0.125

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 30, 'patience': 20,
}

CUSTOM_MODELS = ['GNN', 'GIN', 'GAT', 'MPNN', 'GraphSAGE']
PYG_MODELS = ['DMPNN']
ALL_MODELS = CUSTOM_MODELS + PYG_MODELS


def final_train_val_split(dev_idx, groups, split_seed=SPLIT_SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=FINAL_SPLIT_VAL_RATIO, random_state=split_seed)
    tr_rel, va_rel = next(gss.split(np.arange(len(dev_idx)), groups=np.asarray(groups)[dev_idx]))
    return dev_idx[tr_rel], dev_idx[va_rel]


def _to_long(idx, device):
    return torch.tensor(np.asarray(idx), dtype=torch.long, device=device)


def _check_leaks(splits, groups, tag):
    assert_no_leak(splits['train_idx'], splits['test_idx'], groups, f"{tag} dev/test")
    for k, (tr, va) in enumerate(splits['folds']):
        assert_no_leak(tr, va, groups, f"{tag} fold{k+1}")


def run_model_on_task(model_name, task, params, device, output_root,
                      n_epochs=30, patience=8, seed=11):
    name = task['name']
    out_dir = os.path.join(output, name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n任务: {name} | 模型: {model_name} (Stage1 Base)\n{'='*60}", flush=True)
    t_total_start = time.time()

    if model_name in CUSTOM_MODELS:
        data = load_adj_format(task['dataset_path'], task['target_col'], NODE_VEC_LEN, MAX_ATOMS,
                              ring_flag_value=0, device=torch.device('cpu'))
        # 数据搬到 device
        data_gpu = data
        n = data['n']
        groups = make_group_ids(data['smiles'])
        splits = get_final_splits(n, groups, split_seed=SPLIT_SEED)
        _check_model_and_leaks(splits, groups, name)
        test_idx, cv_folds = splits['test_idx'], splits['folds']

        cv_rows = []
        for fold, (tr, va) in enumerate(cv_folds):
            tr_t = _to_long(tr, 'cuda'); va_t = _to_long(va, 'cuda')
            _set_seed(seed + fold)
            model = build_model(model_name, node_vec_len=NODE_VEC_LEN,
                                hidden_dim=params['hidden_dim'],
                                n_conv=params['n_conv_layers'],
                                n_hidden=params['n_hidden_layers'],
                                p_dropout=params['p_dropout'],
                                readout_mode='fixed_avg').to('cuda')
            t0 = time.time()
            model = train_custom_model(model, params, data_gpu, tr_t, va_t, 'cuda',
                                       n_epochs=n_epochs, patience=patience, seed=seed + fold)
            (r2, mae, rmse), _, _ = eval_custom_model(model, data_gpu, va_t, params['batch_size'], 'cuda')
            print(f"  Fold {fold+1}: R2={r3:.4f} MAE={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
            cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
            del model; torch.cuda.empty_cache()

        final_tr, final_va = final_train_val_split(dev_idx, groups)
        _no_leak(final_tr, final_va, groups, f"{name}/{model_name} final-train-val")
        _set_seed(seed)
        final_model = build_model(model_name, n=NODE_VEC_LEN,
                                  hidden=params['hidden_dim'],
                                  conv=params['n_conv_layers'],
                                  hidden2=params['n_hidden_layers'],
                                  do=params['p_dropout'], readout='fixed_avg').to('cuda')
        final_model = train_custom_model(final_model, params, data(idx_h, _to_long(final_va,'cuda'),
                                    'cuda', seed=self)
        (tr_r2, tr_mae, tr_rmse), _, _ = eval_custom_model(final_model, data2, params['batch_size'])
        (te_r2, te_mae, te_rmse), te_pred, te_true = eval_custom_model(final_model, test)
        torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        del final_model; torch.cuda.empty_cache()

    elif model_name in PYG_MODELS:
        load_pyg_format(...)  # DMPNN 不在此 dry-run, 保留接口
        pass
    else:
        raise ValueError(f"未知模型: {model_name}")

    result = save_results(out_dir, model_name, 'base', name, n,
                          cv_rows, (tr_r2, tr_mae, tr_rmse), (te_r2, te_mae, te_rmse),
                          te_pred, te_true, train_time_sec)
    result.update({'seed': seed, 'run_status': 'OK', 'error_message': ''})
    print(f"  CV-MAE=... Test-MAE={te_mae:.4f}")
    return result


def main():
    null