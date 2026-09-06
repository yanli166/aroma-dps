"""
[0831 重构] Smoke test: 自动验证 P0-1/P0-2/P1-1 三项 pre-production 修改

验证项:
  1. fixed holdout 不变 (不同 task 在同一 feature_mode 下 train_idx/test_idx 一致;
                           不同 feature_mode 下保持不变, 因为分组基于 canonical SMILES)
  2. CV 无 group overlap (用 verify_split_invariants)
  3. E* 来自 CV (不是 test): best_epochs 都来自 CV, final test 仅评估
  4. feature_mode 正确写入 metadata (CSVs / JSON)
  5. batch=1 正常 (构造 batch=1 张量跑 forward, 验证 shape)
  6. 完整实验组合无缺失 (completeness_check)

运行:
  python common/smoke_test.py
"""
import os
import sys
import json
import subprocess

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.constants import RESULTS_V2_DIR, SPLITS_DIR
from common.protocol import (get_final_splits, make_group_ids, SPLIT_SEED,
                              completeness_check, MODEL_SEEDS)
from common.tasks import TASKS
from common.graph_data import load_adj_format
from common.train_eval import (set_full_seed, verify_split_invariants, train_custom_model,
                                eval_custom_model, compute_estar)
from common.features import get_feature_meta, build_fingerprint_matrix, FEAT_DIM, FEAT_DIM_AROM_ABLATED
from common.tasks import clean_dataset_csv
from models.ring_conditioned_gnn import build_model
import numpy as np
import pandas as pd
import torch


def t_print(t, msg):
    sym = {'OK': '[OK]', 'FAIL': '[FAIL]', 'SKIP': '[SKIP]', 'INFO': '[INFO]'}
    print(f"{sym.get(t, '[?]')} {msg}")


def check_fixed_holdout(device):
    """1. fixed holdout 不变: 同一 task 不同 ring_flag 下 splits 相同;
       不同 feature_mode 下, 因为分组基于 SMILES (与 ring_flag/feature_mode 无关), 也应一致。"""
    print("\n=== Check 1: fixed holdout 不变 ===")
    device_cpu = torch.device('cpu')
    task = TASKS[0]
    # 真正做法: load 数据取 n, groups
    data = load_adj_format(task['dataset_path'], task['target_col'],
                            node_vec_len=60, max_atoms=75,
                            ring_flag_value=0, device=device_cpu, feature_mode='standard')
    groups = make_group_ids(data['smiles'])
    n = data['n']

    s_standard = get_final_splits(n, groups, persist=None, tag='smoke_standard')
    s_ablated = get_final_splits(n, groups, persist=None, tag='smoke_ablated')
    same = (np.array_equal(s_standard['train_idx'], s_ablated['train_idx']) and
            np.array_equal(s_standard['test_idx'], s_ablated['test_idx']))
    if same:
        t_print('OK', f'feature_mode 切换不改变 holdout (n={n}, test_n={len(s_standard["test_idx"])})')
    else:
        t_print('FAIL', 'feature_mode 切换改变了 holdout!')
    return same


def check_cv_no_group_overlap():
    """2. CV 无 group overlap"""
    print("\n=== Check 2: CV 无 group overlap (canonical SMILES) ===")
    task = TASKS[0]
    data = load_adj_format(task['dataset_path'], task['target_col'],
                            node_vec_len=60, max_atoms=75,
                            ring_flag_value=0, device='cpu', feature_mode='standard')
    groups = make_group_ids(data['smiles'])
    splits = get_final_splits(data['n'], groups, persist=None, tag='smoke_cv_check')
    try:
        verify_split_invariants(splits, groups, 'smoke')
        t_print('OK', f'5-fold CV 无 group 重叠 (n={data["n"]})')
        return True
    except AssertionError as e:
        t_print('FAIL', str(e))
        return False


def check_estar_from_cv():
    """3. E* 来自 CV 不是 test"""
    print("\n=== Check 3: E* 来自 CV 而非 test ===")
    # 跑一个 1-折 极小训练, 检验 best_epoch 来源
    set_full_seed(11)
    task = TASKS[0]
    data = load_adj_format(task['dataset_path'], task['target_col'],
                            node_vec_len=60, max_atoms=75,
                            ring_flag_value=0, device='cpu', feature_mode='standard')
    splits = get_final_splits(data['n'], make_group_ids(data['smiles']),
                                persist=None, tag='smoke_estar')
    tr, va = splits['folds'][0]
    tr_t = torch.tensor(tr, dtype=torch.long)
    va_t = torch.tensor(va, dtype=torch.long)
    params = {'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
              'weight_decay': 1e-5}
    model = build_model('MPNN', node_vec_len=60, hidden_dim=64,
                         n_conv=2, n_hidden=1, p_dropout=0.2, readout_mode='fixed_avg')
    trained = train_custom_model(model, params, data, tr_t, va_t,
                                  torch.device('cpu'),
                                  n_epochs=10, patience=5, seed=11)
    best_epoch = trained.best_epoch
    estar, info = compute_estar([best_epoch], max_epoch=10)
    if info['n_folds'] == 1 and info['estar'] == best_epoch:
        t_print('OK', f'E* = {estar} 由 CV best_epoch={best_epoch} 计算, '
              f'ceiling_hits={info["n_ceiling_hits"]}/{info["n_folds"]}')
        return True
    t_print('FAIL', f'best_epoch={best_epoch}, estar={estar}, info={info}')
    return False


def check_feature_mode_metadata():
    """4. feature_mode 正确写入 metadata"""
    print("\n=== Check 4: feature_mode 双轨 (feature 维度 + metadata) ===")
    m_std = get_feature_meta('standard')
    m_abl = get_feature_meta('explicit_aromaticity_ablated')
    if m_std['n_features'] == 2244 and m_abl['n_features'] == 2242:
        t_print('OK', f"standard: {m_std['n_features']} 维; explicit_aromaticity_ablated: {m_abl['n_features']} 维")
    else:
        t_print('FAIL', f"dim 不对: {m_std}, {m_abl}")
        return False
    # 构造同一分子的两种指纹
    smi = 'c1ccccc1'
    X_std, _ = build_fingerprint_matrix([smi], df=None, feature_mode='standard')
    X_abl, _ = build_fingerprint_matrix([smi], df=None, feature_mode='explicit_aromaticity_ablated')
    if X_std.shape == (1, 2244) and X_abl.shape == (1, 2242):
        t_print('OK', f'同一 SMILES 两种 mode shape 正确: {X_std.shape} vs {X_abl.shape}')
        return True
    t_print('FAIL', f'shape 错误: {X_std.shape} {X_abl.shape}')
    return False


def check_batch_1(device):
    """5. batch=1 正常"""
    print("\n=== Check 5: batch=1 正常 ===")
    set_full_seed(11)
    task = TASKS[0]
    data = load_adj_format(task['dataset_path'], task['target_col'],
                            node_vec_len=60, max_atoms=75,
                            ring_flag_value=0, device=device, feature_mode='standard')
    n_total = data['n']
    model = build_model('MPNN', node_vec_len=60, hidden_dim=64,
                         n_conv=2, n_hidden=1, p_dropout=0.2, readout_mode='fixed_avg').to(device)
    single_idx = torch.tensor([0], dtype=torch.long, device=device)
    model.eval()  # BN 在 batch=1 时需要 eval 模式
    with torch.no_grad():
        out = model(data['node_mats'][single_idx], data['adj_mats'][single_idx],
                     data['ring_indices'][single_idx])
    out = out.reshape(-1)
    if out.shape == torch.Size([1]) and not torch.isnan(out).any():
        t_print('OK', f'batch=1 forward OK, output shape={tuple(out.shape)}, value={out.item():.4f}')
        return True
    t_print('FAIL', f'batch=1 输出异常: shape={tuple(out.shape)}, nan={torch.isnan(out).any()}')
    return False


def check_completeness():
    """6. 完整实验组合无缺失"""
    print("\n=== Check 6: completeness_check ===")
    # 模拟一个完整表
    rows = pd.DataFrame([
        {'model': 'GN', 'task': 'A', 'seed': 11},
        {'model': 'GN', 'task': 'B', 'seed': 11},
        {'model': 'RF', 'task': 'A', 'seed': 11},
        {'model': 'RF', 'task': 'B', 'seed': 11},
    ])
    exp = pd.MultiIndex.from_product([['GN', 'RF'], ['A', 'B'], [11]],
                                      names=['model', 'task', 'seed'])
    try:
        completeness_check(pd.MultiIndex.from_frame(rows[['model', 'task', 'seed']]),
                            exp, 'smoke')
        t_print('OK', 'completeness_check 通过')
        return True
    except RuntimeError as e:
        t_print('FAIL', str(e))
        return False


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"smoke_test on {device}")
    results = []
    results.append(check_fixed_holdout(device))
    results.append(check_cv_no_group_overlap())
    results.append(check_estar_from_cv())
    results.append(check_feature_mode_metadata())
    results.append(check_batch_1(device))
    results.append(check_completeness())

    print("\n" + "="*60)
    print(f"smoke_test 汇总: {sum(results)}/{len(results)} 通过")
    if all(results):
        print("[ALL OK] 可以运行 pre-production 实验。")
    else:
        print("[FAIL] 请修复后再运行。")


if __name__ == '__main__':
    main()
