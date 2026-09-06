"""
Layer 4 统一训练评估管线

每种方法的训练流程:
1. 加载数据 (load_adj_format)
2. 方法特定的数据预处理 (add_xxx_features)
3. 5折CV + 最终模型 (canonical_splits)
4. 在主测试集上评估
5. 在 lunci6 上评估 (集外验证)

关键修复 (相对 Layer 3):
  - set_full_seed(seed) 必须在模型创建前调用 (保证可复现)
  - lunci6 评估: 用主数据集训练的模型直接测试, max_atoms=85 容纳大分子
  - 每种方法在 lunci6 上评估时做相同的预处理 (add_hammett_features 等)
"""
import os
import sys
import csv
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# 统一导入路径 (与任务约定一致)
sys.path.insert(0, '_PROJ_ROOT + "/code_end"')
sys.path.insert(0, '_PROJ_ROOT + "/unified_models"')

from common.tasks import (
    TASKS, clean_dataset_csv, canonical_splits, compute_metrics,
    EXTERNAL_TEST_FILES, DEFAULT_SEED,
)
from common.graph_data import load_adj_format
from generalization_test.code.splits import prepare_external_test_csv

# 8 种方法
from layer4_substituent.code.m1_hammett_embedding import (
    add_hammett_features, build_model as build_m1,
)
from layer4_substituent.code.m2_monotonicity_loss import (
    compute_sigma_sums, build_model as build_m2, build_loss as build_m2_loss,
)
from layer4_substituent.code.m3_hierarchical_attention import (
    add_position_encodings as add_pos_m3, build_model as build_m3,
)
from layer4_substituent.code.m4_dual_channel import (
    add_dual_adjacency, build_model as build_m4,
)
from layer4_substituent.code.m5_ssl_pretrain import (
    generate_substituent_pairs, pretrain_model, finetune_model, SSLFinetuneModel,
    build_pair_features, build_model as build_m5,
)
from layer4_substituent.code.m6_position_encoding import (
    add_position_features, build_model as build_m6,
)
from layer4_substituent.code.m7_perturbation import (
    add_position_encodings as add_pos_m7, build_model as build_m7,
)
from layer4_substituent.code.m8_multi_scale_pooling import (
    add_substituent_groups, build_model as build_m8,
    add_position_encodings as add_pos_m8,
)


# ============== 训练参数 (与 Layer 2 一致) ==============
DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}

# 数据维度常量
NVL = 60                # 原始节点特征长度
MAX_ATOMS = 75          # 主数据集最大原子数
EXT_MAX_ATOMS = 85      # lunci6 外部测试集最大原子数 (含大分子)
RING_FLAG_VALUE = 10    # label 编码的目标环标记值

# 方法 → 模型构建函数 映射 (m5 单独处理: 预训练+微调)
_BUILDERS = {
    'm1': build_m1, 'm2': build_m2, 'm3': build_m3, 'm4': build_m4,
    'm6': build_m6, 'm7': build_m7, 'm8': build_m8,
}


def set_full_seed(seed):
    """设置完整随机种子 (torch + numpy + random + cuda)

    重要: 必须在模型创建前调用, 修复 Layer 3 的可复现性 bug。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== 方法特定数据预处理 ==============
def preprocess_data(method, data, max_atoms, device):
    """根据方法对数据做特定预处理 (主数据集与 lunci6 共用)

    Args:
        method: 'm1' ~ 'm8'
        data:   load_adj_format 返回的字典
        max_atoms: 当前数据集的最大原子数 (主=75, lunci6=85)
        device: 张量所在设备

    Returns:
        预处理后的 data 字典 (新增方法特定字段)
    """
    if method == 'm1':
        # Hammett σ 嵌入: node_mats 扩展 (N, max_atoms, NVL+6)
        return add_hammett_features(data, max_atoms=max_atoms,
                                    node_vec_len=NVL, device=device)
    elif method == 'm2':
        # 单调性约束: 计算每个分子的 Σσ (仅训练时用于损失, 评估不需要)
        new_data, _ = compute_sigma_sums(data, sigma_mode='both', device=device)
        return new_data
    elif method == 'm3':
        # 分层交叉注意力: 生成位置编码矩阵
        return add_pos_m3(data, max_atoms=max_atoms, device=device)
    elif method == 'm4':
        # 双通道消息传递: 生成共轭/诱导双通道邻接矩阵
        return add_dual_adjacency(data, max_atoms=max_atoms, device=device)
    elif method == 'm5':
        # SSL 预训练: 无额外预处理 (使用标准 node_mats/adj_mats)
        return data
    elif method == 'm6':
        # 位置编码: node_mats 扩展 (N, max_atoms, NVL+6)
        return add_position_features(data, max_atoms=max_atoms,
                                     node_vec_len=NVL, device=device)
    elif method == 'm7':
        # 扰动学习: 生成位置编码矩阵 (区分环原子/取代基原子)
        return add_pos_m7(data, max_atoms=max_atoms, device=device)
    elif method == 'm8':
        # 多尺度池化: 位置编码 + 取代基组 ID
        new_data = add_pos_m8(data, max_atoms=max_atoms, device=device)
        new_data = add_substituent_groups(new_data, max_atoms=max_atoms,
                                          device=device)
        return new_data
    else:
        raise ValueError(f"未知方法: {method} (可选 m1~m8)")


# ============== 前向传播分发 ==============
def _forward_batch(model, method, data, bi):
    """根据方法选择前向传播方式

    不同方法需要不同的额外输入:
        m1/m2/m5/m6: 标准 forward(node_mat, adj_mat) (扩展特征已拼接到 node_mat)
        m3/m7:       forward(node_mat, adj_mat, pos_enc=...)
        m4:          forward(node_mat, conj_adj=..., ind_adj=...)
        m8:          forward(node_mat, adj_mat, pos_enc=..., sub_groups=...)
    """
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    if method in ('m1', 'm2', 'm5', 'm6'):
        return model(node_mats[bi], adj_mats[bi]).squeeze()
    elif method in ('m3', 'm7'):
        return model(node_mats[bi], adj_mats[bi],
                     pos_enc=data['pos_encs'][bi]).squeeze()
    elif method == 'm4':
        return model(node_mats[bi],
                     conj_adj=data['conj_adjs'][bi],
                     ind_adj=data['ind_adjs'][bi]).squeeze()
    elif method == 'm8':
        return model(node_mats[bi], adj_mats[bi],
                     pos_enc=data['pos_encs'][bi],
                     sub_groups=data['sub_groups'][bi]).squeeze()
    else:
        raise ValueError(f"未知方法: {method}")


# ============== 训练单个模型 ==============
def train_model(method, model_name, params, data, train_idx, val_idx, device,
                n_epochs=200, patience=30, seed=42):
    """训练单个模型

    Args:
        method:     'm1' ~ 'm8'
        model_name: 基础模型类型 ('gnn')
        params:     训练超参 (DEFAULT_PARAMS)
        data:       预处理后的数据字典
        train_idx:  训练样本索引 (GPU 张量)
        val_idx:    验证样本索引 (GPU 张量, 用于 early stopping)
        device:     torch.device
        n_epochs:   最大训练轮数
        patience:   early stopping 耐心值
        seed:       随机种子 (在模型创建前调用 set_full_seed)

    Returns:
        训练好的模型

    方法分支:
        m1~m4, m6~m8: 标准训练 (GNN + 不同架构/损失)
        m5:            先预训练 (取代基替换预测) 再微调
    """
    if method == 'm5':
        return _train_m5(params, data, train_idx, val_idx, device,
                         n_epochs, patience, seed)
    return _train_standard(method, model_name, params, data, train_idx, val_idx,
                           device, n_epochs, patience, seed)


def _train_standard(method, model_name, params, data, train_idx, val_idx, device,
                    n_epochs, patience, seed):
    """标准训练循环 (m1~m4, m6~m8)"""
    # 重要: set_full_seed 必须在模型创建前调用
    set_full_seed(seed)

    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)
    outputs = data['outputs']
    node_mats, adj_mats = data['node_mats'], data['adj_mats']

    # m1/m6 的 node_vec_len 已扩展, 模型构建需用原始 (未拼接) 长度
    model_nvl = data.get('orig_node_vec_len', data['node_vec_len'])

    build_kwargs = dict(
        base=model_name, node_vec_len=model_nvl,
        hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'], n_outputs=1,
        p_dropout=params['p_dropout'], mode='label',
    )
    model = _BUILDERS[method](**build_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    # 验证损失统一用 MSE (early stopping 标准)
    val_loss_fn = nn.MSELoss()

    # m2 使用单调性约束损失; 其余用 MSE
    if method == 'm2':
        loss_fn = build_m2_loss()
        sigma_sums = data['sigma_sums']
    else:
        loss_fn = nn.MSELoss()

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = _forward_batch(model, method, data, bi)
            if method == 'm2':
                # MonotonicityLoss 返回 (total, (mse, mono))
                total, _ = loss_fn(preds, outputs[bi], sigma_sums[bi])
                loss = total
            else:
                loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(_forward_batch(model, method, data, bi))
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = val_loss_fn(vp, vt).item()
        scheduler.step(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model


def _train_m5(params, data, train_idx, val_idx, device, n_epochs, patience, seed):
    """Method 5: SSL 预训练 + 微调

    流程:
        1. 从训练集生成取代基替换对 (同骨架优先)
        2. 预训练 SSLPretrainModel 预测 ΔP = P_modified - P_original
        3. 用预训练编码器构建 SSLFinetuneModel, 在下游任务上微调 (带 early stopping)
    """
    # 重要: set_full_seed 必须在模型创建前调用
    set_full_seed(seed)

    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)
    outputs = data['outputs']
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    model_nvl = data.get('orig_node_vec_len', data['node_vec_len'])
    max_atoms = data['max_atoms']

    # 1. 生成取代基替换对 (仅用训练集数据)
    train_idx_np = train_idx.cpu().numpy()
    smiles_list = [data['smiles'][i] for i in train_idx_np]
    targets = outputs[train_idx].cpu().numpy().tolist()
    pairs = generate_substituent_pairs(smiles_list, targets, seed=seed)
    pairs_data = build_pair_features(pairs, node_vec_len=model_nvl,
                                     max_atoms=max_atoms)

    # 2. 构建 SSLPretrainModel 并预训练 (共享 GNN 编码器)
    ssl_pretrain = build_m5(
        base='gnn', node_vec_len=model_nvl, hidden_dim=params['hidden_dim'],
        n_conv=params['n_conv_layers'], n_hidden=1, n_outputs=1,
        p_dropout=params['p_dropout'], mode='label',
    )
    # 预训练轮数: 取主训练轮数的 1/4, 限制在 [10, 50]
    pretrain_epochs = max(10, min(50, n_epochs // 4))
    ssl_pretrain = pretrain_model(
        ssl_pretrain, pairs_data, epochs=pretrain_epochs,
        lr=params['learning_rate'], device=device,
        batch_size=batch_size, verbose=False,
    )

    # 3. 微调: SSLFinetuneModel (复用预训练编码器) + early stopping
    ft_model = SSLFinetuneModel(ssl_pretrain, hidden_dim=params['hidden_dim'],
                                p_dropout=params['p_dropout']).to(device)
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        ft_model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = ft_model(node_mats[bi], adj_mats[bi]).squeeze()
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()

        ft_model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(ft_model(node_mats[bi], adj_mats[bi]).squeeze())
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break

    if best_state is not None:
        ft_model.load_state_dict(best_state)
    del optimizer, scheduler
    return ft_model


# ============== 评估 ==============
@torch.no_grad()
def eval_model(model, method, data, idx, batch_size, device):
    """评估模型 (支持与训练数据不同的 data 字典, 用于 lunci6 集外测试)

    Args:
        model:  训练好的模型
        method: 'm1' ~ 'm8' (决定前向传播方式)
        data:   预处理后的数据字典 (须已做与训练相同的 preprocess_data)
        idx:    评估样本索引
        batch_size: 批大小
        device: 设备

    Returns:
        (r2, mae, rmse), pred, true
    """
    model.eval()
    outputs = data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        preds.append(_forward_batch(model, method, data, bi))
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2, mae, rmse = compute_metrics(true, pred)
    return (r2, mae, rmse), pred, true


# ============== 运行单个方法 × 任务 ==============
def run_method(method, task, params, device, output_root,
               n_epochs=200, patience=30, seed=42):
    """运行单个方法 × 任务: 5折CV + 最终模型 + 主测试集 + lunci6 验证

    Args:
        method:      'm1' ~ 'm8'
        task:        TASKS 中的任务字典
        params:      训练超参
        device:      设备
        output_root: 输出根目录 (结果保存到 output_root/{task}/{method}/)
        n_epochs, patience, seed: 训练控制参数

    Returns:
        结果字典 (含 CV/主测试集/lunci6 指标)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name = task['name']
    out_dir = os.path.join(output_root, name, method)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n任务: {name} | 方法: {method}\n{'='*60}")

    t_total_start = time.time()

    # 1. 加载主数据集
    data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                           ring_flag_value=RING_FLAG_VALUE, device=device)
    n = data['n']

    # 2. 方法特定预处理
    data = preprocess_data(method, data, MAX_ATOMS, device)

    # 3. 规范划分 (5折CV + 最终模型)
    test_idx, cv_folds, final_tr, final_va = canonical_splits(n, seed=seed)

    # 4. 5折交叉验证
    cv_rows = []
    for fold, (tr, va) in enumerate(cv_folds):
        tr_t = torch.tensor(tr, device=device)
        va_t = torch.tensor(va, device=device)
        t0 = time.time()
        model = train_model(method, 'gnn', params, data, tr_t, va_t, device,
                            n_epochs, patience, seed + fold)
        (r2, mae, rmse), _, _ = eval_model(model, method, data, va_t,
                                           params['batch_size'], device)
        print(f"  Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} "
              f"({time.time()-t0:.0f}s)")
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        del model
        torch.cuda.empty_cache()

    # 5. 最终模型 (在 final_train 上训练, final_val 上 early stopping)
    final_tr_t = torch.tensor(final_tr, device=device)
    final_va_t = torch.tensor(final_va, device=device)
    test_idx_t = torch.tensor(test_idx, device=device)
    final_model = train_model(method, 'gnn', params, data, final_tr_t, final_va_t,
                              device, n_epochs, patience, seed)
    (tr_r2, tr_mae, tr_rmse), _, _ = eval_model(final_model, method, data,
                                                final_tr_t, params['batch_size'], device)
    (te_r2, te_mae, te_rmse), te_pred, te_true = eval_model(
        final_model, method, data, test_idx_t, params['batch_size'], device)
    torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    # 6. lunci6 集外验证 (用主数据集训练的模型直接测试)
    ext_r2 = ext_mae = ext_rmse = float('nan')
    ext_pred = ext_true = None
    try:
        ext_csv, ext_target = prepare_external_test_csv(name, test_source='lunci6')
        ext_data = load_adj_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                                   ring_flag_value=RING_FLAG_VALUE, device=device)
        # 与训练时相同的预处理 (max_atoms=85)
        ext_data = preprocess_data(method, ext_data, EXT_MAX_ATOMS, device)
        ext_idx = torch.arange(ext_data['n'], device=device)
        (ext_r2, ext_mae, ext_rmse), ext_pred, ext_true = eval_model(
            final_model, method, ext_data, ext_idx, params['batch_size'], device)
        print(f"  lunci6: R2={ext_r2:.4f} MAE={ext_mae:.4f} RMSE={ext_rmse:.4f} "
              f"(n={len(ext_idx)})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  lunci6 评估失败: {e}")

    del final_model
    torch.cuda.empty_cache()

    train_time_sec = time.time() - t_total_start

    # 7. 指标汇总
    cv_r2 = float(np.mean([r['r2'] for r in cv_rows]))
    cv_r2_std = float(np.std([r['r2'] for r in cv_rows], ddof=1)) if len(cv_rows) > 1 else 0.0
    cv_mae = float(np.mean([r['mae'] for r in cv_rows]))
    cv_rmse = float(np.mean([r['rmse'] for r in cv_rows]))

    print(f"  [{method}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | "
          f"Test: R2={te_r2:.4f} MAE={te_mae:.4f} RMSE={te_rmse:.4f} | "
          f"lunci6: R2={ext_r2:.4f} | Time: {train_time_sec:.0f}s")

    # 8. 保存结果文件
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)
    if ext_pred is not None:
        pd.DataFrame({'true': ext_true, 'pred': ext_pred}).to_csv(
            os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['method', method])
        w.writerow(['task', name])
        w.writerow(['n', n])
        w.writerow(['cv_r2_mean', cv_r2])
        w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae])
        w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', tr_r2])
        w.writerow(['train_mae', tr_mae])
        w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2])
        w.writerow(['test_mae', te_mae])
        w.writerow(['test_rmse', te_rmse])
        w.writerow(['lunci6_r2', ext_r2])
        w.writerow(['lunci6_mae', ext_mae])
        w.writerow(['lunci6_rmse', ext_rmse])
        w.writerow(['train_time_sec', train_time_sec])

    # 散点图 (主测试集)
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(te_true, te_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(te_true.min(), te_pred.min()), max(te_true.max(), te_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'{name} - {method}\nTest R2={te_r2:.4f}, MAE={te_mae:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight')
    plt.close()

    return {
        'method': method, 'task': name, 'n': n,
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae, 'train_rmse': tr_rmse,
        'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'lunci6_r2': ext_r2, 'lunci6_mae': ext_mae, 'lunci6_rmse': ext_rmse,
        'train_time_sec': train_time_sec,
    }
