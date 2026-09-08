"""
芳香性 vs 非芳香拆分测试

将数据集按 RDKit GetIsAromatic 分成芳香和非芳香两组,
分别训练+测试, 与全部数据训练+测试 (对照) 对比。

三种模式:
  a) aromatic:     仅芳香子集训练+测试
  b) non_aromatic: 仅非芳香子集训练+测试
  c) all:          全部数据训练+测试 (对照)

每个模式 × 每个任务 × 每个模型 (GNN, MPNN):
  - 5折CV + 最终模型 (与 Layer 2 一致的 canonical_splits)
  - 超参数与 Layer 2 一致: hidden_dim=128, n_conv=3, n_hidden=2,
    lr=0.001, dropout=0.2, batch_size=64, epochs=200, patience=30
  - ring_flag_value=10 (label 编码)
"""
import os
import sys
import csv
import time
import random
import argparse
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

sys.path.insert(0, '_PROJ_ROOT + "/code_end"')
sys.path.insert(0, '_PROJ_ROOT + "/unified_models"')

from common.tasks import TASKS, clean_dataset_csv, canonical_splits, compute_metrics, DEFAULT_SEED
from common.graph_data import load_adj_format
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel

# 与 Layer 2 一致的超参数
DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}

CUSTOM_MODELS = {
    'GNN': GNNModel,
    'MPNN': MPNNModel,
}

NVL, MAX_ATOMS = 60, 75
RING_FLAG_VALUE = 10  # label 编码

MODES = ['aromatic', 'non_aromatic', 'all']


def set_full_seed(seed):
    """设置完整随机种子 (torch + numpy + random)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_aromatic_smiles(smiles):
    """判断 SMILES 是否含芳香原子 (RDKit GetIsAromatic)"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(a.GetIsAromatic() for a in mol.GetAtoms())


def split_by_aromaticity(smiles_list):
    """按芳香性分组, 返回 (aromatic_indices, non_aromatic_indices) np.ndarray"""
    aromatic_idx = []
    non_aromatic_idx = []
    for i, smi in enumerate(smiles_list):
        if is_aromatic_smiles(smi):
            aromatic_idx.append(i)
        else:
            non_aromatic_idx.append(i)
    return np.array(aromatic_idx, dtype=np.int64), np.array(non_aromatic_idx, dtype=np.int64)


def report_group_stats(outputs, indices, name):
    """报告某组样本的目标值分布"""
    if len(indices) == 0:
        print(f"  [{name}] n=0 (空集)")
        return
    vals = outputs[indices].cpu().numpy()
    print(f"  [{name}] n={len(indices)}, "
          f"target: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"min={vals.min():.4f}, max={vals.max():.4f}")


def train_model(model_name, params, data, train_idx, val_idx, device,
                n_epochs=200, patience=30, seed=42):
    """训练单个模型 (label 编码, 与 Layer 2 train_custom_model 一致)

    - set_full_seed(seed) 在模型创建前调用
    - Adam optimizer + ReduceLROnPlateau scheduler
    - MSE Loss + grad clipping (1.0)
    - Early stopping with patience
    """
    set_full_seed(seed)
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    outputs = data['outputs']
    nvl = data['node_vec_len']
    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    kwargs = dict(node_vec_len=nvl, hidden_dim=params['hidden_dim'],
                  n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
                  n_outputs=1, p_dropout=params['p_dropout'], mode='label')
    model = CUSTOM_MODELS[model_name](**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze()
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze())
                vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
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


@torch.no_grad()
def eval_model(model, data, idx, batch_size, device):
    """评估模型, 返回 (r2, mae, rmse), pred, true"""
    model.eval()
    node_mats, adj_mats, outputs = data['node_mats'], data['adj_mats'], data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        preds.append(model(node_mats[bi], adj_mats[bi]).squeeze())
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    r2, mae, rmse = compute_metrics(true, pred)
    return (r2, mae, rmse), pred, true


def run_mode_test(model_name, task, params, data, subset_indices, mode, device,
                  output_root, n_epochs=200, patience=30, seed=42):
    """执行单个模式 (aromatic/non_aromatic/all) 的训练+评估

    subset_indices: 该模式包含的样本在完整数据集中的索引 (np.ndarray)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name = task['name']
    out_dir = os.path.join(output_root, name, mode, model_name)
    os.makedirs(out_dir, exist_ok=True)

    n_subset = len(subset_indices)
    print(f"\n{'='*60}")
    print(f"任务: {name} | 模型: {model_name} | 模式: {mode} | n={n_subset}")
    print(f"{'='*60}")

    if n_subset < 10:
        print(f"  样本数过少 ({n_subset}), 跳过")
        return None

    # 在子集上生成 canonical_splits (索引相对子集)
    test_idx_rel, cv_folds_rel, final_tr_rel, final_va_rel = canonical_splits(n_subset, seed=seed)

    # 映射回完整数据集索引, 并转为 device 上的 tensor
    subset_indices_t = torch.tensor(subset_indices, device=device, dtype=torch.long)

    def map_idx(rel_idx):
        return subset_indices_t[torch.tensor(rel_idx, device=device, dtype=torch.long)]

    test_idx = map_idx(test_idx_rel)
    final_tr = map_idx(final_tr_rel)
    final_va = map_idx(final_va_rel)

    t_total_start = time.time()

    # 5折 CV
    cv_rows = []
    for fold, (tr_rel, va_rel) in enumerate(cv_folds_rel):
        tr_t = map_idx(tr_rel)
        va_t = map_idx(va_rel)
        t0 = time.time()
        model = train_model(model_name, params, data, tr_t, va_t, device,
                            n_epochs, patience, seed=seed + fold)
        (r2, mae, rmse), _, _ = eval_model(model, data, va_t, params['batch_size'], device)
        print(f"  Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} ({time.time()-t0:.0f}s)")
        cv_rows.append({'fold': fold+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        del model; torch.cuda.empty_cache()

    # 最终模型
    final_model = train_model(model_name, params, data, final_tr, final_va, device,
                              n_epochs, patience, seed=seed)
    (tr_r2, tr_mae, tr_rmse), _, _ = eval_model(final_model, data, final_tr, params['batch_size'], device)
    (te_r2, te_mae, te_rmse), te_pred, te_true = eval_model(final_model, data, test_idx,
                                                             params['batch_size'], device)
    torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
    del final_model; torch.cuda.empty_cache()

    train_time_sec = time.time() - t_total_start

    # ddof=1 样本标准差
    cv_r2 = np.mean([r['r2'] for r in cv_rows])
    cv_r2_std = np.std([r['r2'] for r in cv_rows], ddof=1) if len(cv_rows) > 1 else 0.0
    cv_mae = np.mean([r['mae'] for r in cv_rows])
    cv_rmse = np.mean([r['rmse'] for r in cv_rows])

    print(f"  [{model_name}/{mode}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | "
          f"Test: R2={te_r2:.4f} MAE={te_mae:.4f} RMSE={te_rmse:.4f} | Time: {train_time_sec:.0f}s")

    # 保存结果
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': te_true, 'pred': te_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', name])
        w.writerow(['mode', mode]); w.writerow(['n', n_subset])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', tr_r2]); w.writerow(['train_mae', tr_mae]); w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2]); w.writerow(['test_mae', te_mae]); w.writerow(['test_rmse', te_rmse])
        w.writerow(['train_time_sec', train_time_sec])

    # 散点图
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(te_true, te_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(te_true.min(), te_pred.min()), max(te_true.max(), te_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{name} - {model_name} ({mode})\nTest R2={te_r2:.4f}, MAE={te_mae:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {
        'task': name, 'model': model_name, 'mode': mode, 'n': n_subset,
        'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
        'train_r2': tr_r2, 'train_mae': tr_mae, 'train_rmse': tr_rmse,
        'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse,
        'train_time_sec': train_time_sec,
    }


def main():
    parser = argparse.ArgumentParser(description='芳香性 vs 非芳香拆分测试')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/aromatic_split/results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--models', type=str, default='all',
                        help='逗号分隔: GNN,MPNN 或 all')
    parser.add_argument('--tasks', type=str, default='all',
                        help='逗号分隔任务名, 或 all')
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

    model_list = list(CUSTOM_MODELS.keys()) if args.models == 'all' else args.models.split(',')
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        print(f"\n{'#'*60}\n# 任务: {task['name']}\n{'#'*60}")
        # 加载完整数据 (label 编码, ring_flag_value=10)
        data = load_adj_format(task['dataset_path'], task['target_col'],
                               NVL, MAX_ATOMS, ring_flag_value=RING_FLAG_VALUE, device=device)
        smiles_list = data['smiles']
        outputs = data['outputs']
        n_total = data['n']

        # 按芳香性分组
        aromatic_idx, non_aromatic_idx = split_by_aromaticity(smiles_list)
        all_idx = np.arange(n_total, dtype=np.int64)

        print(f"\n  === 芳香性分组统计 ===")
        report_group_stats(outputs, aromatic_idx, 'aromatic')
        report_group_stats(outputs, non_aromatic_idx, 'non_aromatic')
        report_group_stats(outputs, all_idx, 'all')

        # 三种模式对应的子集索引
        mode_indices = {
            'aromatic': aromatic_idx,
            'non_aromatic': non_aromatic_idx,
            'all': all_idx,
        }

        for mode in MODES:
            subset_indices = mode_indices[mode]
            for model_name in model_list:
                try:
                    res = run_mode_test(model_name, task, params, data, subset_indices,
                                        mode, device, args.output_dir,
                                        args.n_epochs, args.patience, args.seed)
                    if res is not None:
                        all_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{model_name}/{mode}/{task['name']}] 失败: {e}")

    # 汇总
    if all_results:
        cols = ['task', 'mode', 'model', 'n',
                'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
                'train_r2', 'train_mae', 'train_rmse',
                'test_r2', 'test_mae', 'test_rmse', 'train_time_sec']
        df_out = pd.DataFrame(all_results)[cols]
        print(f"\n{'='*60}\n芳香性拆分测试汇总\n{'='*60}")
        print(df_out.to_string(index=False))
        out_csv = os.path.join(args.output_dir, 'aromatic_split_summary.csv')
        df_out.to_csv(out_csv, index=False)
        print(f"\n汇总结果已保存: {out_csv}")

        # 对比表 (pivot: 行=task+model, 列=各 mode 的 test_r2/test_mae/test_rmse/cv_r2/n)
        pivot_rows = []
        for task in task_list:
            for model_name in model_list:
                row = {'task': task['name'], 'model': model_name}
                for mode in MODES:
                    sub = df_out[(df_out['task'] == task['name']) &
                                 (df_out['model'] == model_name) &
                                 (df_out['mode'] == mode)]
                    if len(sub) > 0:
                        s = sub.iloc[0]
                        row[f'{mode}_n'] = int(s['n'])
                        row[f'{mode}_cv_r2'] = s['cv_r2']
                        row[f'{mode}_test_r2'] = s['test_r2']
                        row[f'{mode}_test_mae'] = s['test_mae']
                        row[f'{mode}_test_rmse'] = s['test_rmse']
                    else:
                        row[f'{mode}_n'] = None
                        row[f'{mode}_cv_r2'] = None
                        row[f'{mode}_test_r2'] = None
                        row[f'{mode}_test_mae'] = None
                        row[f'{mode}_test_rmse'] = None
                pivot_rows.append(row)
        df_cmp = pd.DataFrame(pivot_rows)
        cmp_csv = os.path.join(args.output_dir, 'comparison_table.csv')
        df_cmp.to_csv(cmp_csv, index=False)
        print(f"对比表已保存: {cmp_csv}")


if __name__ == '__main__':
    main()
