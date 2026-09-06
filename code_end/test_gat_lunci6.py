"""
快速测试: GAT 和 AttentiveFP(FPAttention) 在 lunci6 上的表现

补测 GAT (HOMA/NICS 之前没测过 lunci6) + 尝试 AttentiveFP
"""
import os, sys, time, csv
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

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
os.chdir(PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)
from common.constants import ORIG_MODELS_ROOT
sys.path.insert(0, ORIG_MODELS_ROOT)

from common.tasks import TASKS, DEFAULT_SEED, compute_metrics
from common.graph_data import load_adj_format
from unified_models.gat.model import GATModel
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from generalization_test.code.splits import prepare_external_test_csv

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results', 'gat_lunci6_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NVL, MAX_ATOMS, EXT_MAX_ATOMS = 60, 75, 85
RING_FLAG_VALUE = 10
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}
CUSTOM_MODELS = {'GNN': GNNModel, 'GAT': GATModel, 'MPNN': MPNNModel}


def set_full_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_eval_lunci6(model_name, task, device, seed=42):
    """训练模型并在 lunci6 上评估"""
    set_full_seed(seed)

    # 主数据集
    train_data = load_adj_format(task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
                                 ring_flag_value=RING_FLAG_VALUE, device=device)
    n = train_data['n']

    # 87.5/12.5 train/val split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(0.125 * n)
    val_idx = torch.tensor(perm[:n_val], device=device)
    train_idx = torch.tensor(perm[n_val:], device=device)

    # lunci6 测试集
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source='lunci6')
    test_data = load_adj_format(ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
                                ring_flag_value=RING_FLAG_VALUE, device=device)
    test_idx = torch.arange(test_data['n'], device=device)

    print(f"  train={len(train_idx)}, val={len(val_idx)}, lunci6_test={len(test_idx)}")

    # 模型 (注意: lunci6 用 EXT_MAX_ATOMS=85, 但 node_vec_len 由数据决定)
    nvl = train_data['node_vec_len']
    kwargs = dict(node_vec_len=nvl, hidden_dim=PARAMS['hidden_dim'],
                  n_conv=PARAMS['n_conv_layers'], n_hidden=PARAMS['n_hidden_layers'],
                  n_outputs=1, p_dropout=PARAMS['p_dropout'], mode='label')
    if model_name == 'GAT':
        kwargs['n_heads'] = 4
    model = CUSTOM_MODELS[model_name](**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['learning_rate'],
                                 weight_decay=PARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PARAMS['patience']//3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    # 训练
    node_mats, adj_mats = train_data['node_mats'], train_data['adj_mats']
    outputs = train_data['outputs']
    batch_size = PARAMS['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    best_val, best_state, pcount = float('inf'), None, 0
    t0 = time.time()
    for epoch in range(1, PARAMS['n_epochs'] + 1):
        model.train()
        perm_t = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm_t[i:i+batch_size]
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
                bi = val_idx[i:i+batch_size]
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
        if pcount >= PARAMS['patience']:
            break
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: val_loss={vloss:.6f} (best={best_val:.6f})")

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估 lunci6 (注意: test_data 有自己的 node_mats/adj_mats)
    model.eval()
    te_node_mats, te_adj_mats = test_data['node_mats'], test_data['adj_mats']
    te_outputs = test_data['outputs']
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            bi = test_idx[i:i+batch_size]
            # 检查 node_vec_len 是否匹配
            te_nvl = te_node_mats[bi].shape[-1]
            if te_nvl != nvl:
                # pad 或 truncate
                batch_nm = te_node_mats[bi]
                if te_nvl < nvl:
                    pad = torch.zeros(*batch_nm.shape[:-1], nvl - te_nvl, device=device)
                    batch_nm = torch.cat([batch_nm, pad], dim=-1)
                else:
                    batch_nm = batch_nm[..., :nvl]
            else:
                batch_nm = te_node_mats[bi]
            preds.append(model(batch_nm, te_adj_mats[bi]).squeeze())
            trues.append(te_outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()

    r2, mae, rmse = compute_metrics(true, pred)

    # 保存预测
    out_dir = os.path.join(OUTPUT_DIR, task['name'], model_name)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'true': true, 'pred': pred}).to_csv(
        os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

    # 同时保存训练集 test (内部测试集) 评估
    with torch.no_grad():
        tp, tt = [], []
        for i in range(0, len(test_idx), batch_size):
            bi = test_idx[i:i+batch_size]
            tp.append(model(te_node_mats[bi], te_adj_mats[bi]).squeeze())
            tt.append(te_outputs[bi])

    print(f"  [{model_name}/{task['name']}] lunci6: R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | Time={train_time:.0f}s")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        'task': task['name'], 'model': model_name,
        'lunci6_r2': r2, 'lunci6_mae': mae, 'lunci6_rmse': rmse,
        'train_time_sec': train_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--models', type=str, default='GAT,GNN,MPNN')
    parser.add_argument('--tasks', type=str, default='all')
    args, _ = parser.parse_known_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_list = args.models.split(',')
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        print(f"\n{'#'*60}\n# 任务: {task['name']}\n{'#'*60}")
        for model_name in model_list:
            try:
                res = train_and_eval_lunci6(model_name, task, device)
                all_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{model_name}/{task['name']}] 失败: {e}")

    # 汇总
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'gat_lunci6_summary.csv'), index=False)
        print(f"\n{'='*60}\n汇总\n{'='*60}")
        print(df.to_string(index=False))
        print(f"\n已保存: {OUTPUT_DIR}/gat_lunci6_summary.csv")


if __name__ == '__main__':
    import argparse
    main()
