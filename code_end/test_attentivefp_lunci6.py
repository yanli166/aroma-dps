"""
快速测试: AttentiveFP (FPAttention) 在 lunci6 上的表现

Layer 2 中 AttentiveFP 已成功训练 (HOMA/NICS/MBCO CV R²=0.89-0.93),
但从未在 lunci6 上做集外测试。本脚本补测 AttentiveFP 的 lunci6 表现,
与 GAT/GNN/MPNN (并行运行中) 对比。
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
from common.graph_data import load_pyg_format
from baseline_gnn.code.pyg_models import build_pyg_model
from generalization_test.code.splits import prepare_external_test_csv

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results', 'attentivefp_lunci6_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NVL, MAX_ATOMS, EXT_MAX_ATOMS = 60, 75, 85
RING_FLAG_VALUE = 10
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}


def set_full_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_eval_lunci6(task, device, seed=42):
    """训练 AttentiveFP 并在 lunci6 上评估"""
    from torch_geometric.loader import DataLoader
    set_full_seed(seed)

    # 主数据集 (PyG format)
    print(f"  加载主数据集...", flush=True)
    train_data_list, nvl = load_pyg_format(
        task['dataset_path'], task['target_col'], NVL, MAX_ATOMS,
        ring_flag_value=RING_FLAG_VALUE, device='cpu')
    n = len(train_data_list)

    # 87.5/12.5 train/val split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(0.125 * n)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    print(f"  train={len(train_idx)}, val={len(val_idx)}", flush=True)

    # lunci6 测试集 (PyG format)
    print(f"  加载 lunci6 测试集...", flush=True)
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source='lunci6')
    test_data_list, _ = load_pyg_format(
        ext_csv, ext_target, NVL, EXT_MAX_ATOMS,
        ring_flag_value=RING_FLAG_VALUE, device='cpu')
    print(f"  lunci6_test={len(test_data_list)}", flush=True)

    # 模型
    in_channels = train_data_list[0].x.size(-1)
    edge_dim = train_data_list[0].edge_attr.size(-1) if train_data_list[0].edge_attr is not None else 4
    # 测试集可能 node_vec_len 不同 (ext_max_atoms=85), 检查并 pad
    test_in_channels = test_data_list[0].x.size(-1)
    print(f"  in_channels: train={in_channels}, test={test_in_channels}, edge_dim={edge_dim}", flush=True)

    model = build_pyg_model('AttentiveFP', in_channels,
                            PARAMS['hidden_dim'], PARAMS['n_conv_layers'],
                            PARAMS['p_dropout'], edge_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['learning_rate'],
                                 weight_decay=PARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PARAMS['patience'] // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    train_subset = [train_data_list[i] for i in train_idx]
    val_subset = [train_data_list[i] for i in val_idx]
    train_loader = DataLoader(train_subset, batch_size=PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=PARAMS['batch_size'], shuffle=False)

    # 训练
    best_val, best_state, pcount = float('inf'), None, 0
    t0 = time.time()
    for epoch in range(1, PARAMS['n_epochs'] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1)
            loss = loss_fn(preds, batch.y.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                vp.append(model(batch.x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1))
                vt.append(batch.y.squeeze(-1))
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
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: val_loss={vloss:.6f} (best={best_val:.6f})", flush=True)

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估 lunci6 (需要处理 in_channels 不一致的情况: pad 或重新建模型)
    model.eval()
    test_loader = DataLoader(test_data_list, batch_size=PARAMS['batch_size'], shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            x = batch.x
            # 检查 in_channels 是否匹配
            if x.size(-1) != in_channels:
                if x.size(-1) < in_channels:
                    # pad
                    pad = torch.zeros(x.size(0), in_channels - x.size(-1), device=device)
                    x = torch.cat([x, pad], dim=-1)
                else:
                    # truncate
                    x = x[..., :in_channels]
            preds.append(model(x, batch.edge_index, batch.batch, batch.edge_attr).squeeze(-1))
            trues.append(batch.y.squeeze(-1))
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()

    r2, mae, rmse = compute_metrics(true, pred)

    # 保存预测
    out_dir = os.path.join(OUTPUT_DIR, task['name'])
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'true': true, 'pred': pred}).to_csv(
        os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

    print(f"  [AttentiveFP/{task['name']}] lunci6: R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | Time={train_time:.0f}s", flush=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        'task': task['name'], 'model': 'AttentiveFP',
        'lunci6_r2': r2, 'lunci6_mae': mae, 'lunci6_rmse': rmse,
        'train_time_sec': train_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--tasks', type=str, default='all')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        print(f"\n{'#' * 60}\n# 任务: {task['name']}\n{'#' * 60}", flush=True)
        try:
            res = train_and_eval_lunci6(task, device)
            all_results.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [AttentiveFP/{task['name']}] 失败: {e}", flush=True)

    # 汇总
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'attentivefp_lunci6_summary.csv'), index=False)
        print(f"\n{'=' * 60}\n汇总\n{'=' * 60}")
        print(df.to_string(index=False))
        print(f"\n已保存: {OUTPUT_DIR}/attentivefp_lunci6_summary.csv", flush=True)


if __name__ == '__main__':
    main()
