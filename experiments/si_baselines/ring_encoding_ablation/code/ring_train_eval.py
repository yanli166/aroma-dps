"""
第三层: 三类环信息编码范式对比实验

流程:
  1. 读取第二层 GNN 基线结果, 按三任务平均 test R2 选出 top5 模型
  2. 对 top5 每个模型分别施加三种环编码:
       - label: Ring Labeling  (节点输入层, ring_flag_value=10)
       - mask:  Ring Masking   (卷积传播层, 自监督式掩码)
       - pool:  Ring Pooling   (卷积输出层, 环原子定向聚合)
  3. 对 top5 每个骨干运行 RA-GCN 组合模型 (label+mask+pool 联合)
  4. 任务: HOMA / NICS(1)zz / MBCO, 8/2 + 5折CV, R2/MAE/RMSE

自定义模型 (GNN/GIN/GAT/MPNN/GraphSAGE) 直接复用原始模型的 label/mask/pool 实现,
不改动骨干; RA-GCN 组合用 combined_ragcn.RAGCNCombined。
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, '/home/ubuntu/data_90/alldata_in_3090/model1')

from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED
from common.graph_data import load_adj_format
from unified_models.gat.model import GATModel
from unified_models.gin.model import GINModel
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from unified_models.graphsage.model import GraphSAGEModel
from ring_encoding_ablation.code.combined_ragcn import RAGCNCombined

DEFAULT_PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}
CUSTOM_MODELS = {
    'GNN': GNNModel, 'GIN': GINModel, 'GAT': GATModel,
    'MPNN': MPNNModel, 'GraphSAGE': GraphSAGEModel,
}
ENCODINGS = ['label', 'mask', 'pool']


def select_top5(gnn_results_csv):
    """从第二层结果选 top5 模型 (按三任务平均 test R2, 仅限自定义模型)"""
    df = pd.read_csv(gnn_results_csv)
    avg = df.groupby('model')['test_r2'].mean().sort_values(ascending=False)
    print(f"第二层平均 test R2 排名:\n{avg.to_string()}")
    # 仅从支持环编码的自定义模型中选取 top5
    custom_avg = avg[avg.index.isin(CUSTOM_MODELS.keys())]
    top5 = custom_avg.head(5).index.tolist()
    print(f"\n自定义模型 Top5: {top5}")
    return top5


def train_one(model, params, data, train_idx, val_idx, device,
              mode, n_epochs=200, patience=30, seed=42):
    """训练单个模型 (支持 label/mask/pool/combined)"""
    torch.manual_seed(seed)
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    mask_mats, ring_indices = data['mask_mats'], data['ring_indices']
    outputs = data['outputs']
    batch_size = params['batch_size']
    n_train, n_val = len(train_idx), len(val_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    def forward_batch(bi):
        if mode == 'label':
            return model(node_mats[bi], adj_mats[bi]).squeeze()
        elif mode == 'mask':
            return model(node_mats[bi], adj_mats[bi], mask_mats[bi]).squeeze()
        elif mode == 'pool':
            return model(node_mats[bi], adj_mats[bi], ring_indices=ring_indices[bi]).squeeze()
        elif mode == 'combined':
            return model(node_mats[bi], adj_mats[bi], mask_mats[bi], ring_indices[bi]).squeeze()

    best_val, best_state, pcount = float('inf'), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = forward_batch(bi)
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx[i:i + batch_size]
                vp.append(forward_batch(bi)); vt.append(outputs[bi])
            vp = torch.cat(vp); vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model


@torch.no_grad()
def evaluate_one(model, data, idx, batch_size, device, mode):
    model.eval()
    node_mats, adj_mats = data['node_mats'], data['adj_mats']
    mask_mats, ring_indices = data['mask_mats'], data['ring_indices']
    outputs = data['outputs']
    preds, trues = [], []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i + batch_size]
        if mode == 'label':
            p = model(node_mats[bi], adj_mats[bi]).squeeze()
        elif mode == 'mask':
            p = model(node_mats[bi], adj_mats[bi], mask_mats[bi]).squeeze()
        elif mode == 'pool':
            p = model(node_mats[bi], adj_mats[bi], ring_indices=ring_indices[bi]).squeeze()
        elif mode == 'combined':
            p = model(node_mats[bi], adj_mats[bi], mask_mats[bi], ring_indices[bi]).squeeze()
        preds.append(p); trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return compute_metrics(true, pred)


def build_model(model_name, mode, params, node_vec_len):
    """构造模型: label/mask/pool 用原始模型, combined 用 RAGCNCombined"""
    if mode == 'combined':
        return RAGCNCombined(backbone=model_name, node_vec_len=node_vec_len,
                             hidden_dim=params['hidden_dim'], n_conv=params['n_conv_layers'],
                             n_hidden=params['n_hidden_layers'], n_outputs=1,
                             p_dropout=params['p_dropout'], n_heads=4)
    kwargs = dict(node_vec_len=node_vec_len, hidden_dim=params['hidden_dim'],
                  n_conv=params['n_conv_layers'], n_hidden=params['n_hidden_layers'],
                  n_outputs=1, p_dropout=params['p_dropout'], mode=mode)
    if model_name == 'GAT':
        kwargs['n_heads'] = 4
    return CUSTOM_MODELS[model_name](**kwargs)


def run_experiment(model_name, encoding, task, params, device, output_root,
                   n_epochs=200, patience=30, seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    name = task['name']
    tag = f"{model_name}_{encoding}"
    out_dir = os.path.join(output_root, name, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n任务: {name} | 模型: {model_name} | 编码: {encoding}\n{'='*60}")

    # ring_flag_value: label/pool/combined 用 10 (注入目标环标记); mask 用 10 (mask 也基于环)
    data = load_adj_format(task['dataset_path'], task['target_col'], 60, 75,
                           ring_flag_value=10, device=device)
    n = data['n']
    splits = canonical_splits(n, seed=seed)
    test_idx, cv_folds, final_tr, final_va = splits
    test_idx = torch.tensor(test_idx, device=device)
    final_tr = torch.tensor(final_tr, device=device)
    final_va = torch.tensor(final_va, device=device)

    cv_rows = []
    for fold, (tr, va) in enumerate(cv_folds):
        tr_t = torch.tensor(tr, device=device); va_t = torch.tensor(va, device=device)
        model = build_model(model_name, encoding, params, data['node_vec_len']).to(device)
        t0 = time.time()
        model = train_one(model, params, data, tr_t, va_t, device, encoding,
                          n_epochs, patience, seed + fold)
        r2, mae, rmse = evaluate_one(model, data, va_t, params['batch_size'], device, encoding)
        print(f"  Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} ({time.time()-t0:.0f}s)")
        cv_rows.append({'fold': fold+1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        del model; torch.cuda.empty_cache()

    final_model = build_model(model_name, encoding, params, data['node_vec_len']).to(device)
    final_model = train_one(final_model, params, data, final_tr, final_va, device, encoding,
                            n_epochs, patience, seed)
    tr_r2, tr_mae, tr_rmse = evaluate_one(final_model, data, final_tr, params['batch_size'], device, encoding)
    te_r2, te_mae, te_rmse = evaluate_one(final_model, data, test_idx, params['batch_size'], device, encoding)
    torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
    print(f"  [{tag}] CV: R2={np.mean([r['r2'] for r in cv_rows]):.4f} | "
          f"Test: R2={te_r2:.4f} MAE={te_mae:.4f} RMSE={te_rmse:.4f}")
    del final_model; torch.cuda.empty_cache()

    cv_r2 = np.mean([r['r2'] for r in cv_rows]); cv_r2_std = np.std([r['r2'] for r in cv_rows])
    cv_mae = np.mean([r['mae'] for r in cv_rows]); cv_rmse = np.mean([r['rmse'] for r in cv_rows])
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['encoding', encoding]); w.writerow(['task', name]); w.writerow(['n', n])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', tr_r2]); w.writerow(['train_mae', tr_mae]); w.writerow(['train_rmse', tr_rmse])
        w.writerow(['test_r2', te_r2]); w.writerow(['test_mae', te_mae]); w.writerow(['test_rmse', te_rmse])
    return {'model': model_name, 'encoding': encoding, 'task': name, 'n': n,
            'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'test_r2': te_r2, 'test_mae': te_mae, 'test_rmse': te_rmse}


def main():
    parser = argparse.ArgumentParser(description='第三层:环编码范式对比')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/aroma-dps-code/ring_encoding_ablation/results')
    parser.add_argument('--gnn_results', type=str,
                        default='/home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv',
                        help='第二层结果文件 (用于选 top5)')
    parser.add_argument('--top5', type=str, default='',
                        help='手动指定 top5 模型 (逗号分隔), 留空则自动从 gnn_results 选取')
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PARAMS['patience'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    params = dict(DEFAULT_PARAMS); params['n_epochs'] = args.n_epochs; params['patience'] = args.patience
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    # 选 top5
    if args.top5:
        top5 = [m for m in args.top5.split(',') if m]
    elif os.path.exists(args.gnn_results):
        top5 = select_top5(args.gnn_results)
    else:
        # 默认全部自定义模型 (若第二层结果不存在)
        top5 = list(CUSTOM_MODELS.keys())
        print(f"未找到 {args.gnn_results}, 使用默认 top5: {top5}")
    # 仅保留支持的自定义模型 (PyG 模型环编码需额外实现, 此处聚焦自定义骨干)
    top5 = [m for m in top5 if m in CUSTOM_MODELS]
    print(f"参与第三层实验的模型: {top5}")

    all_results = []
    # 三种编码 + combined
    for model_name in top5:
        for encoding in ENCODINGS + ['combined']:
            for task in task_list:
                try:
                    res = run_experiment(model_name, encoding, task, params, device,
                                         args.output_dir, args.n_epochs, args.patience, args.seed)
                    all_results.append(res)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"[{model_name}|{encoding}|{task['name']}] 失败: {e}")

    cols = ['task', 'model', 'encoding', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
            'test_r2', 'test_mae', 'test_rmse']
    df_out = pd.DataFrame(all_results)[cols]
    print(f"\n{'='*60}\n第三层环编码消融汇总\n{'='*60}")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(args.output_dir, 'all_ring_encoding_summary.csv'), index=False)


if __name__ == '__main__':
    main()
