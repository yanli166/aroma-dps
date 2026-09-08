"""
第一层: 传统机器学习基线实验 (label编码)

模型: CatBoost, Random Forest, SVM, LightGBM, XGBoost, Extra Trees,
      Ridge Regression, Kernel Ridge Regression (KRR), MLP(浅层2-3层)
输入特征: MACCS(167) + Morgan(2048) + 分子描述符(16) + 环描述符(13) = 2244维
  - 环描述符基于 atom_on_ring 列提取目标环信息 (label编码在ML中的体现)
  - 与GNN中 ring_flag_value=10 在节点特征中标记目标环原子的语义一致
任务: HOMA / NICS(1)zz / MBCO
评估: 与GNN完全一致的 8/2划分 + 5折CV, 指标 R2/MAE/RMSE

复用 common.tasks 的规范划分, 保证与二、三层可比。
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径设置
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED
from common.features import build_fingerprint_matrix


def build_models(seed=DEFAULT_SEED, gpu=None):
    """构造9个传统ML回归模型 (合理默认超参)

    Args:
        seed: 随机种子
        gpu: GPU id (int), None 则全部 CPU。CatBoost/XGBoost/LightGBM 支持 GPU 加速。
    """
    models = {}
    gpu_str = str(gpu) if gpu is not None else None

    # CatBoost (GPU)
    try:
        from catboost import CatBoostRegressor
        cb_kwargs = dict(
            iterations=800, depth=8, learning_rate=0.05,
            loss_function='RMSE', random_seed=seed, verbose=0, allow_writing_files=False)
        if gpu is not None:
            cb_kwargs.update(task_type='GPU', devices=gpu_str)
        models['CatBoost'] = CatBoostRegressor(**cb_kwargs)
    except ImportError:
        pass

    # Random Forest (sklearn, 无GPU支持)
    from sklearn.ensemble import RandomForestRegressor
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=seed, n_jobs=-1)

    # SVM (sklearn, 无GPU支持)
    from sklearn.svm import SVR
    models['SVM'] = SVR(C=10.0, gamma='scale', epsilon=0.05)

    # LightGBM (CPU - n_jobs=1 避免与CatBoost的OpenMP冲突导致死锁)
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=600, max_depth=-1, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
    except ImportError:
        pass

    # XGBoost (CPU hist - GPU 在多模型并发时可能触发 cudaErrorIllegalAddress)
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            tree_method='hist', n_jobs=-1, verbosity=0)
    except ImportError:
        pass

    # Extra Trees (sklearn, 无GPU支持)
    from sklearn.ensemble import ExtraTreesRegressor
    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=seed, n_jobs=-1)

    # Ridge Regression (sklearn, 无GPU支持)
    from sklearn.linear_model import Ridge
    models['Ridge'] = Ridge(alpha=1.0, random_state=seed)

    # Kernel Ridge Regression (sklearn, 无GPU支持)
    from sklearn.kernel_ridge import KernelRidge
    models['KRR'] = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)

    # MLP (浅层 2-3 层, sklearn, 无GPU支持)
    from sklearn.neural_network import MLPRegressor
    models['MLP'] = MLPRegressor(
        hidden_layer_sizes=(256, 128), activation='relu',
        solver='adam', alpha=1e-4, learning_rate_init=1e-3,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed)

    return models


def run_one_model(model_name, model_factory, X, y, splits, task_name, output_root):
    """对单个模型跑完整 5折CV + 测试集评估"""
    test_idx, cv_folds, final_train_idx, final_val_idx = splits
    out_dir = os.path.join(output_root, task_name, model_name)
    os.makedirs(out_dir, exist_ok=True)

    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        model = model_factory()  # 重建保证独立
        t0 = time.time()
        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_va)
        r2, mae, rmse = compute_metrics(y[va_idx], pred)
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse, 'time': time.time() - t0})
        print(f"    [{model_name}] Fold {fold+1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({time.time()-t0:.0f}s)", flush=True)

    cv_r2 = np.mean([r['r2'] for r in cv_rows])
    cv_r2_std = np.std([r['r2'] for r in cv_rows])
    cv_mae = np.mean([r['mae'] for r in cv_rows])
    cv_rmse = np.mean([r['rmse'] for r in cv_rows])

    # 最终模型: 全 train_val 训练 -> 测试集评估
    trainval_idx = np.concatenate([cv_folds[0][0], cv_folds[0][1]])
    for tr, va in cv_folds[1:]:
        trainval_idx = np.unique(np.concatenate([trainval_idx, tr, va]))
    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X[trainval_idx])
    X_te = scaler.transform(X[test_idx])
    model = model_factory()
    model.fit(X_tv, y[trainval_idx])
    test_pred = model.predict(X_te)
    test_r2, test_mae, test_rmse = compute_metrics(y[test_idx], test_pred)
    train_pred = model.predict(X_tv)
    train_r2, train_mae, train_rmse = compute_metrics(y[trainval_idx], train_pred)

    print(f"  [{model_name}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | "
          f"Test: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}", flush=True)

    # 保存
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': y[test_idx], 'pred': test_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', task_name])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', train_r2]); w.writerow(['train_mae', train_mae]); w.writerow(['train_rmse', train_rmse])
        w.writerow(['test_r2', test_r2]); w.writerow(['test_mae', test_mae]); w.writerow(['test_rmse', test_rmse])

    # 散点图
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(y[test_idx], test_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(y[test_idx].min(), test_pred.min()), max(y[test_idx].max(), test_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{task_name} - {model_name}\nTest R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {'model': model_name, 'task': task_name, 'n': len(y),
            'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse}


def main():
    parser = argparse.ArgumentParser(description='第一层:传统ML基线 (MACCS+Morgan指纹)')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--tasks', type=str, default='all', help='逗号分隔任务名, 或 all')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id (CatBoost/XGBoost/LightGBM 加速)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    if args.gpu is not None:
        print(f"GPU 加速已启用: GPU {args.gpu} (CatBoost/XGBoost/LightGBM)")

    all_results = []
    for task in task_list:
        name = task['name']
        print(f"\n{'='*70}\n任务: {name} ({task['metric']})\n数据: {task['dataset_path']}\n{'='*70}", flush=True)
        df = pd.read_csv(task['dataset_path'])
        # 解析 atom_on_ring (保留以备扩展, 指纹只用 smiles)
        smiles_list = df['smiles'].tolist()
        y = df[task['target_col']].astype(float).to_numpy()

        print(f"  提取 MACCS+Morgan+分子描述符+环描述符(label编码) ({len(smiles_list)} 样本)...", flush=True)
        X, valid = build_fingerprint_matrix(smiles_list, df=df)
        # 过滤无效指纹
        if not valid.all():
            print(f"  过滤 {len(valid)-valid.sum()} 个无效样本")
            X, y = X[valid], y[valid]
        print(f"  特征矩阵: {X.shape} (MACCS+Morgan+MolDesc+RingDesc), 目标范围: [{y.min():.4f}, {y.max():.4f}]", flush=True)

        splits = canonical_splits(len(y), seed=args.seed)
        test_idx = splits[0]
        print(f"  划分: train_val={len(y)-len(test_idx)}, test={len(test_idx)}", flush=True)

        models = build_models(seed=args.seed, gpu=args.gpu)
        for model_name, model in models.items():
            # 用工厂函数确保每折独立重建
            factory = (lambda m=model: type(m)(**m.get_params()))
            # catboost/xgb 需特殊处理 get_params
            try:
                res = run_one_model(model_name, factory, X, y, splits, name, args.output_dir)
                all_results.append(res)
            except Exception as e:
                print(f"  [{model_name}] 失败: {e}", flush=True)

    # 汇总
    print(f"\n{'='*70}\n第一层传统ML汇总\n{'='*70}")
    cols = ['task', 'model', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
            'test_r2', 'test_mae', 'test_rmse']
    df_out = pd.DataFrame(all_results)[cols]
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(args.output_dir, 'all_ml_summary.csv'), index=False)
    print(f"\n结果保存: {args.output_dir}")


if __name__ == '__main__':
    main()
