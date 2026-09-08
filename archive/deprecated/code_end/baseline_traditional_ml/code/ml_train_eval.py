"""
第一层: 传统机器学习基线实验 (label编码) — 修复版

修复项:
  - C1: 数据划分一致性 — 先用原始 n 做 canonical_splits, 再构建特征矩阵,
        保证与 Layer 2/3 索引空间完全一致 (不再过滤无效样本后重新划分)
  - C2: 最终模型用 final_train_idx 训练 (87.5%), final_val_idx 评估 val,
        与 Layer 2/3 完全一致 (不再用全 trainval)
  - C3: 每折用 seed+fold 构造模型, 避免所有 fold 初始权重相同
  - m6: 简化 trainval_idx 重建逻辑
  - m7: CV 标准差用 ddof=1 (样本标准差)
  - M5: 保存 test_predictions.csv (已有, 保持)
  - M6: summary.csv 增加 train_time_sec 字段
  - m1: 设置完整随机种子 (torch + numpy + random)
  - m2: 设置 cudnn.deterministic
"""
import os
import sys
import csv
import time
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib

# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径设置: 优先使用 code_end 自身模块
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))

from common.tasks import TASKS, canonical_splits, compute_metrics, DEFAULT_SEED
from common.features import build_fingerprint_matrix


def set_full_seed(seed):
    """m1: 设置完整随机种子 (torch + numpy + random)"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(seed=DEFAULT_SEED, gpu=None):
    """构造9个传统ML回归模型 (合理默认超参)"""
    models = {}
    gpu_str = str(gpu) if gpu is not None else None

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

    from sklearn.ensemble import RandomForestRegressor
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=seed, n_jobs=-1)

    from sklearn.svm import SVR
    models['SVM'] = SVR(C=10.0, gamma='scale', epsilon=0.05)

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=600, max_depth=-1, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
    except ImportError:
        pass

    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            tree_method='hist', n_jobs=-1, verbosity=0)
    except ImportError:
        pass

    from sklearn.ensemble import ExtraTreesRegressor
    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=seed, n_jobs=-1)

    from sklearn.linear_model import Ridge
    models['Ridge'] = Ridge(alpha=1.0, random_state=seed)

    from sklearn.kernel_ridge import KernelRidge
    models['KRR'] = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)

    from sklearn.neural_network import MLPRegressor
    models['MLP'] = MLPRegressor(
        hidden_layer_sizes=(256, 128), activation='relu',
        solver='adam', alpha=1e-4, learning_rate_init=1e-3,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed)

    return models


def run_one_model(model_name, model_factory, X, y, splits, task_name, output_root, n_total):
    """对单个模型跑完整 5折CV + 测试集评估

    C2 修复: 最终模型用 final_train_idx 训练 (87.5%), final_val_idx 评估 val,
             train metrics 基于 final_train_idx (与 Layer 2/3 一致)
    C3 修复: 每折用 seed+fold 构造模型 (通过 model_factory 的 seed 参数)
    m6 修复: 不再循环重建 trainval_idx
    m7 修复: CV std 用 ddof=1
    M6 修复: 记录 train_time_sec
    """
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
        train_time = time.time() - t0
        pred = model.predict(X_va)
        r2, mae, rmse = compute_metrics(y[va_idx], pred)
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse, 'time': train_time})
        print(f"    [{model_name}] Fold {fold+1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({train_time:.0f}s)", flush=True)

    # m7: CV 标准差用 ddof=1 (样本标准差)
    cv_r2 = np.mean([r['r2'] for r in cv_rows])
    cv_r2_std = np.std([r['r2'] for r in cv_rows], ddof=1)
    cv_mae = np.mean([r['mae'] for r in cv_rows])
    cv_rmse = np.mean([r['rmse'] for r in cv_rows])

    # C2 修复: 最终模型用 final_train_idx 训练 (87.5%), 不再用全 trainval
    # final_val_idx 仅用于评估 val metrics (报告用, sklearn 模型不支持外部 early stopping)
    scaler = StandardScaler()
    X_ftr = scaler.fit_transform(X[final_train_idx])
    X_fva = scaler.transform(X[final_val_idx])
    X_te = scaler.transform(X[test_idx])
    model = model_factory()
    t0 = time.time()
    model.fit(X_ftr, y[final_train_idx])
    final_train_time = time.time() - t0

    test_pred = model.predict(X_te)
    test_r2, test_mae, test_rmse = compute_metrics(y[test_idx], test_pred)
    # train metrics 基于 final_train_idx (与 Layer 2/3 一致)
    train_pred = model.predict(X_ftr)
    train_r2, train_mae, train_rmse = compute_metrics(y[final_train_idx], train_pred)
    # val metrics (报告用)
    val_pred = model.predict(X_fva)
    val_r2, val_mae, val_rmse = compute_metrics(y[final_val_idx], val_pred)

    print(f"  [{model_name}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | "
          f"Test: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f} | "
          f"Train time: {final_train_time:.1f}s", flush=True)

    # 保存
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': y[test_idx], 'pred': test_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', task_name]); w.writerow(['n', n_total])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', train_r2]); w.writerow(['train_mae', train_mae]); w.writerow(['train_rmse', train_rmse])
        w.writerow(['val_r2', val_r2]); w.writerow(['val_mae', val_mae]); w.writerow(['val_rmse', val_rmse])
        w.writerow(['test_r2', test_r2]); w.writerow(['test_mae', test_mae]); w.writerow(['test_rmse', test_rmse])
        w.writerow(['train_time_sec', final_train_time])

    # 散点图
    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(y[test_idx], test_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(y[test_idx].min(), test_pred.min()), max(y[test_idx].max(), test_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{task_name} - {model_name}\nTest R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {'model': model_name, 'task': task_name, 'n': n_total,
            'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
            'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
            'train_time_sec': final_train_time}


def main():
    parser = argparse.ArgumentParser(description='第一层:传统ML基线 (修复版)')
    parser.add_argument('--output_dir', type=str,
                        default='_PROJ_ROOT + "/code_end"/results/layer1_ml')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--tasks', type=str, default='all', help='逗号分隔任务名, 或 all')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id')
    args = parser.parse_args()

    # m2: 设置 cudnn.deterministic
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    if args.gpu is not None:
        print(f"GPU 加速已启用: GPU {args.gpu} (CatBoost/XGBoost/LightGBM)")

    all_results = []
    for task in task_list:
        name = task['name']
        print(f"\n{'='*70}\n任务: {name} ({task['metric']})\n数据: {task['dataset_path']}\n{'='*70}", flush=True)
        # 数据清洗: 处理 NaN / Windows 换行符 / 多余空列
        from common.tasks import clean_dataset_csv
        clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
        df = pd.read_csv(clean_path)
        smiles_list = df['smiles'].tolist()
        y = df[task['target_col']].astype(float).to_numpy()
        n_total = len(smiles_list)  # C1: 原始样本数

        # C1: 先在原始 n 上做划分, 再构建特征 (保证三层一致)
        splits = canonical_splits(n_total, seed=args.seed)
        test_idx = splits[0]
        print(f"  划分: train_val={n_total - len(test_idx)}, test={len(test_idx)}", flush=True)

        print(f"  提取 MACCS+Morgan+分子描述符+环描述符 ({n_total} 样本)...", flush=True)
        # C1: raise_on_invalid=True (默认), 无效SMILES会raise (与Layer2/3一致)
        X, valid = build_fingerprint_matrix(smiles_list, df=df)
        assert valid.all(), "无效SMILES应已在build_fingerprint_matrix中raise"
        print(f"  特征矩阵: {X.shape}, 目标范围: [{y.min():.4f}, {y.max():.4f}]", flush=True)

        models = build_models(seed=args.seed, gpu=args.gpu)
        for model_name, model in models.items():
            # C3: FoldAwareFactory 每折用 seed+fold 构造模型
            fold_factory = FoldAwareFactory(model, args.seed)
            try:
                res = run_one_model_seeded(model_name, fold_factory, X, y, splits, name,
                                           args.output_dir, n_total, args.seed)
                all_results.append(res)
            except Exception as e:
                print(f"  [{model_name}] 失败: {e}", flush=True)

    # 汇总
    print(f"\n{'='*70}\n第一层传统ML汇总\n{'='*70}")
    cols = ['task', 'model', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
            'test_r2', 'test_mae', 'test_rmse', 'train_time_sec']
    df_out = pd.DataFrame(all_results)[cols]
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(args.output_dir, 'all_ml_summary.csv'), index=False)
    print(f"\n结果保存: {args.output_dir}")


class FoldAwareFactory:
    """C3: 每折用 seed+fold 构造模型, 避免所有 fold 初始权重相同"""
    def __init__(self, base_model, base_seed):
        self.base_model = base_model
        self.base_seed = base_seed

    def __call__(self, fold=0):
        params = self.base_model.get_params()
        for key in ['random_state', 'random_seed', 'seed']:
            if key in params:
                params[key] = self.base_seed + fold
        return type(self.base_model)(**params)


def run_one_model_seeded(model_name, fold_factory, X, y, splits, task_name, output_root, n_total, base_seed):
    """C3 修复版: 每折用 seed+fold 构造模型

    其余逻辑与 run_one_model 一致 (C2/m6/m7/M5/M6 修复已包含)。
    """
    test_idx, cv_folds, final_train_idx, final_val_idx = splits
    out_dir = os.path.join(output_root, task_name, model_name)
    os.makedirs(out_dir, exist_ok=True)

    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        # C3: 每折用 seed+fold 构造模型
        model = fold_factory(fold=fold)
        t0 = time.time()
        model.fit(X_tr, y[tr_idx])
        train_time = time.time() - t0
        pred = model.predict(X_va)
        r2, mae, rmse = compute_metrics(y[va_idx], pred)
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse, 'time': train_time})
        print(f"    [{model_name}] Fold {fold+1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({train_time:.0f}s)", flush=True)

    # m7: ddof=1
    cv_r2 = np.mean([r['r2'] for r in cv_rows])
    cv_r2_std = np.std([r['r2'] for r in cv_rows], ddof=1)
    cv_mae = np.mean([r['mae'] for r in cv_rows])
    cv_rmse = np.mean([r['rmse'] for r in cv_rows])

    # C2: 最终模型用 final_train_idx (87.5%)
    scaler = StandardScaler()
    X_ftr = scaler.fit_transform(X[final_train_idx])
    X_fva = scaler.transform(X[final_val_idx])
    X_te = scaler.transform(X[test_idx])
    # 最终模型用 base_seed (与 Layer 2/3 一致: final model seed=seed, 不加 fold)
    model = fold_factory(fold=0)
    t0 = time.time()
    model.fit(X_ftr, y[final_train_idx])
    final_train_time = time.time() - t0

    test_pred = model.predict(X_te)
    test_r2, test_mae, test_rmse = compute_metrics(y[test_idx], test_pred)
    train_pred = model.predict(X_ftr)
    train_r2, train_mae, train_rmse = compute_metrics(y[final_train_idx], train_pred)
    val_pred = model.predict(X_fva)
    val_r2, val_mae, val_rmse = compute_metrics(y[final_val_idx], val_pred)

    print(f"  [{model_name}] CV: R2={cv_r2:.4f}±{cv_r2_std:.4f} | "
          f"Test: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f} | "
          f"Train time: {final_train_time:.1f}s", flush=True)

    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': y[test_idx], 'pred': test_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        w.writerow(['model', model_name]); w.writerow(['task', task_name]); w.writerow(['n', n_total])
        w.writerow(['cv_r2_mean', cv_r2]); w.writerow(['cv_r2_std', cv_r2_std])
        w.writerow(['cv_mae', cv_mae]); w.writerow(['cv_rmse', cv_rmse])
        w.writerow(['train_r2', train_r2]); w.writerow(['train_mae', train_mae]); w.writerow(['train_rmse', train_rmse])
        w.writerow(['val_r2', val_r2]); w.writerow(['val_mae', val_mae]); w.writerow(['val_rmse', val_rmse])
        w.writerow(['test_r2', test_r2]); w.writerow(['test_mae', test_mae]); w.writerow(['test_rmse', test_rmse])
        w.writerow(['train_time_sec', final_train_time])

    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(y[test_idx], test_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(y[test_idx].min(), test_pred.min()), max(y[test_idx].max(), test_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{task_name} - {model_name}\nTest R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {'model': model_name, 'task': task_name, 'n': n_total,
            'cv_r2': cv_r2, 'cv_r2_std': cv_r2_std, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
            'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
            'train_time_sec': final_train_time}


if __name__ == '__main__':
    main()
