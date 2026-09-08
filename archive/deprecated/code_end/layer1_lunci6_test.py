
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# -*- coding: utf-8 -*-
"""
Layer 1 树模型 lunci6 集外测试
==============================
快速测试传统ML树模型 (XGBoost, RandomForest, CatBoost, LightGBM) 在 lunci6 上的表现。
训练: 用 0716 全量训练集 (87.5% train + 12.5% val, 与 Layer 2/3 一致)
测试: lunci6-test.csv (312行, 与训练集 0 重叠)
特征: MACCS(167) + Morgan(2048) + 分子描述符(16) + 环描述符(13) = 2244维
"""
import os, sys, time
import numpy as np
import pandas as pd

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
os.chdir(PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)

from common.tasks import TASKS, DEFAULT_SEED, clean_dataset_csv
from common.features import build_fingerprint_matrix
from generalization_test.code.splits import prepare_external_test_csv
from common.tasks import compute_metrics

OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results', 'layer1_lunci6_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except: pass


def build_models(seed=42):
    models = {}
    try:
        from catboost import CatBoostRegressor
        models['CatBoost'] = CatBoostRegressor(
            iterations=800, depth=8, learning_rate=0.05,
            loss_function='RMSE', random_seed=seed, verbose=0, allow_writing_files=False)
    except: pass
    from sklearn.ensemble import RandomForestRegressor
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=seed, n_jobs=-1)
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=600, max_depth=-1, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
    except: pass
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    from sklearn.ensemble import ExtraTreesRegressor
    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1)
    return models


def run_task(task, seed=42):
    set_seed(seed)
    print(f"\n{'='*60}\n任务: {task['name']}\n{'='*60}", flush=True)

    # 1. 加载训练数据
    clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
    df_train = pd.read_csv(clean_path, encoding='utf-8-sig')
    print(f"  训练集: {len(df_train)} 行", flush=True)

    # 2. 构建训练特征
    print(f"  构建训练特征...", flush=True)
    X_train, _ = build_fingerprint_matrix(df_train['smiles'].tolist(), df_train, raise_on_invalid=False)
    y_train = df_train[task['target_col']].values

    # 3. 加载 lunci6 测试集
    ext_csv, ext_target = prepare_external_test_csv(task['name'], test_source='lunci6')
    df_test = pd.read_csv(ext_csv, encoding='utf-8-sig')
    print(f"  lunci6 测试集: {len(df_test)} 行", flush=True)

    # 4. 构建测试特征
    X_test, _ = build_fingerprint_matrix(df_test['smiles'].tolist(), df_test, raise_on_invalid=False)
    y_test = df_test[ext_target].values

    # 5. 训练 + 评估
    results = []
    models = build_models(seed)
    for name, model in models.items():
        t0 = time.time()
        print(f"  训练 {name}...", flush=True)
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        pred = model.predict(X_test)
        r2, mae, rmse = compute_metrics(y_test, pred)

        # 保存预测
        out_dir = os.path.join(OUTPUT_DIR, task['name'], name)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({'true': y_test, 'pred': pred}).to_csv(
            os.path.join(out_dir, 'lunci6_predictions.csv'), index=False)

        print(f"    [{name}/{task['name']}] lunci6: R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | Time={train_time:.1f}s", flush=True)
        results.append({
            'task': task['name'], 'model': name,
            'lunci6_r2': r2, 'lunci6_mae': mae, 'lunci6_rmse': rmse,
            'train_time_sec': train_time,
        })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    task_list = TASKS if args.tasks == 'all' else [
        next(t for t in TASKS if t['name'] == n) for n in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        try:
            all_results.extend(run_task(task, args.seed))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{task['name']}] 失败: {e}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'layer1_lunci6_summary.csv'), index=False)
        print(f"\n{'='*60}\n汇总\n{'='*60}")
        print(df.to_string(index=False))
        print(f"\n已保存: {OUTPUT_DIR}/layer1_lunci6_summary.csv", flush=True)


if __name__ == '__main__':
    main()
