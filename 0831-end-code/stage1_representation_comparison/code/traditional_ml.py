"""
Stage 1: 传统机器学习基线 (固定分子表示) — [0831 重构]

对比固定分子表示 (传统ML) 与可学习分子图表示 (GNN)。

协议要点 (PROTOCOL_SPEC / 参考思路2.txt P1 #9):
  - 数据划分: 用 common.protocol.get_final_splits (split_seed=2026),
    model_seed 绝不进入 split; 同 task 下不同 model seed 的 test 逐字节相同。
  - 分组: make_group_ids(df['smiles']) (canonical SMILES), 启动 assert_no_leak。
  - Train-only scaling: 特征拆成 X_binary(MACCS+Morgan) 与 X_continuous(描述符),
    SVM/KRR/MLP 只对 continuous 特征在"当前训练折"上 fit scaler, 不使用全数据统计量。
  - 模型选择只看 cv_mae (val-MAE); final test 绝不用于选择。
  - 输出: cv_results.csv / final_test_results.csv / per_seed_results.csv (含 run_status)。

Dry-run: model_seed=11, 只跑 SVM/MLP (2 个 scale-sensitive 模型), 3 任务。
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LAST_END_ROOT not in sys.path:
    sys.path.insert(0, LAST_END_ROOT)

from common.tasks import TASKS, get_task, compute_metrics, clean_dataset_csv
from common.features import build_fingerprint_matrix, MACCS_DIM, MORGAN_BITS
from common.train_eval import set_full_seed
from common.protocol import get_final_splits, make_group_ids, assert_no_leak, \
    completeness_check, SPLIT_SEED, MODEL_SEEDS
from common.constants import RESULTS_V2_DIR

# 特征拆分: binary = MACCS(167)+Morgan(2048)=2215; continuous = 分子/环描述符(16+13)
CONTINUOUS_START = MACCS_DIM + MORGAN_BITS  # 2215
FINAL_SPLIT_VAL_RATIO = 0.125  # dev 内部 final 87.5% train / 12.5% val
STAGE1_DIR = os.path.join(RESULTS_V2_DIR, 'stage1', 'ml')


def build_models(seed=11):
    """构造 dry-run 传统 ML 模型 (scale-sensitive, 需 train-only scaling)。"""
    models = {}
    from sklearn.svm import SVR
    models['SVM'] = SVR(C=10.0, gamma='scale', epsilon=0.05)
    from sklearn.neural_network import MLPRegressor
    models['MLP'] = MLPRegressor(
        hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
        alpha=1e-4, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    return models


def final_train_val_split(dev_idx, groups, split_seed=SPLIT_SEED):
    """dev(80%) 内部 group-aware 87.5/12.5 final 划分, 用 split_seed (不掺模型 seed)。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=FINAL_SPLIT_VAL_RATIO, random_state=split_seed)
    tr_rel, va_rel = next(gss.split(np.arange(len(dev_idx)), groups=np.asarray(groups)[dev_idx]))
    return dev_idx[tr_rel], dev_idx[va_rel]


def train_only_scale(X_bin, X_cont, tr_idx, other_idx):
    """train-only scaling: 仅用当前训练折 计算 scaler。"""
    scaler = StandardScaler()
    Xc_tr = scaler.fit_transform(X_cont[tr_idx])
    out = {'scaled': np.hstack([X_bin[tr_idx], Xc_tr])}
    for name, idx in other_idx.items():
        Xc_i = scaler.transform(X_cont[idx])
        out[name] = np.hstack([X_bin[idx], Xc_i])
    return out


class FoldAwareFactory:
    """每折用 model_seed+fold 构造模型 (模型 seed 不进 split)。"""
    def __init__(self, base_model, base_seed):
        self.base_model = base_model
        self.base_seed = base_seed

    def __call__(self, fold=0):
        params = self.base_model.get_params()
        for key in ['random_state', 'random_seed', 'seed']:
            if key in params:
                params[key] = self.base_seed + fold
        return type(self.base_model)(**params)


def run_one_model(model_name, factory, X, y, splits, task_name, output_root,
                  n_total, base_seed, groups):
    """跑完整 protocol 流程: 5折CV + final(dev内87.5/12.5) + test。"""
    dev_idx = np.asarray(splits['train_idx'])
    test_idx = np.asarray(splits['test_idx'])
    cv_folds = splits['folds']

    X_bin = X[:, :CONTINUOUS_START]
    X_cont = X[:, CONTINUOUS_START:]

    out_dir = os.path.join(output_root, task_name, model_name)
    os.makedirs(out_dir, exist_ok=True)

    cv_rows = []
    for fold, (tr, va) in enumerate(cv_folds):
        prep = train_only_scale(X_bin, X_cont, tr, {'va': va})  # scaler 只 fit 训练折
        model = factory(fold=fold)
        t0 = time.time()
        model.fit(prep['scaled'], y[tr])
        train_time = time.time() - t0
        pred = model.predict(prep['va'])
        r2, mae, rmse = compute_metrics(y[va], pred)
        cv_rows.append({'fold': fold + 1, 'r2': r2, 'mae': mae, 'rmse': rmse, 'time': train_time})
        print(f"    [{task_name}|{model_name}] Fold {fold+1}: R2={r2:.4f} MAE={mae:.4f} ({train_time:.0f}s)", flush=True)

    cv_r2 = float(np.mean([r['r2'] for r in cv_rows]))
    cv_mae = float(np.mean([r['mae'] for r in cv_rows]))
    cv_rmse = float(np.mean([r['rmse'] for r in cv_rows]))

    # final model: dev 内 87.5% train / 12.5% val (final 用 model_seed, 不加 fold)
    ft, fv = final_train_val_split(dev_idx, groups)
    assert_no_leak(ft, fv, groups, f"{task_name}/{model_name} final-train-val")
    fp = train_only_scale(X_bin, X_cont, ft, {'fv': fv, 'te': test_idx})
    model = factory(fold=0)
    t0 = time.time()
    model.fit(fp['scaled'], y[ft])
    final_train_time = time.time() - t0
    test_pred = model.predict(fp['te'])
    train_pred = model.predict(fp['scaled'])
    test_r2, test_mae, test_rmse = compute_metrics(y[test_idx], test_pred)
    train_r2, train_mae, train_rmse = compute_metrics(y[ft], train_pred)
    val_pred = model.predict(fp['fv'])
    val_r2, val_mae, val_rmse = compute_metrics(y[fv], val_pred)

    print(f"  [{task_name}|{model_name}] CV: R2={cv_r2:.4f} | "
          f"Test: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f} | Time: {final_train_time:.0f}s", flush=True)

    # 保存
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    pd.DataFrame({'true': y[test_idx], 'pred': test_pred}).to_csv(
        os.path.join(out_dir, 'test_predictions.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value'])
        for k, v in [('model', model_name), ('task', task_name), ('n', n_total),
                     ('cv_r2', cv_r2), ('cv_mae', cv_mae), ('cv_rmse', cv_rmse),
                     ('train_r2', train_r2), ('train_mae', train_mae), ('train_rmse', train_rmse),
                     ('val_r2', val_r2), ('val_mae', val_mae), ('val_rmse', val_rmse),
                     ('test_r2', test_r2), ('test_mae', test_mae), ('test_rmse', test_rmse),
                     ('train_time_sec', final_train_time)]:
            w.writerow([k, v])

    plt.figure(figsize=(7, 7), dpi=120)
    plt.scatter(y[test_idx], test_pred, alpha=0.4, s=10, c='steelblue')
    lims = [min(y[test_idx].min(), test_pred.min()), max(y[test_idx].max(), test_pred.max())]
    plt.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{task_name} - {model_name}\nTest R2={test_r2:.4f}, MAE={test_mae:.4f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), bbox_inches='tight'); plt.close()

    return {'seed': base_seed, 'task': task_name, 'model': model_name, 'config': 'standard',
            'n': n_total, 'cv_r2': cv_r2, 'cv_mae': cv_mae, 'cv_rmse': cv_rmse,
            'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
            'train_time_sec': final_train_time, 'run_status': 'OK', 'error_message': ''}


def main():
    parser = argparse.ArgumentParser(description='Stage 1: 传统ML基线 (train-only scaling)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=int(MODEL_SEEDS[0]))
    parser.add_argument('--models', type=str, default='SVM,MLP', help='逗号分隔: SVM,MLP,KRR')
    parser.add_argument('--tasks', type=str, default='all', help='逗号分隔任务名, 或 all')
    args = parser.parse_args()

    output_root = args.output_dir or STAGE1_DIR
    os.makedirs(output_root, exist_ok=True)
    model_seed = args.seed
    set_full_seed(model_seed)
    model_names = args.models.split(',') if args.models != 'all' else list(build_models().keys())
    task_list = TASKS if args.tasks == 'all' else [get_task(t) for t in args.tasks.split(',')]

    all_results = []
    for task in task_list:
        name = task['name']
        print(f"\n{'='*70}\n任务: {name}\n{'='*70}", flush=True)
        clean_path = clean_dataset_csv(task['dataset_path'], task['target_col'])
        df = pd.read_csv(clean_path)
        smiles_list = df['smiles'].tolist()
        y = df[task['target_col']].astype(float).to_numpy()
        n_total = len(smiles_list)

        groups = make_group_ids(smiles_list)  # canonical SMILES 分组
        splits = get_final_splits(n_total, groups, split_seed=SPLIT_SEED)
        # 铁律校验: dev/test 与每折 train/val、final-train/val 均无泄漏 (get_final_splits 内部已校验 dev/test 与 folds)
        assert_no_leak(splits['train_idx'], splits['test_idx'], groups, f"{name} dev/test")
        for k, (tr, va) in enumerate(splits['folds']):
            assert_no_leak(tr, va, groups, f"{name} fold{k+1}")
        print(f"  划分: dev={len(splits['train_idx'])}, test={len(splits['test_idx'])}, 5折CV", flush=True)

        print(f"  提取指纹+描述符 ({n_total} 样本)...", flush=True)
        X, valid = build_fingerprint_matrix(smiles_list, df=df)
        assert valid.all(), "无效SMILES应已在 build_fingerprint_matrix 中 raise"
        print(f"  特征矩阵: {X.shape}, continuous 列: {CONTINUOUS_START}..{X.shape[1]}", flush=True)

        models = build_models(seed=model_seed)
        for mname in model_names:
            assert mname in models, f"未知模型: {mname}, 可选 {list(models.keys())}"
            factory = FoldAwareFactory(models[mname], model_seed)
            try:
                res = run_one_model(mname, factory, X, y, splits, name, output_root,
                                    n_total, model_seed, groups)
                all_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{mname}|{name}] 失败: {e}", flush=True)
                all_results.append({'seed': model_seed, 'task': name, 'model': mname,
                                    'config': 'standard', 'n': n_total, 'cv_r2': np.nan,
                                    'cv_mae': np.nan, 'cv_rmse': np.nan, 'test_r2': np.nan,
                                    'test_mae': np.nan, 'test_rmse': np.nan,
                                    'train_time_sec': np.nan, 'run_status': 'FAIL',
                                    'error_message': str(e)[:300]})

    # 汇总 + per_seed_results.csv
    df_out = pd.DataFrame(all_results)
    per_seed_path = os.path.join(output_root, 'per_seed_results.csv')
    df_out.to_csv(per_seed_path, index=False)

    # 完整性检查
    expected = [(m, t['name']) for m in model_names for t in task_list]
    actual = list(zip(df_out['model'], df_out['task']))
    completeness_check(actual, expected, 'Stage1 traditional ML')  # 铁律5: 缺失即 raise
    print(f"[完整性] Stage1 ML: PASS (期望 {len(expected)} 组, 实际 {len(actual)} 组)")

    # 指标表
    print(f"\n{'='*70}\nStage 1 传统ML 指标表 (cv_mae / test_mae)\n{'='*70}")
    print(df_out[['task', 'model', 'cv_mae', 'test_mae', 'test_r2', 'run_status']].to_string(index=False))
    df_out.to_csv(os.path.join(output_root, 'all_ml_summary.csv'), index=False)
    print(f"\n结果保存: {output_root}")


if __name__ == '__main__':
    main()