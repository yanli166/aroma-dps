"""
三任务统一定义 + 规范化数据划分

三个任务对应三种芳香性指标 (互不相同的目标列):
  - HOMA       : collet_homa_0702.csv        -> homa_value
  - NICS(1)zz  : nics-nics1zz-out-no3.csv    -> Ring_NICS_ZZ_1
  - MBCO       : lunci2-mbcout.csv           -> homa_value (实为 MBCO 值)

为保证三层实验(传统ML / 通用GNN / 环编码消融)严格可比,
所有层共用本模块的同一套划分: seed=42, 8/2 留出测试集 + train_val 上 5 折 CV。
"""
import os
import numpy as np
from sklearn.model_selection import KFold

DATA_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'

TASKS = [
    {
        'name': 'HOMA',
        'dataset_path': os.path.join(DATA_ROOT, 'collet_homa_0702.csv'),
        'target_col': 'homa_value',
        'metric': 'HOMA (Harmonic Oscillator Model of Aromaticity)',
    },
    {
        'name': 'NICS_1zz',
        'dataset_path': os.path.join(DATA_ROOT, 'nics-nics1zz-out-no3.csv'),
        'target_col': 'Ring_NICS_ZZ_1',
        'metric': 'NICS(1)zz (Nucleus-Independent Chemical Shift)',
    },
    {
        'name': 'MBCO',
        'dataset_path': os.path.join(DATA_ROOT, 'outcsv', 'lunci2-mbcout.csv'),
        'target_col': 'homa_value',
        'metric': 'MBCO (Multiple Bond Order Criterion)',
    },
]

DEFAULT_SEED = 42
TEST_SIZE_RATIO = 0.2
N_FOLDS = 5


def get_task(task_name):
    for t in TASKS:
        if t['name'] == task_name:
            return t
    raise ValueError(f"未知任务: {task_name}, 可选: {[t['name'] for t in TASKS]}")


def canonical_splits(n_samples, seed=DEFAULT_SEED):
    """生成规范划分 (三层实验共用, 保证可比)

    返回:
        test_idx:        np.ndarray, 测试集索引 (20%)
        cv_folds:        list of (train_idx, val_idx), 在 train_val (80%) 上的 5 折
        final_train_idx: np.ndarray, 最终模型训练索引 (87.5% of train_val = 70% total)
        final_val_idx:   np.ndarray, 最终模型 early-stopping 验证索引 (12.5% of train_val)
    """
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    n_test = int(TEST_SIZE_RATIO * n_samples)
    test_idx = all_idx[:n_test]
    trainval_idx = all_idx[n_test:]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    cv_folds = []
    for tr_idx, va_idx in kf.split(trainval_idx):
        cv_folds.append((trainval_idx[tr_idx], trainval_idx[va_idx]))

    rng2 = np.random.RandomState(seed)
    tv_perm = trainval_idx.copy()
    rng2.shuffle(tv_perm)
    n_tv = len(tv_perm)
    n_final_train = int(0.875 * n_tv)
    final_train_idx = tv_perm[:n_final_train]
    final_val_idx = tv_perm[n_final_train:]

    return test_idx, cv_folds, final_train_idx, final_val_idx


def compute_metrics(true, pred):
    """统一指标: R2, MAE, RMSE"""
    from sklearn.metrics import r2_score
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    r2 = r2_score(true, pred)
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    return r2, mae, rmse
