"""
三任务统一定义 + 规范化数据划分 (修复版)

数据更新 (0716):
  - HOMA       : collet_homa_0716.csv   -> homa_value
  - NICS(1)zz  : collet_nics_0716.csv   -> NICS_value
  - MBCO       : collet_mbco_0716.csv   -> mbco_value

集外测试数据:
  - lunci6-test.csv  : SMILES + HOMA + MBCO + NICS_ZZ
  - lunci78-test.csv : SMILES + HOMA + MBCO + NICS_ZZ
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common.constants import AROMA_DATA_ROOT

# 0716 新数据目录 (优先使用环境变量, 回退到 code_end/data1_end)
DATA1_END_DIR = os.environ.get(
    'DATA1_END_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data1_end')
)

TASKS = [
    {
        'name': 'HOMA',
        'dataset_path': os.path.join(DATA1_END_DIR, 'collet_homa_0716.csv'),
        'target_col': 'homa_value',
        'metric': 'HOMA (Harmonic Oscillator Model of Aromaticity)',
    },
    {
        'name': 'NICS_1zz',
        'dataset_path': os.path.join(DATA1_END_DIR, 'collet_nics_0716.csv'),
        'target_col': 'NICS_value',
        'metric': 'NICS(1)zz (Nucleus-Independent Chemical Shift)',
    },
    {
        'name': 'MBCO',
        'dataset_path': os.path.join(DATA1_END_DIR, 'collet_mbco_0716.csv'),
        'target_col': 'mbco_value',
        'metric': 'MBCO (Multiple Bond Order Criterion)',
    },
]

# 集外测试数据路径
EXTERNAL_TEST_FILES = {
    'lunci6':  os.path.join(DATA1_END_DIR, 'lunci6-test.csv'),
    'lunci78': os.path.join(DATA1_END_DIR, 'lunci78-test.csv'),
    'lunci10': os.path.join(DATA1_END_DIR, 'lunci10-test.csv'),
}

# lunci 测试集列名 → 主数据集列名映射
LUNCI_COL_MAP = {
    'SMILES': 'smiles',
    'Ring_Atoms': 'atom_on_ring',
    'HOMA': 'homa_value',
    'NICS_ZZ': 'NICS_value',
    'MBCO': 'mbco_value',
}

DEFAULT_SEED = 42
TEST_SIZE_RATIO = 0.2
N_FOLDS = 5

# 清洗后数据缓存 (避免重复处理)
_CLEAN_CACHE = {}


def clean_dataset_csv(dataset_path, target_col):
    """清洗数据集 CSV: 处理 Windows 换行符、列名空格、NaN 目标值、多余空列

    返回清洗后的临时 CSV 路径 (相同输入只处理一次)。
    """
    key = (dataset_path, target_col)
    if key in _CLEAN_CACHE:
        return _CLEAN_CACHE[key]

    df = pd.read_csv(dataset_path, encoding='utf-8-sig')
    # 去除列名首尾空白和 \r (Windows 换行符残留)
    df.columns = df.columns.str.strip()
    # 去除 Unnamed 空列 (NICS 文件有尾部逗号导致的空列)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    # 去除目标列为 NaN 的行
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col, 'smiles']).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"  [clean_dataset_csv] {os.path.basename(dataset_path)}: 剔除 {dropped} 行 NaN (target={target_col})")

    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(),
                            f"clean_{abs(hash(key))}_{os.path.basename(dataset_path)}")
    df.to_csv(tmp_path, index=False)
    _CLEAN_CACHE[key] = tmp_path
    return tmp_path


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
