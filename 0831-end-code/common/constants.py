"""
统一常量定义 (四阶段实验共用)

保证 CSV 列名、指标列、路径等全局一致。
"""
import os

# ============== 路径配置 ==============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 0831-end-code/common/ -> 0831-end-code/ -> aroma-dps/
_PROJ_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
AROMA_DATA_ROOT = _PROJ_ROOT
AROMA_PROJ_ROOT = _PROJ_ROOT

# 原始模型代码根目录 (unified_models 所在, 卷积层复用)
ORIG_MODELS_ROOT = os.path.join(_PROJ_ROOT, 'unified_models')

# last_end_code 根目录
LAST_END_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# [0831 重构] 新结果根目录 + split 目录
RESULTS_V2_DIR = os.path.join(LAST_END_ROOT, 'results_v2')
SPLITS_DIR     = os.path.join(LAST_END_ROOT, 'splits')

# ============== 统一 CSV 列名 ==============
SUMMARY_METRICS = [
    'cv_r2_mean', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec', 'n',
]

PER_SEED_COLUMNS = [
    'seed', 'task', 'model', 'config',
    'n', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec',
]

AGG_COLUMNS = [
    'task', 'model', 'config',
    'cv_r2_mean', 'cv_r2_std', 'cv_mae_mean', 'cv_rmse_mean',
    'train_r2_mean', 'train_mae_mean', 'train_rmse_mean',
    'test_r2_mean', 'test_r2_std', 'test_mae_mean', 'test_rmse_mean',
    'train_time_sec_mean',
]

METRIC_COLS_FOR_AGG = [
    'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec',
]
