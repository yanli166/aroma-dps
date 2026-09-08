"""
统一常量定义 (三层实验共用)

保证 CSV 列名、指标列、路径等全局一致, 避免 m3 (CSV格式不统一) 问题。
"""
import os

# ============== 路径配置 (m4: 消除硬编码) ==============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# code_end/common/ -> code_end/ -> aroma-dps/
_PROJ_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
AROMA_DATA_ROOT = _PROJ_ROOT
AROMA_PROJ_ROOT = _PROJ_ROOT

# 原始模型代码根目录 (unified_models 所在)
ORIG_MODELS_ROOT = os.path.join(_PROJ_ROOT, 'unified_models')

# ============== 统一 CSV 列名 (m3) ==============
# summary.csv (两列格式: metric, value) 中使用的 metric 名
SUMMARY_METRICS = [
    'cv_r2_mean', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec', 'n',
]

# per_seed_results.csv 的标准列顺序
PER_SEED_COLUMNS = [
    'seed', 'task', 'model', 'encoding',
    'n', 'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec',
]

# 聚合后 all_*_summary.csv 的列
AGG_COLUMNS = [
    'task', 'model', 'encoding',
    'cv_r2_mean', 'cv_r2_std', 'cv_mae_mean', 'cv_rmse_mean',
    'train_r2_mean', 'train_mae_mean', 'train_rmse_mean',
    'test_r2_mean', 'test_r2_std', 'test_mae_mean', 'test_rmse_mean',
    'train_time_sec_mean',
]

# 聚合时需计算 mean/std 的指标列
METRIC_COLS_FOR_AGG = [
    'cv_r2', 'cv_r2_std', 'cv_mae', 'cv_rmse',
    'train_r2', 'train_mae', 'train_rmse',
    'test_r2', 'test_mae', 'test_rmse',
    'train_time_sec',
]
