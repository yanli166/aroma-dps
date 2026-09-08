"""
统一常量定义 (四阶段实验共用)

保证 CSV 列名、指标列、路径等全局一致。
"""
import os

# ============== 路径配置 ==============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# src/aroma_dps/ -> src/ -> <repo root>
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))

# Publication 布局: 数据与产物目录 (环境变量可覆盖, 便于指向外部数据盘)
COLLET_DIR = os.environ.get('AROMA_COLLET_DIR', os.path.join(REPO_ROOT, 'data', 'collet'))
LUNCI10_DIR = os.environ.get('AROMA_LUNCI10_DIR', os.path.join(REPO_ROOT, 'data', 'lunci10'))
REACTIONS_DIR = os.path.join(REPO_ROOT, 'data', 'reactions')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
RESULTS_DIR = os.environ.get('AROMA_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))

# 兼容别名: 从 0831-end-code vendored 进来的模块仍引用这些名字
AROMA_DATA_ROOT = REPO_ROOT
AROMA_PROJ_ROOT = REPO_ROOT
ORIG_MODELS_ROOT = os.path.join(REPO_ROOT, 'unified_models')
LAST_END_ROOT = os.path.join(REPO_ROOT, '0831-end-code')
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
