# baseline_traditional_ml — SI 基线第一层：9 种传统机器学习 × 3 任务

- **论文面板归属**：**SI**（基线对比小节：传统 ML 上界）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/baseline_traditional_ml/`

## 实验设定

CatBoost / RandomForest / SVM(SVR) / LightGBM / XGBoost / ExtraTrees / Ridge / KRR / MLP 共 9 个回归模型，
输入为 MACCS(167) + Morgan(2048, r=2) + 分子描述符(16) + 环描述符(13)，
在 HOMA / NICS(1)zz / MBCO 三任务上跑 8/2 留出 + 5 折 CV（seed=42，与二、三层共用 `common.tasks.canonical_splits`），
指标 R²/MAE/RMSE。

## 已复制文件清单

| 文件 | 来源 |
|---|---|
| `code/ml_train_eval.py` | `/home/ubuntu/aroma-dps-code/baseline_traditional_ml/code/ml_train_eval.py` |
| `code/__init__.py` | 同上目录 |
| `configs/default.yaml` | 同上目录（特征维度、9 模型清单与超参注释） |
| `__init__.py` | 包标记 |
| `run_all.sh` | 顶层启动脚本（GPU 1 跑 CatBoost/XGBoost/LightGBM，其余 CPU） |
| `results/all_ml_summary.csv` | `/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results/all_ml_summary.csv`（成品汇总表，4.1 KB） |

**结果留档说明**：最终 `results/` 逐折/逐模型产物留存于工作目录
`/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results/`（HOMA / MBCO / NICS_1zz 三个任务目录，共 3.6 MB，
含各模型逐折预测与权重），**仓库仅含汇总 CSV**。`results/run.log` 亦未入仓（`.gitignore` 排除 `*.log`）。

## 未复制 / 暂置清单

- 未复制：`code/__pycache__/`、`__pycache__/`（工作目录含 `__init__.cpython-310.pyc`）——垃圾文件（规则 4）。
- 未复制：`results/HOMA/`、`results/MBCO/`、`results/NICS_1zz/`（逐模型/逐折目录）与 `results/run.log`——
  规则 4「各 results/ 下逐折权重目录」+ 日志；数值以 `results/all_ml_summary.csv` 为准。
- 依赖说明：本层 `from common.features import build_fingerprint_matrix`，`common/features.py` **不是工作目录原件**，
  而是仓库内已修复版本；该替换及其导致的数值不可复现问题完整记录在 `../common/README.md`（规则 2）。
- 单文件 >5 MB 的排除项：本目录**无**（来源中不存在 >5 MB 文件）。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `code/ml_train_eval.py`
  - 第 182 行：argparse `--output_dir` 默认值 `'/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results'`
    → 应改为仓库内相对默认（如 `Path(__file__).parents[1]/"results"`）；
  - 第 28–29 行：`PROJ_ROOT = 上溯三级` + `sys.path.insert(0, PROJ_ROOT)`（实测在新布局下解析为
    `experiments/si_baselines`，故 `common.tasks` / `common.features` 已可 import，无需改动此处）。
  - 其余需一并处理：`common/tasks.py` 的 `DATA_ROOT`（`/home/ubuntu/data_90/.../model1`，指向 0702 旧表）。
  - 注：本文件**不含** `/home/ubuntu/data_90` 硬编码（与二、三层不同，它不直接注入原始图数据根目录）。

附带（非 `.py`）：`run_all.sh` 中 `cd /home/ubuntu/aroma-dps-code` 与
`RESULTS=/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results`。迁移时**未修改**（规则 1）。
