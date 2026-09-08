# multiseed_cv — SI 稳健性：5 随机种子 × 5 折 CV（mean ± std）

- **论文面板归属**：**SI**（稳健性/重复性小节：一、二、三层在 5 个种子下的均值 ± 标准差）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/new5zhecv/`（"新 5 折 CV"）→ 目标目录重命名为语义化的 `multiseed_cv/`

## 实验设定

`DEFAULT_SEEDS = [42, 123, 456, 789, 2024]`。脚本**不修改一/二/三层原代码**，仅薄层包装复用其训练函数：
Layer 1 ML（`baseline_traditional_ml.code.ml_train_eval`）、Layer 2 GNN（`baseline_gnn.code.gnn_train_eval`）、
Layer 3 Ring（`ring_encoding_ablation.code.ring_train_eval`），逐种子落盘并支持断点续跑，
最终聚合为 `results/all_layers_summary.csv`（列含 `cv_r2_mean/std`、`test_r2_mean/std` 等，109 行）。

## 已复制文件清单

| 文件 | 来源 |
|---|---|
| `code/run_multiseed.py` | `/home/ubuntu/aroma-dps-code/new5zhecv/code/run_multiseed.py` |
| `run_all_multiseed.sh` | `/home/ubuntu/aroma-dps-code/new5zhecv/run_all_multiseed.sh`（4 GPU 并行总调度） |
| `results/all_layers_summary.csv` | `/home/ubuntu/aroma-dps-code/new5zhecv/results/all_layers_summary.csv`（成品汇总，33.6 KB） |

**结果留档说明**：最终 `results/` 逐种子逐折产物留存于工作目录 `/home/ubuntu/aroma-dps-code/new5zhecv/results/`
（**372 MB**，含 `layer1_ml`、`layer2_gnn`+`layer2_gnn_part{1,2,3}`、`layer3_ring`+`layer3_ring_part{1,1b,2,3}`，
每层下为 `seed_42/seed_123/...` × `HOMA/MBCO/NICS_1zz` 的权重与预测目录），**仓库仅含汇总 CSV**。

## 未复制 / 暂置清单

- 未复制：`/home/ubuntu/aroma-dps-code/new5zhecv/results/` 全部逐种子/逐折目录（372 MB）——规则 4 巨型中间产物。
- 未复制：`/home/ubuntu/aroma-dps-code/new5zhecv/logs/`（808 KB，`layer1_ml.log`、`layer2_gnn_gpu{1,2,3}.log`、
  `layer3_ring_gpu{1,2,3}.log`）——日志；`results/layer3_ring_part1b.log` 同因未复制。
- 未复制：`/home/ubuntu/aroma-dps-code/new5zhecv/code/__pycache__/run_multiseed.cpython-310.pyc`——垃圾文件（规则 4）。
- 未复制：`/home/ubuntu/aroma-dps-code/new5zhecv/new5zhecv/results/layer1_ml/`（12 KB 误建的空嵌套目录）——残留垃圾。
- 说明：本目录 `code/` 内**没有** `__init__.py`（来源即无），跨包 import 依赖 PROJ_ROOT（见下）。
- 本目录不含 `common/features.py`；未复制任何含 1-based ring index bug 的文件（规则 2）。
- 单文件 >5 MB 的排除项：本目录**无**（逐折权重目录整体排除，其中无单文件 >5 MB）。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `code/run_multiseed.py`
  - 第 26 行：`sys.path.insert(0, '/home/ubuntu/data_90/alldata_in_3090/model1')`；
  - 第 24–25 行：`PROJ_ROOT` 上溯三级（实测解析为 `experiments/si_baselines`，
    故对三层包与 `common.tasks` / `common.features` 的复用 import 已可用）；
  - 第 302 / 384 / 388 / 407 行：结果根目录由 `os.path.join(PROJ_ROOT, 'new5zhecv', 'results')` 拼出，
    目录改名后该常量段仍写作 `new5zhecv`，需指向 `multiseed_cv`（属路径相对化，本次未改）。

附带（非 `.py`）：`run_all_multiseed.sh` 第 19 行 `PROJ_ROOT=/home/ubuntu/aroma-dps-code`、
第 139 行内嵌 Python 的 `PROJ_ROOT = '/home/ubuntu/aroma-dps-code'`、`CONDA_SH=/home/ubuntu/apps/anaconda3/...`。
迁移时**均未修改**（规则 1）。

⚠️ 复现提示：Layer 1 的多种子指标由 `common/features.py` 提供特征；本仓库入仓的是**修复版**，
重跑 Layer 1 不会复现 `all_layers_summary.csv` 中的 Layer1-ML 行（详见 `../common/README.md`）。
Layer 2/3 行不受影响。
