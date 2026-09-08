# si_baselines/common — 三层基线共用的 `common` 包（含 features.py 的替换记录）

- **论文面板归属**：SI（基线与稳健性）——三层基线（传统 ML / 通用 GNN / 环编码注入消融）与多种子 CV 共用同一套任务定义与划分，故 `common/` 作为四目录的公共依赖放在本层。
- **迁移日期**：2026-09-08

## 已复制文件清单与来源

| 文件 | 来源绝对路径 | 状态 |
|---|---|---|
| `__init__.py` | `/home/ubuntu/aroma-dps-code/common/__init__.py` | 原样（空文件） |
| `tasks.py` | `/home/ubuntu/aroma-dps-code/common/tasks.py` | 原样（三任务定义 + `canonical_splits` + `compute_metrics`，seed=42 的 8/2 留出 + 5 折 CV） |
| `graph_data.py` | `/home/ubuntu/aroma-dps-code/common/graph_data.py` | 原样（`load_adj_format` / `load_pyg_format` 薄封装） |
| `features.py` | **替换**：`/home/ubuntu/aroma-dps/code_end/common/features.py` | **不是**工作目录原件 |

## 为什么 features.py 是替换版（规则 2 的直接后果）

`baseline_traditional_ml/code/ml_train_eval.py` 与 `multiseed_cv/code/run_multiseed.py` 都执行
`from common.features import build_fingerprint_matrix`，即基线目录需要 `common/features.py` 才能自成一体。

工作目录里的 `/home/ubuntu/aroma-dps-code/common/features.py` **被禁止入仓**：其
`compute_ring_descriptors()` 第 133 行为
`target_ring_atoms = [idx - 1 for idx in atom_on_ring if isinstance(idx, (int, float)) and idx > 0]`，
把 CSV 中已是 0-based 的 `atom_on_ring` 当 1-based 处理，导致目标环标记整体错位一个原子
（同一 bug 也存在于 `code_end/`、`last_end_code/`、`0831-end-code/` 的 `common/features.py`）。

因此改为复制仓库内已修复版本 `/home/ubuntu/aroma-dps/code_end/common/features.py`，其第 117 行为
`target_ring_atoms = [int(idx) for idx in atom_on_ring if isinstance(idx, (int, float)) and 0 <= idx < mol.GetNumAtoms()]`。

**API 兼容性核查**：修复版 `build_fingerprint_matrix(smiles_list, df=None, raise_on_invalid=True)` 与调用点兼容
（新增参数带默认值）；特征维度布局不变（MACCS 167 + Morgan 2048 + 分子描述符 16 + 环描述符 13 = 2244）。
修复版**未提供** `build_fingerprint_matrix_legacy()`（旧版遗留函数），经 grep 确认四份基线代码均未调用，不影响运行。

## ⚠️ 已知不可复现差异（必须在下一轮重跑中处理）

`baseline_traditional_ml/results/all_ml_summary.csv` 与 `multiseed_cv/results/all_layers_summary.csv` 中的
Layer-1 传统 ML 指标，是在**含 bug 的** `features.py`（环描述符错位）下产出的。
本目录入仓的是修复版，因此**用当前代码重跑 Layer 1 不会复现已归档的 CSV 数值**。
Layer 2（通用 GNN）与 Layer 3（环编码消融）走 `common/graph_data.py`，不使用 `common/features.py`，不受该替换影响。
投稿前需以修复版重算 Layer 1 并更新汇总 CSV（或在正文注明该层为旧特征版）。

## 未复制 / 暂置清单

- 未复制：`/home/ubuntu/aroma-dps-code/common/features.py`——含 1-based ring index bug（规则 2，绝对禁止入仓）。
- 未复制：`/home/ubuntu/aroma-dps-code/common/__pycache__/*.pyc`——垃圾文件（规则 4）。
- 未复制：`/home/ubuntu/aroma-dps/code_end/common/constants.py`——修复版 `features.py` 不依赖它；
  本目录的 `tasks.py` / `graph_data.py` 保持工作目录原样（读取 0702 旧表），未与 `code_end/common/` 中
  已按 0716 表 + 路径配置化重构的同名模块混用，以免改变已归档基线实验的输入语义。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `tasks.py`（第 16 行 `DATA_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'`；三任务的 `dataset_path`
  拼自该根目录下的**旧表名** `collet_homa_0702.csv`、`nics-nics1zz-out-no3.csv`、`outcsv/lunci2-mbcout.csv`）
- `graph_data.py`（第 19 行 `ORIG_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'`，用于定位预生成图数据）

⚠️ 相对化时注意数据版本：`tasks.py` 读的是 **0702 旧表**，而仓库 `data/collet/` 入仓的是 **0716 表**
（`collet_{homa,nics,mbco}_0716.csv`）。基线已归档数值来自 0702 表，重跑前需决定"指向 0716 并重跑"还是
"补入 0702 表以复现原值"，不能默默替换。

另需注意（已实测）：四份基线脚本的 `PROJ_ROOT = 上溯三级` 在新布局下解析为
`/home/ubuntu/aroma-dps/experiments/si_baselines`，因此 `common.tasks` / `common.features` /
`common.graph_data` 以及跨包 import（`baseline_traditional_ml.code.*`、`ring_encoding_ablation.code.*`、
`baseline_gnn.code.*`）**开箱可用**；仍缺的是 `unified_models.*`（位于仓库根 `/home/ubuntu/aroma-dps/unified_models`，
不在 PROJ_ROOT 下，第二/三层需要它）与硬编码的 `/home/ubuntu/data_90/alldata_in_3090/model1`。
这两项属"路径相对化"待办，本次未改（规则 1）。
