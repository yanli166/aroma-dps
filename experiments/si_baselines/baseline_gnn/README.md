# baseline_gnn — SI 基线第二层：通用 GNN backbone × 3 任务

- **论文面板归属**：**SI**（基线对比小节：不注入目标环信息的通用 GNN）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/baseline_gnn/`

## 实验设定

自实现 backbone（GNN / MPNN / GAT / GIN / GraphSAGE，复用仓库根 `unified_models/*/model.py`）
+ PyG 模型（AttentiveFP、DMPNN，见 `code/pyg_models.py`），在 HOMA / NICS(1)zz / MBCO 上
8/2 留出 + 5 折 CV（seed=42，与一、三层共用 `common.tasks.canonical_splits`），200 epoch / patience 30。

## 已复制文件清单

| 文件 | 来源 |
|---|---|
| `code/gnn_train_eval.py` | `/home/ubuntu/aroma-dps-code/baseline_gnn/code/gnn_train_eval.py` |
| `code/pyg_models.py` | 同上目录 |
| `code/__init__.py` | 同上目录 |
| `configs/default.yaml` | 同上目录 |
| `__init__.py` | 包标记 |
| `run_all.sh` | 顶层启动脚本（GPU 0） |
| `results/all_gnn_summary.csv` | `/home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv`（成品汇总表，3.2 KB） |

**结果留档说明**：最终 `results/` 逐折权重与训练曲线留存于工作目录
`/home/ubuntu/aroma-dps-code/baseline_gnn/results/`（HOMA / MBCO / NICS_1zz 三个任务目录，共 16 MB），
**仓库仅含汇总 CSV**。`results/run.log` 未入仓。

## 未复制 / 暂置清单

- 未复制：`code/__pycache__/`、`__pycache__/`（`__init__.cpython-310.pyc`）——垃圾文件（规则 4）。
- 未复制：`results/HOMA/`、`results/MBCO/`、`results/NICS_1zz/`（逐模型逐折权重目录，含 `.pt`）与 `results/run.log`
  ——规则 4；单个权重文件虽 <5 MB，但属"逐折权重目录"整体排除。
- 依赖说明：本层用 `common.tasks` + `common.graph_data`（不 import `common.features`），
  因此**不受 `features.py` 的 bug 修复替换影响**；`common/` 的组装说明见 `../common/README.md`。
  另依赖仓库根 `unified_models/{gnn,mpnn,gat,gin,graphsage}/model.py`（已在仓库内）与
  `baseline_gnn.code.pyg_models`（函数内 import，依赖 PROJ_ROOT 推导，见下）。
- 单文件 >5 MB 的排除项：本目录**无**。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `code/gnn_train_eval.py`
  - 第 287 行：argparse `--output_dir` 默认值 `'/home/ubuntu/aroma-dps-code/baseline_gnn/results'`；
  - 第 29 行：`sys.path.insert(0, '/home/ubuntu/data_90/alldata_in_3090/model1')`（原始图数据根目录）；
  - 第 27–28 行：`PROJ_ROOT = 上溯三级` + `sys.path.insert(0, PROJ_ROOT)`（实测在新布局下解析为
    `experiments/si_baselines`，故第 130 行的 `baseline_gnn.code.pyg_models` 已可 import）；
  - 该处的**遗留缺口**是 `unified_models.{gnn,mpnn,gat,gin,graphsage}.model`：包在仓库根
    `/home/ubuntu/aroma-dps/unified_models/`，不在 PROJ_ROOT 下，需追加仓库根到 `sys.path`（或改走 `src/aroma_dps/models`），本次未改。

附带（非 `.py`）：`run_all.sh` 中 `cd /home/ubuntu/aroma-dps-code`、
`RESULTS=/home/ubuntu/aroma-dps-code/baseline_gnn/results`。迁移时**未修改**（规则 1）。
