# ring_encoding_ablation — SI 基线第三层：环信息注入位置消融（label / mask / pool / combined）

- **论文面板归属**：**SI**（环编码范式对比：Ring Labeling 输入层 / Ring Masking 传播层 / Ring Pooling 输出层 / RA-GCN 联合）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/ring_encoding_ablation/`

## 实验设定

先读第二层 GNN 基线汇总（`../baseline_gnn/results/all_gnn_summary.csv`）按三任务平均 test R² 选 **top5** 骨干
（GNN / GIN / GAT / MPNN / GraphSAGE），再对每个骨干施加三种环编码 `ENCODINGS = ['label','mask','pool']`，
并对每个骨干跑 `combined_ragcn.RAGCNCombined`（label+mask+pool 联合）；
任务 HOMA / NICS(1)zz / MBCO，8/2 留出 + 5 折 CV（seed=42，共用 `common.tasks.canonical_splits`），
指标 R²/MAE/RMSE。默认超参 `hidden_dim=128, n_conv_layers=3, epochs=200, patience=30`。

## 已复制文件清单

| 文件 | 来源 |
|---|---|
| `code/ring_train_eval.py` | `/home/ubuntu/aroma-dps-code/ring_encoding_ablation/code/ring_train_eval.py` |
| `code/combined_ragcn.py` | 同上目录 |
| `code/__init__.py` | 同上目录 |
| `configs/default.yaml` | 同上目录（含 `top5_source` 指向第二层汇总 CSV） |
| `__init__.py` | 包标记 |
| `run_all.sh` | 顶层启动脚本 |
| `results/all_ring_encoding_summary.csv` | `/home/ubuntu/aroma-dps-code/ring_encoding_ablation/results/all_ring_encoding_summary.csv`（成品汇总，9.4 KB） |

**结果留档说明**：最终 `results/` 逐折权重留存于工作目录
`/home/ubuntu/aroma-dps-code/ring_encoding_ablation/results/`（HOMA / MBCO / NICS_1zz 各约 16 MB，合计 48 MB），
**仓库仅含汇总 CSV**。`results/run.log` 未入仓。

## 未复制 / 暂置清单

- 未复制：`code/__pycache__/`、`__pycache__/`（`__init__.cpython-310.pyc`、`__init__.cpython-313.pyc`）——垃圾文件（规则 4）。
- 未复制：`results/HOMA/`、`results/MBCO/`、`results/NICS_1zz/`（`GAT_label`、`GAT_mask`、`GAT_pool`、`GAT_combined` 等
  逐模型逐折权重目录）与 `results/run.log`——规则 4。
- 依赖说明：本层用 `common.tasks` + `common.graph_data`，不 import `common.features`，
  故不受 `features.py` 修复替换影响（见 `../common/README.md`）。
  `code/combined_ragcn.py` 直接复用 `unified_models/*/model.py` 中的卷积层（仓库根已存在该包，但需路径处理，见下）。
- 本目录不含 `common/features.py`；未复制任何含 1-based ring index bug 的文件（规则 2）。
- 单文件 >5 MB 的排除项：本目录**无**。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `code/ring_train_eval.py`
  - 第 221 行：`--output_dir` 默认 `'/home/ubuntu/aroma-dps-code/ring_encoding_ablation/results'`；
  - 第 223 行：`--gnn_results` 默认 `'/home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv'`
    → 应指向仓库内 `../baseline_gnn/results/all_gnn_summary.csv`；
  - 第 28 行：`sys.path.insert(0, '/home/ubuntu/data_90/alldata_in_3090/model1')`；
  - 第 26–27 行：`PROJ_ROOT` 上溯三级（实测解析为 `experiments/si_baselines`，
    因此 `ring_encoding_ablation.code.combined_ragcn` 与 `common.*` 已可 import；
    缺口同为仓库根的 `unified_models.*`）。
- `code/combined_ragcn.py`
  - 第 19–20 行：`ORIG_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'` + `sys.path.insert(0, ORIG_ROOT)`
    （该注入是为 import 原始 `unified_models` 骨干；仓库内已有 `unified_models/`，需改指仓库根）。

附带（非 `.py`）：`run_all.sh` 的 `cd /home/ubuntu/aroma-dps-code`、`RESULTS=...`、
`--gnn_results /home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv`；
`configs/default.yaml` 的 `top5_source: /home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv`。
迁移时**均未修改**（规则 1）。
