# best_model_package — 三任务最终交付模型 + 统一推理接口 AromaticityPredictor

- **论文面板归属**：Methods（最终模型与推理设置）＋ Fig.5 / Fig.6 全部下游应用的**唯一权重来源**；
  也是 `src/aroma_dps/inference/ring_prediction.py` 待封装的实现参照。
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/best_model_package/`（原样复制，未改动代码、权重与格式）

## 内容

这是三任务（HOMA / NICS(1)zz / MBCO）最终交付模型包，含架构、修复版图构建、终训脚本与统一推理接口
`AromaticityPredictor`：一次加载三个 checkpoint，输入 `SMILES + atom_on_ring`（0-based），
同时输出三个芳香性指标，支持 `predict()` / `predict_batch()` 与命令行调用。

## 已复制文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `model_arch.py` | 5.9 KB | Ring-Conditioned GNN 架构定义 |
| `graph_utils.py` | 5.7 KB | 图构建（**自包含、已修复** `target_ring_atoms = [int(idx) for idx in atom_on_ring]`） |
| `train_final.py` | 9.5 KB | 终训脚本（读 `collet_*_0716.csv`，训练/验证/测试口径见 `metrics.json`） |
| `predict.py` | 4.2 KB | `AromaticityPredictor` 统一推理接口 |
| `homa_best.pt` | 1.8 MB | HOMA 最优权重（`use_projection=true`） |
| `nics_1zz_best.pt` | 1.7 MB | NICS(1)zz 最优权重 |
| `mbco_best.pt` | 1.7 MB | MBCO 最优权重 |
| `metrics.json` | 1.0 KB | 三任务 val_r2 / val_mae / val_rmse / best_epoch / 样本量 / `ring_flag_value` / 权重文件名 |

指标快照（`metrics.json`）：HOMA val R²=0.9908、NICS_1zz val R²=0.9814、MBCO val R²=0.9823；
三任务均 `ring_flag_value=1`（binary 目标环标记），仅 HOMA 使用 projection 头。

## ⚠️ 权重分发方式（投稿前必办）

**权重最终应走 Zenodo / GitHub Release，而非 git。** 当前仓库 `.gitignore` 已含 `*.pt`，
因此这三个 `.pt`（合计约 5.2 MB）只是**本地暂存**，不会随 commit 进入 git 历史。
发布流程：上传至 Zenodo 取得 DOI → 在 Release 与本 README 写入版本号与 md5/sha256 →
`predict.py` 增加按 DOI/URL 拉取并校验哈希的缓存逻辑。git 中只保留代码 + `metrics.json` + 校验和。

## 未复制 / 暂置清单

- 未复制：`/home/ubuntu/aroma-dps-code/best_model_package/__pycache__/`
  （`graph_utils.cpython-310.pyc`、`graph_utils.cpython-313.pyc`、`model_arch.cpython-310.pyc`、
  `predict.cpython-310.pyc`）——垃圾文件（规则 4）。
- 未复制：`/home/ubuntu/aroma-dps-code/best_model_package/train.log`（5.2 KB 终训日志）——
  日志类，`.gitignore` 排除 `*.log`；数值结论以 `metrics.json` 为准，原件留存工作目录。
- 单文件 >5 MB 的排除项：本包**无**（最大文件 `homa_best.pt` 1.84 MB，未触及 5 MB 上限）。
- 规则 2 说明：`graph_utils.py` 经核对为**修复版**（第 51 行 `target_ring_atoms = [int(idx) for idx in atom_on_ring]`；
  其第 6 行仅在文档字符串中引述被禁的 `[idx - 1 for idx in atom_on_ring]` 以说明 bug 来由，非真实代码行），故允许入仓。
  本包**不需要**也**未复制**任何 `common/features.py`。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 的 `.py` 文件：

- `train_final.py`（第 36 行 `DATA_ROOT = '/home/ubuntu/aroma-dps-code/code_end/data1_end'`，
  其拼出的 `collet_homa_0716.csv` / `collet_nics_0716.csv` / `collet_mbco_0716.csv`
  **已迁入仓库** `data/collet/`，相对化只需改指该目录）
- `predict.py`（第 6 行文档字符串用法示例中的 `AromaticityPredictor('/home/ubuntu/aroma-dps-code/best_model_package')`，
  示例包路径应改为仓库相对路径或 `PKG_DIR` 自身；属文档示例，非运行期硬编码）

`model_arch.py`、`graph_utils.py` 不含绝对路径；`predict.py` 的 checkpoint 与 `metrics.json` 均按
`os.path.dirname(os.path.abspath(__file__))` 定位，可随目录整体移动。
