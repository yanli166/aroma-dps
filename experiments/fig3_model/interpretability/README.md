# Fig.3f — 可解释性 / 归因分析（interpretability）

- **面板归属**：**Fig.3f（可解释性 / 归因分析）**
- **来源**：`/home/ubuntu/aroma-dps-code/0908-end-code/`（代码 `scripts/`、产物 `results/`、`predictions/`、`figures/`、`attention_viz/`、`attention_l10val*/`、根目录 `logs_*.log`）
- **迁移日期**：2026-09-08
- **迁移原则**：**只复制，不改代码**；未重写逻辑、未改路径、未删文件。`__pycache__/`、`*.pyc`、>5MB 单文件排除（本批实际无 >5MB 文件）。

本目录是三任务归因 / 显著性分析（l10val 方案）与 merged / l10val 重训对照（merged 方案）的归档根目录。

## 本目录结构

| 子目录 | 内容 | 说明文件 |
|---|---|---|
| `scripts/` | 13 个 `.py`（两条实验线的全部代码） | `scripts/README.md`（含执行顺序图、**两个已知代码问题**、仓库外路径依赖与待相对化清单、ring index 自查结论） |
| `results/` | 指标 JSON、per-ringfamily CSV、5 个预测 CSV、2 张性能 PNG、4 个归因图集目录、`logs/` 6 个日志 | `results/README.md`（含**关键结论数值**、**待清理的旧选样残留图清单**、provenance 说明） |
| `pending/` | **无数据文件**，仅登记暂置项 | `pending/README.md`（`NICS_1zz_merged_best.pt` / `MBCO_merged_best.pt` 待补训、`plot_merged_figures.py` 三任务图未运行） |

## 关联位置（代码 / 数据 / 权重按用户框架分置于仓库三处）

| 类型 | 位置 |
|---|---|
| 代码 | 本目录 `scripts/` |
| 结果与图 | 本目录 `results/` |
| 数据表 | `/home/ubuntu/aroma-dps/data/interpretability/`（说明见 `/home/ubuntu/aroma-dps/data/README.md`） |
| 模型权重 | `/home/ubuntu/aroma-dps/models/interpretability/`（说明见其 `README.md`；**最终权重应走 Zenodo / Release**） |

## 快速导航

- 想复跑 → 先读 `scripts/README.md` §3（**当前脚本不可直接在新位置运行**：10/13 个脚本硬编码仓库外绝对路径，且 `ROOT` 目录约定与本仓库分置结构不匹配）。
- 想引数字 → `results/README.md` §2。
- 想知道哪些结论还缺证据 → `pending/README.md`。

> 本目录下 `results/` 的各子目录与 `results/logs/` 有意**不放各自 README**，以严格保持源目录相对结构与内容原貌；统一说明集中在 `results/README.md`。

## 面板编号待协调事项（不改变本次归属声明）

本目录按用户决定统一声明归属 **Fig.3f**。但仓库现有权威映射 `p0_verification/paper_result_dependency_manifest.csv` 中，**Fig.3f 一行当前登记的是 "Mask pretraining comparison (direct / random_mask / ring_mask)" → `0831-end-code/stage3_mask_pretraining/`**；而 `TARGET_FRAMEWORK_JACS.md` 曾建议把可解释性编为独立的 `fig7_interpretability`。
即面板号存在**一处冲突**：需要么把 mask-pretraining 改号、要么把本批可解释性内容改号。在 manifest 修订前，本目录所有文件仍按任务要求标注 Fig.3f；引用时请以最终拍板后的编号为准。
