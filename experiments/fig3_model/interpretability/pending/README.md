# Fig.3f — 暂置项（PENDING）

- **面板归属**：Fig.3f（可解释性 / 归因分析）
- **来源**：`/home/ubuntu/aroma-dps-code/0908-end-code/`（本目录对应的源实验根目录）
- **迁移日期**：2026-09-08

本目录**不含任何数据文件**，仅登记"merged 方案尚未完成、因而本次无可复制对象"的暂置项。
下列产物在源目录 `/home/ubuntu/aroma-dps-code/0908-end-code/` 中**根本不存在**，故不属于遗漏未搬，而是等待补训/补跑；补齐后应按 `../scripts/README.md` 的目录约定回填至对应位置。

---

## 暂置项清单

| # | 暂置项 | 期望位置 | 状态 | 由哪个脚本产出 | 阻塞原因 / 证据 |
|---|---|---|---|---|---|
| P1 | `NICS_1zz_merged_best.pt` | `/home/ubuntu/aroma-dps/models/interpretability/` | **待补训** | `../scripts/train_merged.py --task NICS_1zz`（Step 2） | 训练启动后未跑完。`results/logs/logs_train_nics.log` 全文只有 RDKit `UFFTYPER: Warning: hybridization set to SP3 …` 警告，**无任何 epoch、无 `[val]`/`[test]` 指标行、无 `[saved]` 行**，即在建图/预处理阶段中断。输入数据 `data/interpretability/merged_NICS_1zz.csv` 已就绪（8744 行，dev 5311 / test 1281 / l10 2152）。 |
| P2 | `MBCO_merged_best.pt` | `/home/ubuntu/aroma-dps/models/interpretability/` | **待补训** | `../scripts/train_merged.py --task MBCO`（Step 2） | 无任何运行痕迹（`logs/` 中连部分日志都不存在）。输入数据 `data/interpretability/merged_MBCO.csv` 已就绪（8851 行，dev 5360 / test 1339 / l10 2152）。 |
| P3 | merged 三任务 parity 图 + 旧基线 vs merged 三任务指标对比图 | `../results/figures/` | **从未运行** | `../scripts/plot_merged_figures.py`（Step 4） | 脚本本身完好且已复制，但其图需 `results/{HOMA,NICS_1zz,MBCO}_metrics.json` 三件套；现仅有 `HOMA_metrics.json`，故 Step 4 从未执行，`figures/` 下只有 Step 6（`plot_homa_figures.py`）产出的 2 张 HOMA 图。 |

### 连带阻塞

- **P1/P2 未补 → P3 无法运行**（缺 metrics.json）。
- **P1/P2 未补 → `../scripts/saliency_merged.py` 不能作为独立脚本运行**：其 `MODEL_FILES`（第 67–71 行）硬编码引用 `NICS_1zz_merged_best.pt` 与 `MBCO_merged_best.pt`，直接执行必然 `FileNotFoundError`。详见 `../scripts/README.md` §2 问题②。现有 5 类代表分子的三任务归因对比图（其 `three_task_comparison/` 输出）**亦属暂置**。
- **旧基线对照目前只有 HOMA**：`results/results/old_baseline_metrics.json` 仅含 `"HOMA"` 一个键（R²=0.9944 / MAE=0.1001 / n_test=1335），因为 `eval_old_baseline.py`（Step 3）的对比同样以 merged 模型已训完为前提。补训 NICS/MBCO 后 Step 3 需一并重跑，否则 Fig.3f 的"与旧基线对比"论据只覆盖 HOMA。

---

## 补齐后的验收清单（供后续操作者核对）

1. 训练：`models/interpretability/{NICS_1zz,MBCO}_merged_best.pt` 出现，且 `config.seed == 11`、`config.ring_flag_value == 1`、`use_projection == false`（NICS/MBCO 无 projection，与现有两个 l10val 权重一致）。
2. 指标：新增 `results/{NICS_1zz,MBCO}_metrics.json`；`predictions/{NICS_1zz,MBCO}_test_predictions.csv`。
3. 基线：`old_baseline_metrics.json` 扩展为含三任务。
4. 图：运行 `plot_merged_figures.py` 后，`figures/` 下应新增三任务 parity 与三任务指标对比图。
5. 全程**不得**改动 `../scripts/` 下已归档的 13 个脚本的现有逻辑；如需改动，请在 `../scripts/README.md` 增补变更记录，避免与本目录 provenance 脱钩。

> 提醒：本目录所列项目缺失是**实验进度问题，不是迁移遗漏**。搬运时源目录的实际文件已 100% 复制完毕（校验方式见 `../results/README.md` 与 `../scripts/README.md`）。
