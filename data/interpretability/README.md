# `data/interpretability/` — Fig.3f 可解释性数据集

- **面板归属**：Fig.3f（可解释性 / 归因分析）
- **来源**：`/home/ubuntu/aroma-dps-code/0908-end-code/data/`
- **迁移日期**：2026-09-08
- **文件数**：10（6 个 `.csv` + 4 个 `.json`），均 <5MB；已与源目录 `diff -r` 验证逐字节一致。

完整说明（列语义、`split` 取值定义、merged 与 l10val 两套方案的差异、三任务共享的 272 个验证分子 / 1087 个训练分子、逐文件行数与 split 分布、**MD5 校验和表**）见上级文档：

→ **`/home/ubuntu/aroma-dps/data/README.md` 的「`interpretability/` — Fig.3f 可解释性 / 归因分析数据集」一节**

配套代码：`/home/ubuntu/aroma-dps/experiments/fig3_model/interpretability/scripts/`
配套结果：`/home/ubuntu/aroma-dps/experiments/fig3_model/interpretability/results/`
配套权重：`/home/ubuntu/aroma-dps/models/interpretability/`

一句话摘要：本目录是 Fig.3f 两条实验线的**环级训练/验证表** —— `merged_*` 系列（`split` = dev/test/l10，用于"并入 lunci10 后与旧基线在同一 collet test 上对比"）与 `*_l10val` 系列（`split` = train/l10_val，用于"在 lunci10 分子级留出集上训练三任务模型并做 IG + \|x·∇x\| 双归因"）。`atom_on_ring` 全程 **0-based**。
