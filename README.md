# Aroma-DPS：目标环条件化的芳香性预测（RC-GNN）

环级芳香性描述符（HOMA / NICS(1)zz / MBCO）的图神经网络预测，及其在去芳香化反应
上的应用。**本 README 是"论文面板 ↔ 代码 ↔ 复现命令"的对照表**；
目录设计依据与逐节点代码来源见 [`TARGET_FRAMEWORK_JACS.md`](TARGET_FRAMEWORK_JACS.md)。

- 论文方法学（可复用库）：[`src/aroma_dps/`](src/aroma_dps/)
- 面板级权威映射：[`p0_verification/paper_result_dependency_manifest.csv`](p0_verification/paper_result_dependency_manifest.csv)
- 环境：`/home/ubuntu/apps/anaconda3/envs/torch_env/bin/python`（Python 3.10, torch, PyG, rdkit, sklearn）

## 复现协议（全篇统一）

| 要素 | 取值 |
|------|------|
| 数据划分 | `get_final_splits(SPLIT_SEED=2026)`：分子级固定 80/20 group holdout + dev 内 GroupKFold(5) |
| 训练轮数 | E\* = median(各折 best epochs)，随后在 dev 全量重训 |
| 双特征轨道 | `standard` 与 `explicit_aromaticity_ablated` |
| 环索引语义 | CSV `atom_on_ring` 为 **0-based**（P0-2 审核，20,605 行）；1-based 版本一律不入仓 |
| ΔA 符号 | ΔHOMA=HOMA_R−HOMA_P；ΔMCBO=MCBO_R−MCBO_P；ΔNICS\*=NICS_P−NICS_R（正值=芳香性损失） |
| 非发表协议 | `canonical_splits(seed=model_seed)` → 测试集随模型 seed 变化，结果标记为 non-publication（见 [archive/deprecated/](archive/deprecated/)） |

## 面板 ↔ 代码 ↔ 命令

| 论文面板 | 代码位置 | 结果位置 | 复现命令 |
|----------|----------|----------|----------|
| **Fig.2** 数据集构建（DFT/NICS/MBCO 流程、立体描述符、lunci10 构建） | [`experiments/fig2_dataset/`](experiments/fig2_dataset/) | 输入表在 [`data/`](data/) | `bash experiments/fig2_dataset/nics_pipeline/run_all.sh`（需 Gaussian+Multiwfn） |
| **Fig.3a–3i** 环条件化 GNN 消融链（stage1–6） | [`0831-end-code/`](0831-end-code/)（冻结管线）；等价库实现见 `src/aroma_dps/{featurization,models,training}` | `0831-end-code/results_v2/` | `cd 0831-end-code && bash run_all_v2_parallel.sh`（5 seeds × 2 feature modes × 6 stages） |
| **Fig.3f** 可解释性（|x·∇x| 显著性 + Integrated Gradients） | [`experiments/fig3_model/interpretability/scripts/`](experiments/fig3_model/interpretability/scripts/) | [`experiments/fig3_model/interpretability/results/`](experiments/fig3_model/interpretability/results/) | 见该目录 [README](experiments/fig3_model/interpretability/README.md) |
| **Fig.4a–4h** 未见骨架泛化 / lunci10 外部基准 | 代码树 [`0901-end-code/fig4_lunci10/`](0901-end-code/fig4_lunci10/)；**当前图件的实际驱动脚本** [`experiments/fig4/generalization/`](experiments/fig4/generalization/) | [`0901-end-code/results/`](0901-end-code/results/)（`fig4_lunci10_final_v2/`, `fig4_generalization_retrain_v4_noe/`） | `python experiments/fig4/launcher.py --experiment <name>`（仅覆盖旧代码树，见 README 的告警） |
| **Fig.5** 策划反应验证（Aroma/CADA/Fsp3-NPR1 图件） | [`experiments/fig5_application/curated_reactions/`](experiments/fig5_application/curated_reactions/) | 同目录 `figures/` | `python experiments/fig5_application/curated_reactions/generate_aroma_figures.py` |
| **Fig.6** USPTO 反应级应用（去芳香化筛选） | [`experiments/fig6_uspto/`](experiments/fig6_uspto/)（`fig6_v2_main.py` / `fig6_v2_montage.py` / `dearom_screen*.py`） | 同目录 `fig6_analysis_v2/` 等 | 见 [`experiments/fig6_uspto/1README.md`](experiments/fig6_uspto/1README.md) |
| **SI** 传统 ML / GNN / 环编码消融 / 多 seed CV | [`experiments/si_baselines/`](experiments/si_baselines/) | 各子目录 `results/*.csv` | `bash experiments/si_baselines/<layer>/run_all.sh` |
| 端到端推理 API | [`src/aroma_dps/inference/ring_prediction.py`](src/aroma_dps/inference/ring_prediction.py) + [`models/best_model_package/`](models/best_model_package/) | — | `RingAromaticityPredictor().predict_reaction_pairs([((smi_r, ring_r), (smi_p, ring_p))])`<br>`ring_*` = 0-based 目标环原子索引 |

回归测试（划分确定性、ΔA 符号、环族标签、scaler 只在训练集 fit）：

```bash
/home/ubuntu/apps/anaconda3/envs/torch_env/bin/python -m pytest tests/ -q   # 32 passed
```

## 目录结构

```
aroma-dps/
├── src/aroma_dps/       # 论文方法学：config / data / featurization / models / training / chemistry / inference
├── data/                # 统一输入：collet(三任务) · lunci10 · lunci_external · reactions · interpretability
├── models/              # best_model_package（3 个 .pt + metrics.json）· interpretability 权重
├── experiments/         # fig2_dataset · fig3_model(+interpretability) · fig4(+generalization) · fig5_application · fig6_uspto · si_baselines
├── 0831-end-code/       # Fig.3 冻结管线（当前仍由此出图；run_all_v2_parallel.sh）
├── 0901-end-code/       # Fig.4 代码树 + results/（69 MB，131 文件）
├── unified_models/      # 跨阶段共享骨干（已抽取进 src/aroma_dps/models/backbones）
├── configs/ · scripts/ · tests/ · p0_verification/
└── archive/deprecated/  # 非发表协议代码（⚠️ 仍被 Fig.3/Fig.4 运行时依赖，见其 README）
```

## Data Availability

训练用分子-环级标签（HOMA / NICS(1)zz / MBCO）由 DFT 计算得到，输入构象取自
Collet 等（2019）公开数据集；外部基准 lunci6/7/8/10 为公开发表数据。
本仓库 `data/` 下保存的是**本项目自行计算/整理后的派生表**（原始 Gaussian/Multiwfn
中间产物体积过大未入仓，位置见各目录 README）。
⚠️ 待补：仓库 `.gitignore` 含 `*.csv`/`*.xlsx`/`*.pt`，因此 `data/` 与 `models/` 下的
数据与权重**目前不会被 git 跟踪**——发布前必须决定是改用 whitelisting、还是走外部
数据仓储（如 Zenodo/OSF）并给出 DOI。

## Code Availability

发布前需：(1) 决定 `archive/deprecated/` 是否随仓库公开；(2) 完成下方"已知缺口"；
(3) 许可证与 CITATION 文件（当前仅有 "Research use only"，非标准许可）。

## 已知缺口（发布前必须处理）

1. **面板编号冲突**：manifest 里 Fig.3f = mask pretraining，而可解释性工作也要放 3f；需重排字母。
2. **Fig.4 代码与结果不同代**：`0901-end-code/fig4_lunci10/` 在 `fig4_lunci10_final/` 中
   留下空面板（`02_novelty`/`03_pairwise`/`06_position`，且 `05_hammett/` 不存在），
   当前图件由 `experiments/fig4/generalization/` 产出且**尚未接入 launcher**。
3. **仍含绝对路径**：`experiments/fig4/generalization/`、`experiments/fig5_application/`、
   `experiments/fig3_model/interpretability/scripts/`、`models/best_model_package/`
   共 **50 个 .py** 硬编码 `/home/ubuntu/aroma-dps-code`（为保持搬运逐字节可追溯，本轮未改动）；
   另有 3 个 `p0_verification/audit_*.py` 指向工作目录属**有意为之**（审核正在运行的进程）。
4. **划分清单不一致**：`n_total` 6923 vs 6820（用户稍后处理）。
5. **归档耦合**：publication 代码运行时依赖 `archive/deprecated/code_end/`（详见其 README）。
6. **`src/aroma_dps` 尚未接管 Fig.3/Fig.4 出图**；`inference/reaction_prediction.py` 是显式 NOT-IMPLEMENTED 占位。
