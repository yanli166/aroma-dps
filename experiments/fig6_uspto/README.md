# Fig.6 — USPTO 反应级应用（USPTO-scale reaction application）

## 论文面板归属

本目录承载论文 **Fig.6 "USPTO-scale ML-enabled chemical discovery"** 的完整代码与成品结果，
链条为：**去芳构化反应挖掘 → 原子映射与 Tier-A 精筛 → 环对（reactant/product target-ring pair）ML 样本
→ 环索引校正 → 逐环芳香性预测 → 化学空间/统计分析与出图**。

- 来源绝对路径：`/home/ubuntu/aroma-dps-code/uspto-5k/`
- 迁移日期：**2026-09-08**
- 迁移方式：**原样复制**（`cp -p` / `rsync -a`），未改动任何代码、路径或格式；
  全部复制文件已与来源逐字节 `cmp` 校验一致。

## ⚠️ 命名与面板归属差异（务必先读，避免后续混淆）

本目录内有一批 **名字带 `fig5`、但面板归属为 Fig.6** 的文件：
`fig5_analysis.py`、`fig5_chemspace.py`、`fig5_schematic_si.py`、以及整个 `fig5_analysis/` 结果树
（内含 `Fig5a_workflow` … `Fig5f_chemical_space_discordant`、`S1`–`S10`）。

原因是任务书的**编号在项目中后期被改过**：

- `fig6.txt`（21 KB，早期任务书）原文要求：「…对 reaction-level aromaticity change 进行系统分析
  并生成**论文 Fig.5 的正式数据和图片**」，并逐节定义 `Fig.5a`–`Fig.5e`。
  → 这就是 `fig5_analysis.py` 及其输出被命名为 Fig.5 的由来。
- `fig62.txt`（18 KB，后续任务书）随后**冻结了论文结构**：
  `Fig.5 Curated real-reaction DFT validation` / `Fig.6 USPTO-scale ML-enabled chemical discovery`，
  并明确「因此新的 Fig.6 不再负责证明模型预测准确；Fig.5 已经负责 DFT reaction validation」。

**结论**：本目录中所有 `fig5_*` 代码与产物，在当前冻结结构下**属于 Fig.6 面板**（USPTO 反应级分析），
而**不是** `experiments/fig5_application/curated_reactions/`（83 对策划反应）那组图。
按「只复制、不改代码」的要求，本次迁移**未重命名**这些文件与目录；重命名与面板编号统一留待作者决定。

## 已复制清单（共 249 个文件 + 本 README）

### 1. 管线脚本（22 个 `.py`，位于本目录根）

| 阶段 | 文件 |
|---|---|
| 去芳构化初筛 | `dearom_screen.py`、`dearom_screen_fast.py` |
| 原子映射 | `map_uspto_rxnmapper.py` |
| Stage-2 精筛 / 分档 | `dearom_stage2_exact.py`、`summarize_manual_review.py` |
| 环对 ML 样本构建 | `build_ml_ring_pairs.py` |
| **索引校正** | **`fix_ring_indices.py`**（见下节核实结论） |
| 逐环性质预测 | `predict_ring_properties_fixed.py`、`fig6_v2_predict_spectator.py`（旁观环对照） |
| 分析与出图（旧 Fig.5 编号） | `fig5_analysis.py`、`fig5_chemspace.py`、`fig5_schematic_si.py`、`chemical_space_multiview.py` |
| 分析与出图（新 Fig.6 编号） | `fig6_v2_data_audit.py`、`fig6_v2_pair_inventory.py`、`fig6_v2_panel_a.py`、`fig6_v2_main.py`、`fig6_v2_ring_x_reaction.py`、`fig6_v2_reaction_family.py`、`fig6_v2_montage.py`、`fig6_v2_panel_f_ring.py` |
| 采样 | `example_ml_loader.py` |

**`.sh` 文件：来源目录 `uspto-5k/` 下不存在任何 `.sh`**（已 `find -name "*.sh"` 确认为 0）。
管线的实际调用命令以代码块形式写在 `1README.md`、`DEAROM_FAST_README.md`、`DEAROM_SCREEN_README.txt` 内。

### 2. README / 流程说明文档（7 个，位于本目录根）

`1README.md`（索引校正 + 预测主流程工单）、`README_RING_PAIRS.md`、`README_STAGE2.md`、
`DEAROM_FAST_README.md`、`DEAROM_SCREEN_README.txt`、`fig6.txt`（旧 Fig.5 编号任务书）、
`fig62.txt`（冻结论文结构、新 Fig.6 任务书）。

分析子目录下另有 4 份 `.md` 分析报告与 1 份 `README_summary.txt`（随结果树一并复制）：
`fig5_analysis/FIG5_ANALYSIS_REPORT.md`、`fig5_analysis/chemical_space_multiview/README_summary.txt`、
`fig6_analysis_v2/REPORT/FIG6_ANALYSIS_REPORT.md`、`fig6_analysis_v2/DATA_AUDIT/Fig6_DATA_AUDIT.md`、
`fig6_analysis_v2/REACTION_FAMILY_OPTIONAL/reaction_family_audit_report.md`。

### 3. `dearom_ring_pairs_A_tierA/` — Tier-A 最终 ML 样本与预测表（12 个文件，全部 < 5 MB）

- **`ring_property_ml_samples_fixed.csv`（3.47 MB）— ✅ 已修复版本**（`fix_ring_indices.py` 的直接输出，含
  `smiles_model` / `target_atom_indices_model` / `target_ring_mask_model` / `index_status` 等校正列）
- `ring_property_ml_samples_fixed_index_report.json`（1 KB）— 修复统计报告（核实依据）
- `ring_property_ml_samples_fixed_index_errors.csv`（8 KB）— 未通过 PASS 门槛的样本（本次为 0 条）
- `ring_property_ml_samples.csv`（2.40 MB）— **修复前**版本，一并保留以对照差异
- `ring_property_predictions.csv`（4.02 MB）— 用校正后索引跑出的逐环样本级预测
- `ring_pair_aromaticity_predictions.csv`（0.82 MB）/ `ring_pair_aromaticity_predictions_complete.csv`（0.85 MB，4,377 对）— 环对级 Δ 汇总，**Fig.6 分析的直接输入**
- `ring_pairs_ml.csv`（4.51 MB）/ `ring_pairs_ml.jsonl`（4.52 MB）/ `ring_pair_report.json` / `ring_pair_errors.csv` / `descriptor_discordant_cases.csv`

### 4. `fig5_analysis/` — 反应级芳香性分析成品（101 个文件，保持原有相对子目录结构）

`01_master`（1）· `02_global_change`（3）· `03_descriptor_coupling`（3）· `04_pca`（3）·
`05_discordant`（3）· `06_chemical_space`（2）· `07_ring_family`（2）· `08_statistics`（3）·
`09_fig5_main`（33：`Fig5a`–`Fig5f` 与 chemspace 配色图，png/svg/pdf）·
`10_fig5_si`（28：`S1`–`S10`）· `chemical_space_multiview`（19：PCA/t-SNE/UMAP 多视角图）·
`FIG5_ANALYSIS_REPORT.md`（1）· `logs/`（来源即空目录）。

### 5. `fig6_analysis_v2/` — 新版 Fig.6 面板成品（107 个文件，保持原有相对子目录结构）

根（6：`Fig6_CORE_candidate` / `Fig6_ENHANCED_candidate` 的 png/svg/pdf）·
`DATA_AUDIT`（5）· `PANEL_A_PIPELINE`（4）· `PANEL_B_TARGET_SPECTATOR`（11）·
`PANEL_C_GLOBAL_LOSS`（7）· `PANEL_D_RING_FAMILY`（9，含 `contact_sheets/` 13）·
`PANEL_E_STRUCTURAL_CHANGE`（5）· `PANEL_F_DISCORDANT`（12）·
`REACTION_FAMILY_OPTIONAL`（3，含 `reaction_family_contact_sheets/` 4）·
`REPORT`（1）· `SI`（10：`S11`–`S12`）· `VARIANCE_DECOMPOSITION_OPTIONAL`（17）。

## ✅ `fix_ring_indices.py` 核实结论

**1) 它修复的不是 ring index 的 1-based/0-based bug，而是 canonical SMILES 原子索引错位（index realignment）。**
文件头 docstring 自述问题：`target_atom_indices` 由解析 `smiles_mapped` 得到，
而喂给模型的是**另行 canonicalize 的 plain `smiles`**；RDKit canonicalization 会**重排原子顺序**，
因此两套索引一般不可互换。这与 `common/features.py` 中
`target_ring_atoms = [idx - 1 for idx in atom_on_ring]` 的 **1-based off-by-one bug 是两个不同问题**。

**2) 修复方法（代码判读，未改动）**：以 `smiles_mapped` 为权威 → 按 atom-map number 定位目标原子 →
`Chem.Mol(mol)` 拷贝后仅清零 map 编号（**不改变原子顺序**）→ canonical `MolToSmiles` →
读取 `_smilesAtomOutputOrder`（`output_order[new_index] = old_index`）→ 反查 old→new 索引 →
重新解析 emitted plain SMILES，用 `ring_induced_graph_check()` 验证目标原子仍构成
「边数 = 原子数、每个目标原子内部度 = 2」的简单环。产出列
`smiles_model`、`target_atom_indices_model`、`target_ring_mask_model`、`index_changed`、
`existing_smiles_matches_model`、`target_cycle_valid_model`、`index_status`；
并明确「模型**只应**消费 `smiles_model` + `target_atom_indices_model`」。

**3) 修复已生效，且最终数据表确实被后续 Fig.6 脚本读取。** 证据链：

- 报告 `dearom_ring_pairs_A_tierA/ring_property_ml_samples_fixed_index_report.json`：
  `n_samples = 8886`、`n_pass = 8886`、`n_fail = 0`、`n_index_changed = 2347`（**26.41% 的样本索引确实被重排**）、
  `PASS_UNCHANGED = 6539`、`PASS_REINDEXED = 2347`、结构一致 8886、环校验有效 8886。
  → 修复**已跑通且全量通过**；26.41% 的重排比例正说明原索引假设的危险程度（`1README.md` Step 2 亦如此解读）。
- 调用链写在 `1README.md`：Step 1 `fix_ring_indices.py --input dearom_ring_pairs_A_tierA/ring_property_ml_samples.csv
  --output …_fixed.csv` → Step 3 `predict_ring_properties_fixed.py --input …_fixed.csv
  --output ring_property_predictions.csv --pairs-output ring_pair_aromaticity_predictions.csv`。
  `predict_ring_properties_fixed.py` 文件头也写明 “**Requires the output of fix_ring_indices.py**”。
- 下游读取核实：`fig5_analysis.py:31-34` 与 `fig6_v2_main.py:63-66` 读取的都是
  `ring_pair_aromaticity_predictions_complete.csv` / `ring_pairs_ml.csv` /
  `spectator_ring_property_ml_samples_fixed.csv`，即**校正后索引产出的表**；
  `FIG5_ANALYSIS_REPORT.md` 亦注明 “Data fixed at: dearom_ring_pairs_A_tierA/ring_pair_aromaticity_predictions_complete.csv”。
- 旁观环（spectator）分支同样已修复并纳入：`fig6_analysis_v2/PANEL_B_TARGET_SPECTATOR/`
  下有 `spectator_ring_property_ml_samples_fixed.csv`（4.67 MB）及其 `_index_report.json` / `_index_errors.csv`，
  且被 `fig6_v2_main.py:66` 读取。

→ 因此本目录中 `ring_property_ml_samples_fixed.csv` 标注为「**已修复版本**」，
Fig.6 的成品数字建立在修复后的索引之上，**不存在「只有未修复版本」的情形**。

## 未复制 / 暂置清单

### 规则 2（含未修复 ring index 1-based bug）

- 已用 `grep -n "for idx in atom_on_ring"` 与 `grep -n "idx - 1"` 核查全部 22 个待复制 `.py`：
  **无一含 `[idx - 1 for idx in atom_on_ring]` 真实代码行**，故本目录无因规则 2 被排除的文件。
- 来源库中已知含该 bug 的 4 个文件（**不属于 uspto 管线，本就未复制**）：
  `/home/ubuntu/aroma-dps-code/{common,code_end,last_end_code,0831-end-code}/common/features.py`。

### 规则 4 — 原始反应大表（单文件 > 5 MB）

| 文件 | 大小 |
|---|---|
| `USPTO_STEREO_normalized.csv` | 138.5 MB |
| `USPTO_STEREO.csv` | 121.3 MB |
| `rxnmapper-main.zip`（第三方 atom-mapping 工具包） | 9.2 MB |
| `USPTO_50K.csv` | 5.5 MB |

### 规则 4 — 原子映射 / 筛选中间产物目录（整体未复制）

| 目录 | 大小 | 说明 |
|---|---|---|
| `dearom_stage2_full/` | 85 MB | Stage-2 全量输出：`stage2_reactions.csv` 23.3 MB、`mapped_reactions_cache.csv` 20.6 MB（原子映射缓存）、`rejected.csv` 18.5 MB、`stage2_ring_evidence.csv` 16.8 MB 均 > 5 MB；`tier_A_exact.csv` 2.10 MB、`tier_B_plausible.csv` 1.23 MB、`tier_C_review.csv` 1.48 MB、`stage2_report.json`、`decision_breakdown.csv`、`manual_review_sample.csv` 虽 < 5 MB，但属中间产物，按规则 4 一并排除 |
| `dearom_stereo_fast_full/` | 61 MB | `screened_candidates.csv` 22.1 MB、`candidate_rings.csv` 15.9 MB、`priority_3.csv` 15.9 MB、`priority_1.csv` 5.7 MB |
| `dearom_ring_pairs_A/` | 28 MB | 非 Tier-A 全环版本：`ring_pairs_ml.csv` 10.85 MB、`ring_pairs_ml.jsonl` 10.82 MB、`ring_property_ml_samples.csv` 5.85 MB |
| `dearom_50k_firstpass/` | 20 MB（207 文件） | `screened_reactions.csv` 10.5 MB + `review_images/` 人工复核结构图 |
| `dearom_stage2_pilot/` | 2.0 MB（10 文件） | pilot 规模试跑 |
| `dearom_stereo_fast_10k/` | 616 KB（6 文件） | 10k 子集调参 |
| `dearom_stereo_firstpass/` | 440 KB（15 文件） | 含 `review_images/` |
| `__pycache__/`（`fig6_v2_main.cpython-31{0,3}.pyc`） | — | 统一排除；`*.pyc`、`.ipynb_checkpoints/` 同 |

结果树内另有 **5 个 > 5 MB 单文件** 未复制（其余同目录文件均已复制）：

- `fig5_analysis/07_ring_family/ring_family_annotations.csv` — 6.48 MB
- `fig5_analysis/chemical_space_multiview/pca_nMCBO_reactant_product.png` — 6.36 MB
- `fig5_analysis/chemical_space_multiview/pca_NICSstar_reactant_product.png` — 6.25 MB
- `fig5_analysis/chemical_space_multiview/pca_HOMA_reactant_product.png` — 5.91 MB
- `fig6_analysis_v2/PANEL_B_TARGET_SPECTATOR/spectator_ring_property_predictions.csv` — 5.36 MB

另按任务清单排除：`python.txt`（12 KB）— 以 `.txt` 保存的 chemical-space 脚本草稿（420 行，
与 `fig5_chemspace.py` / `chemical_space_multiview.py` 均不相同），非 `.py` 管线脚本亦非说明文档，
故未复制，仅在此登记其存在。

### 复现缺口（重要）

1. 已复制脚本仍读取**未复制的中间产物**：`fig6_v2_main.py:69,100` 读 `dearom_ring_pairs_A/ring_pairs_ml.csv`
   （10.85 MB，规则 4 排除）；`fig6_v2_main.py:294` 读 `dearom_stage2_full/tier_A_exact.csv`（2.10 MB，
   按规则 4 的中间产物排除）。端到端重跑还需 `USPTO_50K.csv` / `USPTO_STEREO*.csv` 与 rxnmapper。
   建议这批表走 Zenodo/Release，在仓库内只留路径 + hash 的 manifest。
2. 模型权重未入仓：`predict_ring_properties_fixed.py` 与 `fig6_v2_predict_spectator.py` 依赖
   `/home/ubuntu/aroma-dps-code/best_model_package/`（`--model-package`）。
3. 图数据表：`fig5_analysis/07_ring_family/` 仅余 `ring_family_counts.csv` 与 `ring_family_delta_summary.csv`，
   逐样本的 `ring_family_annotations.csv` 因体积未复制，Fig.6d 环族图的原始标注需按脚本重建。

## 含绝对路径、待相对化的 .py 文件（14 个）

均位于本目录根（`/home/ubuntu/aroma-dps-code/uspto-5k` 硬编码为 `ROOT` 等常量）：

`fig5_analysis.py` · `fig5_chemspace.py` · `fig5_schematic_si.py` · `fig6_v2_data_audit.py` ·
`fig6_v2_main.py` · `fig6_v2_montage.py` · `fig6_v2_pair_inventory.py` · `fig6_v2_panel_a.py` ·
`fig6_v2_panel_f_ring.py` · `fig6_v2_reaction_family.py` ·
`fig6_v2_ring_x_reaction.py` · `chemical_space_multiview.py`（3 处，含具体输入表与输出目录） ·
`predict_ring_properties_fixed.py`（docstring 内写明期望的
`/home/ubuntu/aroma-dps-code/best_model_package/`；`--model-package` 为 `required=True` 无默认值） ·
`fig6_v2_predict_spectator.py`（L21 `ROOT`、L96 命令串中的 `/home/ubuntu/aroma-dps-code/best_model_package`）

未含绝对路径、可直接以 CLI 参数运行的 8 个：`build_ml_ring_pairs.py`、`dearom_screen.py`、
`dearom_screen_fast.py`、`dearom_stage2_exact.py`、`example_ml_loader.py`、`fix_ring_indices.py`、
`map_uspto_rxnmapper.py`、`summarize_manual_review.py`。
本目录内 **无文件引用 `/home/ubuntu/data_90`**。文档侧 `1README.md`（4 处）、`DEAROM_FAST_README.md`（3）、
`README_STAGE2.md`（3）、`README_RING_PAIRS.md`（1）、`fig6.txt`（5）、`fig62.txt`（3）、`python.txt` 未复制
亦含绝对路径，随脚本一并相对化。按要求，**本次迁移未作任何改动**。
