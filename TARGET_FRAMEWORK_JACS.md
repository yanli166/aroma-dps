# Aroma-DPS 目标框架设计（按 JACS 论文结构组织代码）

> 生成时间：2026-09-08
> 审核范围：`/home/ubuntu/aroma-dps/`（publication 仓库，git）、`/home/ubuntu/aroma-dps-code/`（23GB 工作目录，无 git，有运行中进程）、`/home/ubuntu/aroma-dps-code/0908-end-code/`（最新可解释性实验）
> 依据文件：`p0_verification/paper_result_dependency_manifest.csv`（Fig.3/4 ↔ 脚本官方映射）、`aroma-dps_vs_aroma-dps-code_diff_report.md`、`configs/default.yaml`、`experiments/fig3|fig4|fig5|fig6/config.yaml`

---

## 一、现状审核结论（三份代码的角色）

| 代码库 | 角色 | 问题 |
|---|---|---|
| `aroma-dps/`（git） | publication 快照：0831（Fig.3）+ 0901（Fig.4）+ unified_models + 新建的 src/experiments/tests/p0_verification | ① `src/aroma_dps/` 号称 single source of truth，但**只有 4 个模块真正实现**（任务别名、`chemistry/ring_mapping.py`、`data/splits.py`、`evaluation/metrics.py`），其余是 re-export 壳或 docstring 占位 stub；② `archive/` 是空壳（只有 README，旧文件未移动）；③ `code_end/` 已被 mapping 全部标记 archive/superseded 但仍留在仓库；④ README 停留在 Initial commit 状态，未提新结构；⑤ 数据、权重、结果全部不在仓库 |
| `aroma-dps-code/0831|0901` + `best_model_package/` + `baseline_*` + `nics_pipeline/` + `uspto-5k/` 等 | **实际产生论文数字的权威运行版**（Fig.3/4/5/6 全部） | 无 git；`unified_models` 靠外部路径 `/home/ubuntu/data_90/alldata_in_3090/model1/`；与仓库侧有 103 个文件差异（80 路径、2 个 ring-index 修复、5 个 split manifest 不一致 n_total=6923 vs 6820） |
| `aroma-dps-code/0908-end-code/` | 最新一轮：**合并重训 + l10 留出验证 + 三任务归因可解释性**（merged/l10val 两套方案，4 个 `.pt` 模型、3 任务 l10val 指标、IG+saliency 双归因图） | **完全未纳入 git 仓库**；merged 方案只跑完 HOMA（NICS/MBCO merged 模型缺失）；`train_homa_l10val.py` 保存预测时崩溃、靠 `resave_homa_l10val_preds.py` 抢救（bug 未修）；`plot_merged_figures.py` 未运行；`attention_l10val*/` 有旧选样残留图 |

**论文面板映射现状**（来自 paper_result_dependency_manifest.csv，权威）：Fig.3a–3i → `0831-end-code/` stage1–6；Fig.4a–4h → `0901-end-code/fig4_lunci10/`；Fig.5（策划反应验证）与 Fig.6（USPTO 反应级应用）在 `experiments/fig5|fig6/` 里**只有 config 没有 launcher**，其声明依赖的 `src/aroma_dps/inference/` 是 stub；0908 的可解释性工作在 manifest 中**尚无任何面板归属**。

---

## 二、目标框架（按 JACS 正文结构组织）

设计原则：**论文每个 Results 小节 / Figure 对应一个 `experiments/figN_*/` 目录，每个 SI 章节对应 `experiments/si_*/` 或 `data/`，Methods 对应 `src/aroma_dps/` 核心包**。所有代码收敛到 git 仓库一处，工作目录只保留数据与大体量结果。

```
aroma-dps/
├── README.md                      # 论文↔代码总映射 + 复现指南（需重写，见问题 P1-3）
├── pyproject.toml
├── configs/
│   └── default.yaml               # 全局协议（已有：任务名/种子/双轨 feature_mode/provenance 门禁）
│
├── src/aroma_dps/                 # ══ Methods（计算与实验方法）══
│   ├── chemistry/                 # ring_mapping.py ✅已实现 / ring_family.py ⚠️stub
│   ├── data/                      # splits.py ✅已实现 / datasets·manifests·scalers ⚠️stub
│   ├── featurization/             # graph.py(re-export) / features.py ⚠️stub
│   ├── models/                    # ring_conditioned + gat/gin/mpnn(backbones，并入 unified_models)
│   ├── training/                  # trainer(re-export) / cv·estar·checkpoint ⚠️stub
│   ├── evaluation/                # metrics.py ✅已实现 / ood·uncertainty ⚠️stub
│   └── inference/                 # ring_prediction·reaction_prediction ⚠️stub（Fig.5/6 依赖）
│
├── data/                          # ══ Data Availability（数据或 manifest+获取说明）══
│   ├── collet/                    # collet_{homa,nics,mbco}_0716.csv（训练集）
│   ├── lunci10/                   # lunci10_unified.csv / lunci10-test-corrected.csv（外部基准）
│   ├── merged_l10val/             # 0908 的 merged_*.csv 与 *_l10val.csv + manifest
│   └── reactions/                 # 汇总.xlsx / 汇总_stereo.xlsx（83 对去芳构化反应手工表）
│
├── models/                        # ══ 最终模型与推理 API ══
│   ├── best_model_package/        # 三任务最优权重 + AromaticityPredictor
│   └── merged_l10val/             # 0908 的 4 个 {TASK}_{merged,l10val}_best.pt + metrics.json
│
├── experiments/
│   ├── fig2_dataset/              # ══ Fig.2 数据集构建与描述符计算 ══
│   │   ├── nics_pipeline/         #   Gaussian/Multiwfn → HOMA/MBCO/NICS 后处理管线
│   │   ├── stereo_descriptors/    #   11 维立体描述符（ETKDG+MMFF 构象）
│   │   └── lunci10_construction/  #   38 环骨架×30 取代基：xTB→G16 构建流程
│   ├── fig3_model/                # ══ Fig.3a–3i 表征与环条件消融 ══（wrapper 已有）
│   │   ├── → 0831-end-code/ stage1–6 + stats + splits
│   │   └── interpretability/      #   【用户已定：Fig.3f】0908 全套 scripts/results/pending
│   ├── fig4/                      # ══ Fig.4a–4h 泛化 ══（wrapper 已有）
│   │   ├── launcher.py            #   → 0901-end-code/fig4_lunci10/
│   │   └── generalization/        #   新增：实际产出当前 Fig.4 图件的 fig4_*.py 驱动脚本
│   ├── fig5_application/          # ══ Fig.5 策划去芳构化反应验证 ══
│   │   └── curated_reactions/     #   6 个 generate_*.py + lunci8 + figures/ + 反应表
│   ├── fig6_uspto/                # ══ Fig.6 USPTO 反应级应用 ══
│   │   └── → uspto-5k 管线（筛选→原子映射→分析出图）
│   ├── si_baselines/              # ══ SI：基线与稳健性 ══
│   │   ├── baseline_traditional_ml/  # 9 种传统 ML × 3 任务
│   │   ├── baseline_gnn/             # 7 种通用 GNN backbone
│   │   ├── ring_encoding_ablation/   # label/mask/pool/combined 注入位置消融
│   │   ├── multiseed_cv/             # 5 种子 × 5 折 CV mean±std
│   │   └── common/                   # 指向修复版 features（见第六节副作用）
│   └── （fig7_interpretability 已取消，可解释性并入 fig3_model/interpretability）
│
├── tests/                         # 回归测试（已有：别名/ring 索引/划分泄漏/弃用警告/完整性门禁）
├── p0_verification/               # P0-1~4 审计 + paper_result_dependency_manifest（已有）
└── archive/deprecated/            # 废弃件（已实际移入，但 ⚠️ 仍被 Fig.3/4 运行时依赖，见第六节）
```

> 注：原设计的 `docs/` 尚未建立（P2-1 仍未做）。

---

## 三、框架各节点的代码来源与迁移方式

| 目标位置 | 来源（现状路径） | 迁移方式 | 状态 |
|---|---|---|---|
| `src/aroma_dps/chemistry/ring_mapping.py`、`data/splits.py`、`evaluation/metrics.py`、`__init__.py` | `aroma-dps/src/aroma_dps/` | 保留（已是权威） | ✅ 已实现 |
| `src/aroma_dps/models/` backbones | `aroma-dps/unified_models/{gnn,mpnn,gat,gin,graphsage}/model.py` + `unified_models/common/graphs.py` | **实体并入** src（消灭"仓库内 re-export 指向 legacy 目录"的倒挂） | ✅ 已并入：5 个 backbone + `CONV_LAYERS`，逐字节 verbatim |
| `src/aroma_dps/models/ring_conditioned.py` 实现 | `aroma-dps/0831-end-code/models/ring_conditioned_gnn.py`、`ring_readout.py` | 实体迁入 src（保留 ring_flag=10 消融定义，勿与最终 binary 方案统一） | ✅ 已迁入，并在 docstring 中显式恢复该约束（曾被 cp 覆盖丢失） |
| `src/aroma_dps/training/`、`featurization/features.py` | `aroma-dps/0831-end-code/common/{train_eval,features}.py`（0-based 修复版） | 实体迁入 src | ✅ 已迁入：`training/trainer.py`、`featurization/{features,graphs,graph_data,graph}.py`（features 取 **0-based 修复版**） |
| `src/aroma_dps/chemistry/ring_family.py` | `aroma-dps-code/fig4_ring_family_audit.py`（SMARTS 环族分类法） | 抽取为库函数 | ✅ 已实现（2 处修正 + 稠环 fusion_state 口径告警；单测覆盖苯/吡啶/呋喃/咪唑/环己烷/越界索引） |
| `src/aroma_dps/inference/ring_prediction.py` | `aroma-dps-code/best_model_package/predict.py`（AromaticityPredictor）+ `0901-end-code/fig4_lunci10/evaluation/run_external_delta.py`（ΔHOMA=HOMA_R−HOMA_P 等） | 合并实现 | ✅ 已实现：`RingAromaticityPredictor` 加载 3 权重 + `predict_reaction_pairs`（ΔA 约定入单测） |
| `src/aroma_dps/inference/reaction_prediction.py` | `aroma-dps-code/uspto-5k/dearom_screen*.py`、`dearom_stage2_exact.py`、`build_ml_ring_pairs.py` | 抽取核心函数入库，脚本留 experiments | ⚠️ 仍为显式 NOT-IMPLEMENTED 占位，docstring 指名 `experiments/fig6_uspto/` 真实脚本 |
| `experiments/fig3_model/` | `aroma-dps/experiments/fig3/`（已有 launcher）→ `0831-end-code/` stage1–6 + `stats/significance_test.py` | 保留现状，统计脚本纳入 | ✅ 就绪；另已迁入 0908 可解释性到 `interpretability/` |
| `experiments/fig4/` | `aroma-dps/experiments/fig4/`（已有 launcher）→ `0901-end-code/fig4_lunci10/`；**另新增 `generalization/`** ← `aroma-dps-code/fig4_*.py`（6 个驱动脚本） | 保留现状 + 迁入驱动脚本 | ⚠️ 代码树与当前图件**不同代**，驱动脚本未接 launcher（见 P0-5） |
| `experiments/fig2_dataset/nics_pipeline/` | `aroma-dps-code/nics_pipeline/`（01–07 配置驱动管线） | 整体迁入（相对路径化） | ✅ 已迁入 10 文件（含 run_all.sh），逐字节核对；仍含绝对路径 |
| `experiments/fig2_dataset/stereo_descriptors/` | `aroma-dps-code/compute_stereo_descriptors.py` | 迁入 | ✅ 已迁入 2 文件；仍含绝对路径 |
| `experiments/fig2_dataset/lunci10_construction/` | `aroma-dps-code/lunci10/`（run_g16_parallel.py 等流程脚本，**不含** xtb_output/gaussian_inputs 大文件） | 仅迁脚本 | ✅ 已迁入脚本 + 2 个 begin.csv；`1_xtb.py` 取自 lunci9（非原件，已在 README 注明） |
| `experiments/fig5_application/curated_reactions/` | `aroma-dps-code/generate_aroma_figures.py`、`generate_cada_figures.py`、`generate_scatter_figures.py`、`generate_fsp3_npr1_figures.py`、`generate_nics_violin_by_ring.py`、`generate_single_nics_figures.py` + `lunci8/predict_lunci8.py` | 迁入 + 统一数据入口指向 `data/reactions/汇总_stereo.xlsx` | ✅ 已迁入 51 MB / 89 文件（6 个 generate_*.py + figures + lunci8），338 文件 cmp 通过 |
| `experiments/fig6_uspto/` | `aroma-dps-code/uspto-5k/`（`dearom_screen_fast.py`→`dearom_stage2_exact.py`→`build_ml_ring_pairs.py`→`fix_ring_indices.py`→`fig5_analysis.py`/`fig6_v2_*.py`） | 迁入管线脚本（确认 `fix_ring_indices.py` 修复后的最终版） | ✅ 已迁入 148 MB / 249 文件；`fix_ring_indices.py` 口径见 P0-6 |
| `experiments/fig7_interpretability/` | `aroma-dps-code/0908-end-code/scripts/` 13 个脚本（prepare_merged_data→train_merged→eval_old_baseline→saliency；prepare_*_l10val→train_*_l10val→saliency_l10val） | 迁入 + 修 `train_homa_l10val.py` 保存预测崩溃 bug（合并 resave 抢救逻辑）+ 清理 attention_l10val 旧选样残留 | ✅ 已按用户决定迁入 `experiments/fig3_model/interpretability/`（scripts 13 + results 57 文件（含 33 PNG）+ pending） |
| `experiments/si_baselines/` | `aroma-dps-code/baseline_traditional_ml/`、`baseline_gnn/`、`ring_encoding_ablation/`、`new5zhecv/` | 迁入（相对路径化） | ⚠️ 已迁入 4 层基线 + results CSV；**副作用**：Layer-1 汇总 CSV 产出自含 bug 特征，用修复版 features 重跑不可复现 |
| `models/best_model_package/` | `aroma-dps-code/best_model_package/`（model_arch.py、graph_utils.py、train_final.py、predict.py + 3 权重） | 迁入；权重走 Release/Zenodo 而非 git | ✅ 已迁入 9 文件（3 个 .pt + metrics.json + 修复版 graph_utils.py） |
| `models/merged_l10val/` | `aroma-dps-code/0908-end-code/models/` 4 个 `.pt` + `results/*_metrics.json` | 同上 | ✅ 0908 的 4 个 .pt 已迁入 `models/interpretability/`；merged NICS/MBCO 缺失，用户将补训（P0-2 保持） |
| `data/*` | `aroma-dps-code/code_end/data1_end/collet_*_0716.csv`、`lunci10/lunci10_unified.csv` 等、`0908-end-code/data/`、`汇总.xlsx`/`汇总_stereo.xlsx` | 小 CSV/xlsx 直接入仓或 Zenodo；git 只放 manifest+hash | ✅ 已迁入 `data/{collet,lunci10,lunci_external,reactions,interpretability}`（7.6 MB），collet 三表与 data1_end 字节相同 |
| `archive/deprecated/` | `aroma-dps/code_end/`（mapping 已全标 archive）+ `unified_models/ml_cross_task.py` + 0831 内弃用脚本（`run_pretrain_eval.py`、`cross_arch_eval.py`、`run_multiseed.py`、`gnn_baseline.py`）+ `0901` 的 `run_external_absolute.py` v1 | **实际移动**到 archive/（目前是空政策） | ✅ 已移动（0831 v1 生成层 / 0901 v1 评估 / ml_cross_task / code_end 整簇）；⚠️ 带来新的运行时耦合，见第六节 |
| 不迁入（留工作目录） | `aroma-dps-code/last_end_code/`（0831 的前代）、`lunci3/4_mol_images/`、`lunci7_8_out_nics/`、`lunci8[-3]/lunci9*` 计算中间产物、`results*`（20+GB）、工作汇总 md（个人日志） | 不入仓；结果按 manifest 路径引用 | — |

---

## 四、本文档的内容、格式与方法说明

**内容**：第一节是现状审核结论；第二节是目标目录树（每个节点标注对应的论文章节）；第三节是"目标位置 ← 来源 → 迁移方式 → 状态"映射表，这是后续执行的工单；第五节是问题清单。

**格式约定**：

- 路径一律用仓库相对路径或绝对路径原文，不缩写；
- 状态标记：✅ 已就绪 / ⚠️ 部分（壳、stub、未执行的政策）/ ❌ 未开始；
- 面板编号（Fig.3a 等）以 `p0_verification/paper_result_dependency_manifest.csv` 为唯一权威，不另起炉灶；新增的可解释性章节暂编号 fig7，**最终是正文图还是 SI 需你拍板**；
- 迁移方式分四类：保留 / 实体迁入 src（消灭 re-export 倒挂）/ 迁入 experiments（脚本相对路径化）/ 移入 archive。

**审核方法**：三路并行摸底——(1) git 仓库全量结构 + src 逐模块行数核查（区分真实实现与 stub）+ 同名文件跨目录抽查对比；(2) 0908-end-code 逐脚本/日志/产物梳理，重建"数据准备→重训→评估→归因→出图"完整链条；(3) aroma-dps-code 顶层 fig4_*.py、generate_*.py、各 lunci* 目录、xlsx 数据表逐一确认职责。面板级事实一律回溯到 manifest 与 config，不靠推断。

---

## 五、问题清单（按严重度分级）

### P0（阻断论文完整性，必须先解决）

1. **0908 可解释性工作完全游离在 git 仓库与 manifest 之外**。三任务 l10val 指标（HOMA R²=0.962、NICS R²=0.975、MBCO R²=0.995）和 IG 归因图是"模型看环内而非取代基"这一核心论证的唯一证据链，却没有面板归属、没有 provenance 记录。需：迁入仓库 → 在 manifest 注册面板 → 补 ProvenanceRecorder。
2. **merged 重训只完成 HOMA**。`NICS_1zz_merged_best.pt`、`MBCO_merged_best.pt` 不存在，而 `saliency_merged.py` 引用了它们；`plot_merged_figures.py` 设计的三任务图未生成。要么补齐训练，要么在论文叙事中明确 merged 方案只用于 HOMA 论证。
3. **两套 split manifest 不一致**（n_total=6923 仓库 vs 6820 工作目录）。论文最终数字绑定哪套划分必须拍板，否则 Fig.3 全部面板的测试集定义含糊。
4. **`results_v2/` 与 `results_v2_backup_pre_bugfix/` 并存**。manifest 中 Fig.3c/3d/3f/3g/3i 指向 backup_pre_bugfix 目录，需确认这是 ring-index 修复前的结果是否仍有效（修复不影响训练，但影响描述符），并在文档中写明依据。

5. **Fig.4 代码树与当前图件不同代，且有面板根本没有结果**。
   `0901-end-code/fig4_lunci10/` 写向 `results/fig4_lunci10_final/`，而该目录中
   `02_novelty/`、`03_pairwise/`、`06_position/` 为空、`05_hammett/` **不存在**；
   论文当前 Fig.4a–4d 实际来自 `results/fig4_lunci10_final_v2/`（09-08，`fig4_rerun_seed11.py`）
   与 `fig4_generalization_retrain_v4_noe/`（09-08 16:14）。这批驱动脚本已复制到
   `experiments/fig4/generalization/` 但**未接入 `launcher.py`**，且仍硬编码工作目录。
   → manifest 中 Fig.4 的 source_script / result_dir 需按实际重写。
6. **Fig.5 的 `lunci8/predict_lunci8.py` 环索引 off-by-one（与你决定④直接冲突）**。
   该脚本 `atom_on_ring_1idx = [idx + 1 for idx in ring]` 并注明"1-indexed 与训练数据一致"，
   但它 import 的 `unified_models/common/graphs.py::Graph` 明确按 **0-based 直接使用**
   （`graphs.py:83-84`；P0-2 审核结论亦为 0-based）。即目标环标记整体错位一个原子。
   另：它用 `baseline_gnn` 权重（非最终 RC 模型），且 `sys.path` 优先指向工作目录。
   **实测证据**（`c1ccccc1C(=O)O`，真实环原子 0–5）：0-based 输入被标记 `[0,1,2,3,4,5]`，
   1-based 输入被标记 `[1,2,3,4,5,6]` —— 环原子 0 丢标记、非环羧基碳 6 被误标。
   已在 `tests/test_regression.py` 加锁该端到端口径（`Graph` 标记原子必须等于传入索引）。
   → 索引口径修正并重跑前，其产物**不得作为 Fig.5 依据**；已就地标注，未擅自改动。
7. **Fig.3f 面板字母冲突**：`paper_result_dependency_manifest.csv` 中 3f = mask pretraining
   （`run_mask_pretrain_v2.py`，publication=True），而你把可解释性也定为 3f。
   需二选一：可解释性改 3j/新字母，或 mask pretraining 移位并同步 manifest + launcher。
   （关联：本轮已把 `run_all_v2_parallel.sh` Stage 3 从非发表的 `run_pretrain_eval.py`
   改回 `run_mask_pretrain_v2.py`，若移位需一并复核既有 `results_v2/stage3/` 出自哪个脚本。）
8. **可解释性论证样本量过小**：三任务 `*_attribution_stats.csv` 各仅 **6 个样例**。
   ring/substituent 归因比中位数（IG）HOMA 24.8 / NICS 3.4 / MBCO 14.8，
   但 **NICS 有 1/6 例 `ig_ring_over_sub = 0.73 < 1`**，即"模型看环不看取代基"在 NICS 上并非逐例成立。
   → 扩大样例数并给分布统计，或明确改用 GI 口径（GI 三任务全部 >1，min 2.4）。

### P1（结构债，影响可复现性与投稿）

1. **`src/aroma_dps/` 名不副实**：宣称 single source of truth，实际仅 4 个模块落地；Fig.5/6 的 config 指向的 `inference/` 是 stub，launcher 不存在。建议按第三节映射表把实现实体迁入 src，让 legacy 目录变成纯历史。
   → ✅ **本轮已处理**：backbones/readout/ring_conditioned/features/graphs/graph_data/trainer 实体入 src，
   `chemistry/ring_family.py`、`inference/ring_prediction.py` 新实现；仅 `inference/reaction_prediction.py` 仍为显式占位（Fig.6 API 未入库）。
2. **`archive/` 政策空转**：`code_end/` 已被 mapping 全部标记 archive/superseded 却原地保留，0831 内的弃用脚本（canonical_splits 系列）也在原处。要么真移动，要么放弃 archive 目录——维持现状只会让审稿代码审计时混淆。
   → ✅ **已实际移动**，但 ⚠️ **产生新问题（P1-6）**：publication 代码在运行时依赖 `code_end/`。
3. **README 过时**：未提 src/experiments/tests/p0_verification，Usage 只有两条 legacy 命令。需重写为"论文面板→命令→产物路径"的复现指南，并补 Data/Code Availability 声明（JACS 硬性要求，建议权重+数据走 Zenodo DOI）。
   → ✅ **已重写**为面板↔代码↔命令对照表 + Data/Code Availability；Zenodo DOI 与 LICENSE 仍待你决定。
4. **依赖外部绝对路径**：工作目录版 `ORIG_MODELS_ROOT=/home/ubuntu/data_90/alldata_in_3090/model1/`。unified_models 已复制入仓是好事，但要统一所有脚本只认仓库内副本。
5. **0908 的两个代码质量问题**：`train_homa_l10val.py` 保存预测时 `atom_on_ring` KeyError 靠抢救脚本绕过（bug 未修）；`attention_l10val/` 等目录混有旧选样残留图（13:19 批次），入仓前需清理。
6. **归档并不"只读"**：Fig.3/Fig.4 有 20 个模块把 `code_end` 加进 `sys.path`、并读 `code_end/data1_end/*.csv`
   与 `code_end/layer4_substituent/code/hammett_constants.py`。为让归档动作不破坏运行，本轮把 39 处路径改指
   `archive/deprecated/code_end/`，并把 `data1_end`(1.9 MB) 与 layer2/3 CSV(17 MB) 补回归档。
   → 应改为让 Fig.3/4 依赖 `src/aroma_dps` + `data/`，归档才能真正变成历史（见 6.3）。
7. **`.gitignore` 使本轮迁入的数据与权重不被 git 跟踪**：仓库忽略 `*.csv`/`*.xlsx`/`*.pt`/`*.pth`，
   实测 `git status --ignored` 有 **208 条路径 / 305 MB 对 git 完全不可见**
   （含 `data/` 7.6 MB、`models/` 13 MB、`0901-end-code/results/` 69 MB、各 `experiments/*/results/`）。
   发布前必须三选一：whitelist 关键文件、走 git-lfs、或外部仓储 + DOI。
   当前 git 侧只记录了 66 个修改 + 98 个删除（archive 移动）+ 22 个新目录。
8. **50 个 .py 仍硬编码 `/home/ubuntu/aroma-dps-code`**（fig6 14 / fig3_model 10 / fig5 9 / fig4 6 /
   si_baselines 3 / fig2 3 / best_model_package 2），另有 `predict_lunci8.py` 等指向
   `/home/ubuntu/data_90/alldata_in_3090/model1`。本轮为保证"逐字节可追溯"未改；
   改动清单已写入各目录 README。
9. **SI Layer-1 基线的可复现性副作用**：`experiments/si_baselines/common/features.py` 用的是**修复版**
   （0-based），而入仓的 `all_ml_summary.csv` 产出自含 bug 特征的旧版；用现仓库代码重跑**不可复现该 CSV**。

### P2（建议项）

1. 中文任务书/日志（`fig4.txt`、`ODD.txt`、`工作汇总_*.md`、`四阶段实验框架详细展开.md`）建议收敛到 `docs/`，工作日志类不进 publication 仓库。
2. `lunci6_取代基泛.txt` 的配对 Δ 训练方案（paired_delta_v5）与当前 Fig.4f 的推理期 Δ 是两条路线，若论文不用 paired_delta 训练，应在 archive 中标注，避免误引。
3. 统计检验（`stats/significance_test.py`）没有独立结果目录，建议在 fig3 wrapper 中固定其输入只读 v2 协议产物（manifest 已有此要求，需 launcher 强制）。

---

## 六、执行记录（2026-09-08 本轮"复制粘贴"）

### 6.1 你的四项决定如何落地

| 决定 | 执行情况 | 遗留 |
|---|---|---|
| ① 可解释性放 **Fig.3f** | 0908 全套迁入 `experiments/fig3_model/interpretability/{scripts,results,pending}`（13 脚本 / 57 结果文件含 33 PNG / 5.5 MB）+ `data/interpretability/`(5.2 MB, 11 文件) + `models/interpretability/`(7.0 MB, 4 个 `.pt`)，逐字节 `cmp` 通过 | manifest 中 3f 仍写着 mask pretraining → **面板字母冲突待你重排**（P0-7） |
| ② merged 模型你补训 | **未生成任何假结果**。缺的 `NICS_1zz_merged_best.pt` / `MBCO_merged_best.pt` 及其图件登记在 `interpretability/pending/`，`plot_merged_figures.py` 未运行 | P0-2 保持开放 |
| ③ split manifest 6923 vs 6820 你之后补 | **一个 split 文件都没动** | P0-3 保持开放 |
| ④ **ringdex 只有修复后才能放进去，否则不要放** | 全仓 4 份 `features.py`（`0831-end-code/common`、`experiments/si_baselines/common`、`src/aroma_dps/featurization`、`archive/deprecated/code_end/common`）+ `models/best_model_package/graph_utils.py` + `unified_models/common/graphs.py` 逐一核查：**均为 0-based 修复版**；含 `idx - 1 for idx in atom_on_ring` 的 1-based 变体（工作目录 4 份）**未入仓** | ⚠️ 但发现 `lunci8/predict_lunci8.py` 自行构造 1-based 索引 → 见 P0-6 |

### 6.2 迁入总量与校验方式

| 目标 | 文件数 | 体积 | 校验 |
|---|---|---|---|
| `experiments/{fig2_dataset,si_baselines}` + `data/*` + `models/best_model_package` | 72 | 8.1 MB | 全部 `cmp` |
| `experiments/fig5_application/` + `experiments/fig6_uspto/` | 338 | 199 MB | 全部 `cmp` |
| `experiments/fig3_model/interpretability/` + `data/interpretability` + `models/interpretability` | 13+57+4+11 | 17.7 MB | 全部 `cmp` |
| `0831-end-code/results_v2/` + `results_v2_backup_pre_bugfix/`（Fig.3 结果，此前仓库为 0 文件；2026-09-09 复查补做） | 344 | 5.9 MB | `diff -r` 与工作目录逐字节一致 |
| `0901-end-code/results/`（Fig.4 结果，此前仓库为 0 文件） | 131 | 69 MB | 数量一致 + `cmp` |
| `archive/deprecated/code_end/{data1_end,results}`（补归档自洽） | 1198 | 19 MB | 全部 `cmp` |
| `experiments/fig4/generalization/`（6 个 `fig4_*.py` 驱动） | 6 | 162 KB | 全部 `cmp` |
| `src/aroma_dps/`（实体化，非复制） | — | — | `pytest **32 passed**` + 推理端到端冒烟（对 2 条训练表内记录 HOMA 误差 0.017 / 0.007） |

### 6.3 本轮不得不动的"非搬运"修复（只碰路径与导入，无算法改动）

1. **`_PROJ_ROOT` 占位符还原（49 处 / 44 文件）**：仓库把绝对路径相对化时，把变量写成了字符串字面量
   （`PROJ_ROOT = "_PROJ_ROOT"`、`Path("_PROJ_ROOT/0901-end-code/…")`、
   `ORIG_MODELS_ROOT = "_PROJ_ROOT + "/unified_models""`），还原为其上方已正确计算的 `_PROJ_ROOT`。
   其中 `training/check_internal_pair.py` 与 `training/train_internal_pair_models.py` 原本是**硬 SyntaxError**（引号嵌套），
   即这两个 Fig.4 脚本在仓库里从未可运行。现除归档内的 `0831_v1_generation/gnn_baseline.py`
   （已知不可运行的伪代码草稿，政策上不改不删）外，`compileall` 全树通过。
   *该缺陷在 Initial commit 即存在（已核对 `git show HEAD`），非本轮搬运造成。*
2. **`code_end` 路径改指归档（39 处）**：`code_end/` 移入 `archive/deprecated/` 后，
   Fig.3/Fig.4 的 `sys.path`、`data1_end`、`layer4_substituent` 全部悬空（这也是此前"15/46 模块
   `No module named 'common'`"的根因）。改指后 **46/46 Fig.4 模块导入通过**。
3. **`0831-end-code/common/tasks.py`**：`DATA1_END_DIR` 回退值同步指向归档；
   已验证三任务 `dataset_path` 全部解析到真实文件。
4. **`check_internal_pair.py`**：`from rdkit.Chem import MurckoScaffold` → `rdkit.Chem.Scaffolds`（当前 rdkit 顶层不导出）。
5. **`hammett_analysis.py`**：删除 `HAMMETT_SIGMA_META, HAMMETT_SIGMA_PARA` 两个**全仓库无定义、本文件也未使用**的导入
   （此前该脚本无法 import，`05_hammett/` 也因此不存在）。
6. **`src/aroma_dps/config.py` 的 `REPO_ROOT` 多退了一层目录**（3 个 `os.pardir` → 2 个）：
   实测 `REPO_ROOT` 原为 `/home/ubuntu`，导致 `COLLET_DIR` / `MODELS_DIR` /
   `inference/ring_prediction.DEFAULT_PKG_DIR` 全部指到仓库外，
   `RingAromaticityPredictor()` 直接 `ModuleNotFoundError: graph_utils`。
   → 已修并加 `TestConfigPaths` 锁定（该缺陷是我本轮 vendoring 时引入的，非搬运内容自带）。

### 6.4 暂置清单（无产物或需你补，未伪造）

- `NICS_1zz_merged_best.pt`、`MBCO_merged_best.pt` 与 `plot_merged_figures.py` 产物（→ 决定②）
- Fig.4 空面板：`02_novelty/`(0 文件)、`03_pairwise/`(0)、`06_position/`(0)、`05_hammett/`(**目录不存在**)
- `layer3_ring_fixed` 的 375 个 `best_model.pth`（291 MB，重跑 `train_l10_exposure.py` 才需要）
- `src/aroma_dps/inference/reaction_prediction.py`（Fig.6 的入库 API，仍为显式占位）
- `lunci10/gaussian_inputs/*.chk`（60 个，最大 6.7 MB）留工作目录
