# Fig.5 — 策划去芳构化反应验证（curated reactions）

## 论文面板归属

本目录承载论文 **Fig.5 "Curated real-reaction DFT validation"** 的全部绘图代码与成品图：
基于手工整理的 83 对反应物/产物去芳构化反应（166 行 = 83 pair），
展示反应前后芳香性指标变化（ΔHOMA = HOMA_R − HOMA_P、ΔMBCO = MBCO_R − MBCO_P、
ΔNICS = NICS_P − NICS_R，三者统一为「正值 = 芳香性损失」）以及芳香性与立体/构象 descriptor 的耦合关系。

注意：本目录与 `experiments/fig6_uspto/` 中同名的 `fig5_analysis.py` / `fig5_analysis/` **无关**。
后者是 USPTO 管线早期编号遗留，实际归属 Fig.6 面板，详见 `experiments/fig6_uspto/README.md`。

## 来源与迁移

- 来源绝对路径：`/home/ubuntu/aroma-dps-code/`
- 迁移日期：**2026-09-08**
- 迁移方式：**原样复制**（`cp -p`），未改动任何一行代码、未调整任何路径、未重新格式化；
  全部复制文件已与来源逐字节 `cmp` 校验一致。

## 已复制清单（共 89 个文件）

### 1. 绘图脚本（6 个，位于本目录根）

| 文件 | 职责 |
|---|---|
| `generate_aroma_figures.py` | Aroma_Fig01–07：芳香性主图（配对斜率、按反应类型/环族、小提琴、Δ 分布） |
| `generate_cada_figures.py` | Fig01–Fig13：CADA 方案主图（reactant vs product、Δ by type/ring、descriptor 相关、形状空间、组成矩阵、indole 专题、3D 补充） |
| `generate_scatter_figures.py` | Fig14–Fig20：全局/indole stereo vs aroma 散点与 focus 图（RPD/PBF vs HOMA） |
| `generate_fsp3_npr1_figures.py` | Clean_Fig01–06：Fsp3 / NPR1 与芳香性关系、shape space |
| `generate_nics_violin_by_ring.py` | `Single_NICS_violin_by_ring.png`：按环类型分组的 NICS 小提琴图 |
| `generate_single_nics_figures.py` | Single_Fsp3_vs_NICS.png、Single_NPR1_vs_NICS.png |

### 2. 绘图方案任务书

- `参考1.txt`（原样保留来源文件名）：CADA 19 图方案的**任务书/绘图方案原文**，
  即本组 `generate_*.py` 各图的构图与统计口径来源（含 166 行 / 83 pair 的配对约定与 Δ 符号约定）。
  归属文档，非代码。

### 3. `figures/` — 成品出版级图（36 张 PNG）+ 配对数据表

- `Aroma_Fig01_global_paired_slope.png` … `Aroma_Fig07_delta_distribution.png`（7 张）
- `Clean_Fig01_Fsp3_vs_aroma_global.png` … `Clean_Fig06_shape_space_indole.png`（6 张）
- `Fig01_reactant_vs_product.png` … `Fig20_indole_focus_RPD_vs_HOMA.png`（20 张，CADA 方案系列）
- `Single_Fsp3_vs_NICS.png`、`Single_NPR1_vs_NICS.png`、`Single_NICS_violin_by_ring.png`（3 张）
- `paired_data.xlsx`（36 KB，83 对反应的配对长表，多数脚本的直接输入）
- 全部单张均 < 5 MB（最大 `Aroma_Fig03_by_ring_family.png` 2.64 MB），无一被体积规则排除。

### 4. `lunci8/` — 模型预测反应物/产物芳香性 + stereo↔aroma 关联

脚本（3 个）：

- `predict_lunci8.py` — 用 5 种 GNN backbone 对该组反应物/产物逐环预测 HOMA / NICS_1zz / MBCO
- `plot_stereo_vs_aroma.py` — 立体/构象 descriptor vs 预测芳香性（V1–V6，输出 `plots/`）
- `plot_stereo_v2.py` — descriptor v2 扩展版（V1–V7，输出 `plots_v2/`、`plots_v3/`）

结果：

- `lunci8-predicted.csv`（12 KB，模型预测输出表）
- `plots/` — 6 张 PNG + `lunci8_with_stereo_metrics.csv`
- `plots_v2/` — 16 张 PNG + `lunci8_with_stereo_v2.csv`
- `plots_v3/` — 16 张 PNG + `lunci8_with_stereo_v2.csv`

## 未复制 / 暂置清单

### 规则 2（含未修复 ring index 1-based bug 的文件）

- 本目录**所有已复制文件均不含** `[idx - 1 for idx in atom_on_ring]` 这一真实代码行，
  已用 `grep -n "for idx in atom_on_ring"` 逐个核查全部待复制 .py。
- 来源库中已知含该 bug 的文件（**不在本目录复制范围内，故未复制**）：
  - `/home/ubuntu/aroma-dps-code/common/features.py`（第 133 行为真实 bug 代码）
  - `/home/ubuntu/aroma-dps-code/code_end/common/features.py`（第 116 行）
  - `/home/ubuntu/aroma-dps-code/last_end_code/common/features.py`（第 111 行）
  - `/home/ubuntu/aroma-dps-code/0831-end-code/common/features.py`（第 174 行）
- 需排除误判的一处（**不是 `features.py` 那行代码，因此相关文件已复制**）：
  - `plot_stereo_v2.py:303` 是 `ring_0idx = [i - 1 for i in ring_atoms_1idx if i > 0]`，
    是为 RDKit 取坐标而做的 1-based→0-based 转换，与 `features.py` 的环标记错位 bug 无关。

#### ⚠️ 复查更正：`predict_lunci8.py` 有端到端 off-by-one（不属于上面的"误判"）

上一轮只按"是否含 `[idx - 1 for idx in atom_on_ring]` 字面代码"筛查，因而放过了它；
按数据流端到端追踪后确认它确实是**同类错误**（框架文档 P0-6）：

| 环节 | 事实 |
|---|---|
| `predict_lunci8.py:171` | `atom_on_ring_1idx = [idx + 1 for idx in ring]`，注释称"1-indexed 与训练数据 atom_on_ring 一致" |
| 该注释是否成立 | **不成立**：P0-2 审核（20,605 行）确认训练 CSV 的 `atom_on_ring` 是 **0-based** |
| 实际消费的类 | `predict_lunci8.py:54` `from unified_models.common.graphs import Graph`；`graphs.py:83-84` 注释"CSV atom_on_ring 已是 0-based, 直接使用"，**不做任何 -1 还原** |
| 后果（实测） | 以 `c1ccccc1C(=O)O` 为例，真实环原子为 0–5：0-based 输入 → 被标记原子 `[0,1,2,3,4,5]`；该脚本的 1-based 输入 → `[1,2,3,4,5,6]`。**环原子 0 丢失标记，非环的羧基碳 6 被误标为目标环原子**（`ring_indices` 属性同样整体右移一位） |
| 其他两点 | 它加载的是 `baseline_gnn/results/...` 权重（**非**最终 RC 模型）；`sys.path` 首位为 `/home/ubuntu/aroma-dps-code`，其次 `/home/ubuntu/data_90/...`，在仓库内无法直接运行 |

→ **`lunci8-predicted.csv` 与 `plots*/` 中依赖该预测的图，在索引口径修正并重跑前不得作为 Fig.5 依据。**
本目录仍原样保留该脚本（它是既有产物的 provenance），未擅自修改。

### 规则 4（体积/中间产物排除）

- `lunci8/gaussian_inputs/`（2.2 GB，425 个文件）— Gaussian 输入与 scratch，未复制
- `lunci8/xtb_output/`（42 MB，138 个子目录）— xtb 逐分子输出，未复制
- `lunci8/plots_lunci3/`（18 MB，17 个文件）— 由 **未列入本组清单** 的
  `lunci8/predict_and_plot_lunci3.py`（41 KB）生成，脚本与产物一并暂置
- `lunci8/xyz_files/`（540 KB）— 3D 坐标中间产物，未复制
- `lunci8/1_xtb.py`（9.4 KB）、`lunci8/run_xtb_batch.sh`（1.4 KB）— xtb 计算脚本，非本组 Fig.5 代码，未复制
- `lunci8/predict_and_plot_lunci3.py`（41 KB）— lunci3 系列脚本，任务清单未纳入，未复制
- `lunci8/lunci8-begin.csv`（1.8 KB）、`lunci8/lunci3-all-mark-3.csv`（20 KB）、
  `lunci8/lunci3-all-mark-4.csv`（9.4 KB）— 输入/标注表，未复制
- `__pycache__/`、`*.pyc`、`.ipynb_checkpoints/` — 全库统一排除

### 依赖与待核实项（重要）

1. **数据表位置**：`generate_*.py` 与 `参考1.txt` 的数据源是 `汇总_stereo.xlsx`（166 行 = 83 对）
   及 `汇总.xlsx`，按规划**不在本目录**，位于仓库 `data/reactions/`（由数据搬运同事负责，非本次迁移范围）。
   本次搬运期间该同事已交付，已核实
   `data/reactions/汇总_stereo.xlsx` 与 `data/reactions/汇总.xlsx`
   与来源 `/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx`、`/home/ubuntu/aroma-dps-code/汇总.xlsx`
   **逐字节一致**（2026-09-08）。本目录 `figures/paired_data.xlsx` 为派生的配对长表，非原始手工表。
2. `plot_stereo_vs_aroma.py:52` 与 `plot_stereo_v2.py:54` 读取的是
   `/home/ubuntu/aroma-dps-code/lunci8/lunci8-predicted-draw.csv`（4.1 KB），
   **不在任务复制清单内故未复制**；复现这两个脚本前需一并补齐该表。
3. `predict_lunci8.py` 的运行依赖未入仓：
   - `PROJ_ROOT = /home/ubuntu/aroma-dps-code`、
     `ORIG_ROOT = /home/ubuntu/data_90/alldata_in_3090/model1`（外部 `unified_models` 包）
   - 权重 `/home/ubuntu/aroma-dps-code/baseline_gnn/results/{HOMA/MPNN,NICS_1zz/GNN,MBCO/GraphSAGE}/best_model.pth`
     （按规则 4「逐折权重目录」未复制，权重走 Release/Zenodo）
4. **索引约定待核实（未改动代码，仅记录）**：`predict_lunci8.py:170-171` 明确以
   **1-based** 构造 `atom_on_ring_1idx` 并传给 `Graph(...)`，注释称「与训练数据一致」；
   而运行时实际解析到的 `/home/ubuntu/data_90/alldata_in_3090/model1/unified_models/common/graphs.py:83-84`
   写的是「CSV atom_on_ring 已是 0-based, 直接使用」（`target_ring_atoms = list(self.atom_on_ring)`），
   仓库内 `src/aroma_dps/featurization/graphs.py` 与 `unified_models/common/graphs.py` 亦为 0-based 直接用法。
   两处约定方向相反，需作者确认 `lunci8` 预测结果的环标记是否受到影响；本目录代码保持来源原样，未作修改。

## 含绝对路径、待相对化的 .py 文件（9 个）

| 文件（相对本目录） | 绝对路径出现处 |
|---|---|
| `generate_aroma_figures.py` | L35 `汇总_stereo.xlsx`、L36 输出 `figures` |
| `generate_cada_figures.py` | L5(docstring)、L37 `汇总_stereo.xlsx`、L38 输出 `figures` |
| `generate_scatter_figures.py` | L30 `汇总_stereo.xlsx`、L31 输出 `figures` |
| `generate_fsp3_npr1_figures.py` | L26、L27 |
| `generate_nics_violin_by_ring.py` | L26、L27 |
| `generate_single_nics_figures.py` | L26、L27 |
| `lunci8/predict_lunci8.py` | L10-12 权重路径、L44 `PROJ_ROOT`、L45 `ORIG_ROOT`(`/home/ubuntu/data_90/...`)、L308、L311 |
| `lunci8/plot_stereo_vs_aroma.py` | L52 `CSV_PATH`、L53 `OUT_DIR` |
| `lunci8/plot_stereo_v2.py` | L54 `CSV_PATH`、L55 `OUT_DIR` |

上述路径均指向 `/home/ubuntu/aroma-dps-code`（或 `/home/ubuntu/data_90`），
入仓后需改为仓库相对路径或可配置入口，但**本次迁移按要求未作任何改动**。
