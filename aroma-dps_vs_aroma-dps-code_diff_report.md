# aroma-dps vs aroma-dps-code 目录对比报告

> 生成时间: 2026-09-07  
> 审计人: 科研软件工程师

---

## 一、结论：哪个是"正确"的？

**两者都"正确"，但用途完全不同。**

| 维度 | `aroma-dps/` (GitHub 仓库) | `aroma-dps-code/` (工作目录) |
|------|---------------------------|----------------------------|
| **定位** | Publication repository | 实验训练工作目录 |
| **路径** | `/home/ubuntu/aroma-dps/` | `/home/ubuntu/aroma-dps-code/` |
| **大小** | 11 MB | 23 GB |
| **.py 文件数** | 233 | 283 |
| **Git** | 有 (5 commits) | 无 |
| **数据文件** | 无原始数据 | 有 (`code_end/data1_end/` 等) |
| **结果文件** | 无 | 有 (23GB 主要是 results) |
| **当前状态** | 已重构 (含 src/, experiments/, tests/) | 原始状态 (8月31日版本) |
| **正在运行** | 否 | **是 (7个进程)** |

---

## 二、两套目录的关系

```
aroma-dps/          ← GitHub publication repo
├── 0831-end-code/    ← 从 aroma-dps-code 上传时复制的快照
├── 0901-end-code/   ← 同上
├── code_end/         ← 同上
├── unified_models/   ← 从 data_90/model1 复制
├── src/aroma_dps/    ← 我们新建的 publication package
├── experiments/      ← 我们新建的实验 wrappers
├── tests/            ← 我们新建的 regression tests
├── p0_verification/  ← 我们新建的 P0 审计
├── archive/          ← 我们新建的归档目录
└── configs/scripts/  ← 我们新建的配置

aroma-dps-code/      ← 实验工作目录 (23GB)
├── 0831-end-code/    ← 实际训练代码 (正在运行)
├── 0901-end-code/    ← Fig.4 代码
├── code_end/         ← 历史代码 + 原始数据 (data1_end/)
│   └── data1_end/     ← collet_homa/nics/mbco_0716.csv
├── last_end_code/    ← 上一版本代码
├── best_model_package/ ← 最佳模型打包
├── results/          ← 所有实验结果 (20+GB)
├── figures/          ← 生成的图表
├── docs/             ← 工作文档
├── lunci3-10/        ← 各轮 OOD 测试数据
├── uspto-5k/         ← USPTO 反应数据
└── unified_models → (不存在, 通过 ORIG_MODELS_ROOT 指向 /home/ubuntu/data_90/alldata_in_3090/model1/)
```

---

## 三、详细差异分析

### 3.1 差异总览

| 目录 | 有差异的文件数 | 差异性质 |
|------|--------------|---------|
| `0831-end-code/` | 10 | 路径配置 + ring index fix + split manifest |
| `0901-end-code/` | 40 | 几乎全是路径硬编码 → 相对路径的修改 |
| `code_end/` | 53 | 同上路径修改 + 部分 ring index fix |
| `unified_models/` | N/A | aroma-dps-code 中不存在此目录 |
| **总计** | **103** | |

### 3.2 差异分类

#### 类型 A：路径配置差异 (影响：仅影响可移植性，不影响科学结果)

**这是最大的差异来源，占约 80 个文件。**

GitHub 仓库中的代码已将硬编码路径改为相对路径：

```python
# aroma-dps/ (GitHub, 已修改为相对路径)
PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ORIG_MODELS_ROOT = str(PROJ_ROOT / "unified_models")

# aroma-dps-code/ (工作目录, 原始硬编码)
PROJ_ROOT = Path("/home/ubuntu/aroma-dps-code")
ORIG_MODELS_ROOT = "/home/ubuntu/data_90/alldata_in_3090/model1"
```

**受影响文件** (代表性)：
- `0831-end-code/common/constants.py`
- `code_end/common/constants.py`
- `0901-end-code/fig4_lunci10/training/*.py` (全部 7 个)
- `0901-end-code/fig4_lunci10/data/*.py` (全部 6 个)
- `0901-end-code/fig4_lunci10/evaluation/*.py` (全部 8 个)
- `0901-end-code/fig4_lunci10/audit/*.py` (全部 7 个)
- `0901-end-code/fig4_lunci10/analysis/*.py` (全部 6 个)
- `0901-end-code/fig4_lunci10/reporting/*.py` (全部 4 个)
- `code_end/generalization_test/code/*.py` (约 20 个)

**影响评估**：
- **数值结果变化**: 否。路径只影响文件读写位置，不影响模型训练逻辑。
- **运行中实验受影响**: 否。运行中的进程已加载代码到内存，文件修改不影响。
- **风险**: 如果在 `aroma-dps/` 中运行这些脚本，相对路径需要 `unified_models/` 在仓库内（已复制），但 `data1_end/` 原始数据不在仓库中（需手动放置或 symlink）。

#### 类型 B：ring index fix (影响：descriptor 计算，不影响模型训练)

```python
# aroma-dps/ (GitHub, Commit 2 修复为 0-based)
target_ring_atoms = [int(idx) for idx in atom_on_ring if isinstance(idx, (int, float)) and 0 <= idx < mol.GetNumAtoms()]

# aroma-dps-code/ (工作目录, 原始 1-based, 有 bug)
target_ring_atoms = [idx - 1 for idx in atom_on_ring if isinstance(idx, (int, float)) and idx > 0]
```

**受影响文件**：
- `0831-end-code/common/features.py` L174
- `code_end/common/features.py` L116

**影响评估**：
- **数值结果变化**: 仅影响 `compute_ring_descriptors()` 输出的描述符值（环原子选错了一个位置）。
- **模型训练受影响**: **否**。模型训练走 `graphs.py::process_and_save_data()`，该函数本来就正确使用 0-based，未修改。
- **运行中实验受影响**: 否。运行中进程加载的是 `aroma-dps-code/` 的原始版本。

#### 类型 C：split manifest 差异 (影响：数据划分不同)

```json
// aroma-dps/ (GitHub, 9月6日生成)
{"split_seed": 2026, "n_total": 6923, "n_dev": 5537, "n_test": 1386}

// aroma-dps-code/ (工作目录, 9月7日生成)
{"split_seed": 2026, "n_total": 6820, "n_dev": 5479, "n_test": 1341}
```

**差异原因**：两套 split manifest 的生成时间不同。`aroma-dps-code/` 中的是 9月7日新生成的（可能是 stage2 standard 运行时覆盖的），`aroma-dps/` 的是上传时的快照。

**受影响文件**：
- `0831-end-code/splits/final_holdout_meta_stage2_standard.json`
- 其他 4 个 split manifest JSON

**影响评估**：
- **数值结果变化**: 如果使用不同 split manifest，测试集不同，指标会不同。
- **运行中实验受影响**: 运行中的进程会动态生成 split，不一定读这些 JSON。但如果有脚本读了旧 manifest 做缓存，可能不一致。
- **需要关注**: 应确认论文最终使用哪套 split。

#### 类型 D：目录结构差异

| | aroma-dps/ (GitHub) | aroma-dps-code/ (工作目录) |
|---|---|---|
| `unified_models/` | **有** (从 data_90 复制) | **无** (通过 `ORIG_MODELS_ROOT` 指向 `/home/ubuntu/data_90/alldata_in_3090/model1/`) |
| `src/aroma_dps/` | **有** (新建) | 无 |
| `experiments/` | **有** (新建) | 无 |
| `tests/` | **有** (新建) | 无 |
| `p0_verification/` | **有** (新建) | 无 |
| `archive/` | **有** (新建) | 无 |
| `code_end/data1_end/` | **无** (原始数据未上传) | **有** (`collet_homa/nics/mbco_0716.csv`) |
| `last_end_code/` | 无 | **有** |
| `best_model_package/` | 无 | **有** |
| `results/` | 无 | **有** (20+GB) |
| `figures/` | 无 | **有** |
| `docs/` | 无 | **有** |
| `lunci3-10/` | 无 | **有** |
| `uspto-5k/` | 无 | **有** |

#### 类型 E：仅存在于一侧的文件

**仅在 aroma-dps-code/ 中** (工作目录独有)：
- 原始数据: `code_end/data1_end/collet_*.csv`
- 历史代码: `last_end_code/`, `best_model_package/`, `baseline_gnn/`, `baseline_traditional_ml/`, `generalization_test/`, `ring_encoding_ablation/`
- OOD 数据: `lunci3_mol_images/`, `lunci4_mol_images/`, `lunci7_8_out_nics/`, `lunci8/`, `lunci8-3/`, `lunci9/`, `lunci10/`
- USPTO: `uspto-5k/`
- 工作文档: `工作汇总_*.md`, `代码库详细梳理.md`, `四阶段实验框架详细展开.md`, `汇总.xlsx`, `汇总_stereo.xlsx`
- 生成脚本: `generate_aroma_figures.py`, `generate_cada_figures.py`, `generate_scatter_figures.py` 等
- 诊断: `lunci6补充诊断_配对资源与噪声地板.txt`

**仅在 aroma-dps/ 中** (GitHub 独有)：
- Publication package: `src/aroma_dps/`
- Experiments: `experiments/`
- Tests: `tests/`
- P0 Verification: `p0_verification/`
- Archive: `archive/`
- Configs: `configs/`
- Scripts: `scripts/`

---

## 四、运行中实验的代码来源确认

当前 7 个运行中进程全部从 `/home/ubuntu/aroma-dps-code/0831-end-code/` 启动：

| PID | 脚本 | seed | feature_mode | 启动时间 |
|-----|------|------|-------------|---------|
| 282928 | `run_stage4_v2.py` | 22 | standard | 12:02 |
| 329472 | `run_stage4_v2.py` | 44 | explicit_aromaticity_ablated | 12:20 |
| 329488 | `run_stage4_v2.py` | 55 | standard | 12:20 |
| 329503 | `run_stage4_v2.py` | 55 | explicit_aromaticity_ablated | 12:20 |
| 764513 | `ring_conditioning_ablation.py` | 11 | standard | 15:21 |
| 776759 | `run_stage4_v2.py` | 11 | explicit_aromaticity_ablated | 15:26 |
| 949006 | `run_final_membership.py` | 11 | standard | 16:40 |

**所有进程使用的代码版本**：
- `constants.py`: 硬编码路径版本 (`/home/ubuntu/aroma-dps-code`)
- `features.py`: 原始 `idx-1` 版本 (1-based, 有 bug 但不影响模型训练)
- `ORIG_MODELS_ROOT`: `/home/ubuntu/data_90/alldata_in_3090/model1`
- split protocol: `get_final_splits(SPLIT_SEED=2026)` (publication protocol)

**结论**：运行中实验使用的是 `aroma-dps-code/` 的代码，与 GitHub 仓库 `aroma-dps/` 完全独立。我们的重构工作没有影响运行中实验。

---

## 五、核心问题回答

### Q1: "哪个正确？"

两者都"正确"，但用途不同：

- **`aroma-dps-code/` 是"科学正确"的** — 它是实际产生实验结果的代码，包含完整的数据路径、模型权重、结果文件。正在运行的 7 个进程从这里启动。
- **`aroma-dps/` 是"publication 正确"的** — 它已经过重构（相对路径、ring index fix、publication package 结构），适合提交给期刊和 GitHub。但不包含原始数据，不能直接运行实验。

### Q2: "二者的差异是什么？"

**本质上是一个 "working copy" 和 "publication copy" 的关系**：

1. **路径配置** (80 个文件): GitHub 版本改为相对路径，工作版本是硬编码绝对路径。这是上传 GitHub 前做的预处理。
2. **ring index fix** (2 个文件): GitHub 版本修复了 `idx-1` bug (Commit 2)，工作版本保留原始代码。
3. **split manifest** (5 个文件): 两套 manifest 的 n_total 不同，可能因生成时间不同。
4. **目录结构**: GitHub 版本新增了 `src/`, `experiments/`, `tests/` 等 publication 结构；工作版本有完整的 `data1_end/`, `results/`, `figures/` 等。
5. **unified_models 位置**: GitHub 版本在仓库内；工作版本指向外部数据盘 `/home/ubuntu/data_90/alldata_in_3090/model1/`。

### Q3: "需要做什么？"

1. **立即可做**: 将 `aroma-dps-code/` 中缺失的代码文件（`last_end_code/`, `best_model_package/`, `generate_*.py` 等）补充到 GitHub 仓库。
2. **等实验结束后**: 将 `aroma-dps-code/` 的最新结果同步到 GitHub 仓库（排除大型数据文件）。
3. **需要确认**: split manifest 的差异 — 确认论文最终使用哪套 split (n_total=6923 vs 6820)。
4. **路径策略**: 决定 publication 仓库是否保留相对路径（当前方案），还是添加环境变量 fallback 兼容两种使用方式。
