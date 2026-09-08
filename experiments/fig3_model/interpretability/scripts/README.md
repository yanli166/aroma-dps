# Fig.3f — 可解释性 / 归因分析脚本

- **面板归属**：Fig.3f（可解释性 / 归因分析）
- **来源**：`/home/ubuntu/aroma-dps-code/0908-end-code/scripts/`
- **迁移日期**：2026-09-08
- **迁移方式**：原样复制，**未修改任何代码逻辑、路径或文件内容**（`diff -r` 已验证逐字节一致）。
- **文件数**：13 个 `.py`（源目录 `scripts/` 中全部 Python 文件即此 13 个；源目录的 `__pycache__/`、`*.pyc` 按规则未复制）。

---

## 0. 这批实验在回答什么问题

论文需要证明：RC-GNN/MPNN 预测目标环芳香性时，**依据的是环内共轭原子的成键环境，而不是取代基侧链的电子效应**。
为此跑了**两条相互独立的实验线**：

| | 线 1：merged 方案 | 线 2：l10val 方案 |
|---|---|---|
| 做法 | collet a/b + lunci10 合并去重后重训 | lunci10 按**分子** 80/20 留出作验证集训练三任务模型 |
| 评测口径 | test = collet holdout（与旧基线同一 test，apples-to-apples） | 同一组 **272 个 l10 验证分子 / 436 条环记录** |
| 结论 | 并入 lunci10 不损害 collet 上的精度（R² 持平、MAE 略优） | 在同 6 个验证分子上做 IG + \|x·∇x\| 双归因，环内/取代基比值远大于 1 |
| 完成度 | **仅 HOMA 跑完**（NICS_1zz / MBCO 未训，见 `../pending/`） | **三任务全部跑完** |

---

## 1. 执行顺序与依赖关系

### 线 1 — merged 方案（Step 1 → 6）

```
Step 1  prepare_merged_data.py
        ├─ 输入(仓库外): code_end/data1_end/{collet_homa_0716,nics,mbco}.csv
        │               lunci10/lunci10_unified.csv
        └─ 输出: ../data/merged_{HOMA,NICS_1zz,MBCO}.csv + merge_summary.json
             │
Step 2  train_merged.py --task {HOMA|NICS_1zz|MBCO} --gpu N
             └─ 输出: models/{task}_merged_best.pt, results/{task}_metrics.json,
                      predictions/{task}_test_predictions.csv
             │
Step 3  eval_old_baseline.py --gpu N
        ├─ 输入: merged_{task}.csv 的 test 子集 + 仓库外 best_model_package 旧权重
        └─ 输出: predictions/old_HOMA_test_predictions.csv, results/old_baseline_metrics.json
             │
Step 4  plot_merged_figures.py            ← ★ 从未运行（三任务图缺失，见 ../pending/）
             │
Step 5  saliency_homa_driver.py           ← 依赖并 import saliency_merged.py
        │       （saliency_merged.py 为图库 + 三任务驱动，本身不单独跑）
        └─ 输出: ../attention_viz/（HOMA 5 分子 saliency、类别对比、统计面板、attention_stats_HOMA.csv）
             │
Step 6  plot_homa_figures.py
        └─ 输出: ../figures/HOMA_merged_test_parity.png, HOMA_old_vs_merged_metrics.png
```

依赖链：`Step1 → Step2 → Step3 → Step6`；`Step2 → Step5`（需要 `HOMA_merged_best.pt`）；`Step1 → Step5`（`saliency_merged.py` 会读 `data/merged_{task}.csv` 取真值）；`Step4` 依赖 `Step2 + Step3`，但因三任务权重不全而**从未执行**。

### 线 2 — l10val 方案（Step A → C）

```
Step A  prepare_homa_l10val.py           （lunci10 分子级 GroupShuffleSplit, seed=2026, frac=0.20）
        └─ 输出: ../data/homa_l10val.csv (split: train/l10_val) + homa_l10val_manifest.json
             │
        ─── 以下两线共用 homa_l10val.csv 的分子级划分 ───
        │
Step A' prepare_task_l10val.py --task {NICS_1zz|MBCO}
        │       ★ 直接复用 homa_l10val.csv 中 source=='l10' 的 train/val 分子归属，
        │         以保证三任务验证分子完全相同（272 个分子 / 436 条记录）
        └─ 输出: ../data/{task}_l10val.csv + {task}_l10val_manifest.json
             │
Step B  train_homa_l10val.py --gpu 0     ← ★ 保存预测阶段崩溃（见 §2 问题①）
        └─ 输出: models/HOMA_l10val_best.pt（已存盘）；results/ 与 predictions/ **未写出**
             │
        resave_homa_l10val_preds.py      ← ★★ 事后抢救脚本：复用已训好的 .pt 补写产物
        └─ 输出: results/HOMA_l10val_metrics.json, results/HOMA_l10val_per_ringfamily.csv,
                 predictions/HOMA_l10val_predictions.csv
             │
        train_task_l10val.py --task {NICS_1zz|MBCO} --gpu 0   （无此 bug，一次跑通）
        └─ 输出: models/{task}_l10val_best.pt, results/{task}_l10val_metrics.json,
                 results/{task}_l10val_per_ringfamily.csv, predictions/{task}_l10val_predictions.csv
             │
Step C  saliency_l10val.py --task {HOMA|NICS_1zz|MBCO} --gpu 0
        └─ 输出: ../attention_l10val/（HOMA）、../attention_l10val_nics/、../attention_l10val_mbco/
                 各含 6 张双归因 PNG + 1 个 attribution_stats.csv
```

依赖链：`StepA → StepA' → StepB → (resave) → StepC`。`StepC --task X` 需要对应 `models/X_l10val_best.pt` **和** `predictions/X_l10val_predictions.csv`（HOMA 的 predictions 正是由 `resave` 补出来的，故 resave 是 HOMA 分支上**不可跳过**的一环）。

---

## 2. 已知代码问题（原样保留，未修复）

### 问题① `train_homa_l10val.py` 保存预测时崩溃，产物由 `resave_homa_l10val_preds.py` 事后补写（bug 未修）

- 现象：训练与早停**已正常完成**（日志 `results/logs/logs_train_homa_l10val.log` 末尾已打印 `[l10_val] n=436 R2=0.9615 MAE=0.0371 …`），随后在写预测 CSV 阶段抛 `KeyError: 'atom_on_ring'` 崩溃。
- 根因位置：`train_homa_l10val.py` 第 185–186 行 —— `key = lambda x: (str(x["smiles"]), str(sorted(x["atom_on_ring"])))` 作用在 `l10_meta.iterrows()` 上；而 `l10_meta` 的 `atom_on_ring` 列是由第 178 行 `l10m["atom_on_ring"] = l10m["ring_atoms"].apply(parse_aor)` 从 `lunci10_unified.csv` 的 `ring_atoms` 列派生的，该 DataFrame 在此处实际不含 `atom_on_ring`（列名/来源不匹配）。崩溃时 `models/HOMA_l10val_best.pt` 已在第 172 行之前保存，故权重未丢。
- 处置：**bug 未修**。改用 `resave_homa_l10val_preds.py` 加载已存好的 `HOMA_l10val_best.pt`，在 `homa_l10val.csv` 的 `l10_val` 子集上重推一遍，补写 `results/HOMA_l10val_metrics.json`、`results/HOMA_l10val_per_ringfamily.csv`、`predictions/HOMA_l10val_predictions.csv`（该脚本用 `smiles + sorted(atom_on_ring)` 作为键去 `lunci10_unified.csv` 反查 `ring_name/sub_name/sub_type/ring_pos`，规避了原崩溃点）。
- **provenance 注意**：`HOMA_l10val_best.pt` 是训练产物（可信）；但 HOMA 的 metrics/predictions 是**第二次推理**的产物。由于模型与权重、输入数据完全一致，指标数值与训练时日志中打印的 `R2=0.9615 MAE=0.0371` 一致，可交叉核对。
- 对照：`train_task_l10val.py`（NICS/MBCO）含结构相同的代码，但**未触发**该崩溃，一次跑通并直接写出全部产物。

### 问题② `saliency_merged.py` 引用了不存在的 NICS_1zz / MBCO merged 权重

- 位置：`saliency_merged.py` 第 67–71 行 `MODEL_FILES = {'HOMA': 'HOMA_merged_best.pt', 'NICS_1zz': 'NICS_1zz_merged_best.pt', 'MBCO': 'MBCO_merged_best.pt'}`，并在第 144–145 行对 `MODEL_FILES.items()` 逐个 `torch.load`。
- 事实：`models/` 中**只有 `HOMA_merged_best.pt`**；`NICS_1zz_merged_best.pt`、`MBCO_merged_best.pt` 不存在（Step 2 未对 NICS 完成，见 `results/logs/logs_train_nics.log` —— 该日志只有 RDKit UFFTYPER 警告、无任何 epoch/指标输出，说明 NICS merged 训练启动后未跑完）。
- 后果：`saliency_merged.py` **作为独立脚本直接运行必然 `FileNotFoundError`**。实际可用的产物全部来自 `saliency_homa_driver.py`：它 `import saliency_merged as sm` 只复用其中的绘图/归因函数，并自行把 `TASK` 限定为 `"HOMA"`、只加载 `HOMA_merged_best.pt`（日志 `logs_saliency_homa.log` 可证：仅 `HOMA (merged 重训)` 一节）。
- 同理 `plot_merged_figures.py` 也依赖三任务 metrics.json，因此从未运行。

### 附① `resave_homa_l10val_preds.py` 硬编码 `cuda:0`
`resave_homa_l10val_preds.py` 与 `train_homa_l10val.py` 均直接 `torch.device("cuda:0" if …)`，无 `--gpu` 接口；重跑需用 `CUDA_VISIBLE_DEVICES` 间接控制设备。其余脚本用 `--gpu` 参数。

### 附② `plot_merged_figures.py` 残留调试行（未影响产物，因该脚本从未运行）
第 107 行 `new_met = json.load(open(os.path.join(RESULT_DIR, "HOMA_metrics.json")))  # placeholder` 是一行遗留占位代码，`new_met` 后续**并未被使用**（真正的逐任务指标在第 116 / 136 / 160 行按 `f"{t}_metrics.json"` 重新读取）。因该脚本从未执行（见 `../pending/README.md` P3），此问题目前**不会**造成失败；但一旦补齐 NICS/MBCO 权重后运行 Step 4，这行仍会因 `HOMA_metrics.json` 存在而静默通过。相对化改造时可一并清理。

### 附③ `eval_old_baseline.py` 的 `--task` 缺省会跑全部三任务
第 58–59 行 `--task` 默认 `None` = 评估 `OLD_FILES` 全部三项（`homa_best.pt` / `nics_1zz_best.pt` / `mbco_best.pt`）。现存 `old_baseline_metrics.json` 只有 `HOMA` 一个键，说明源实验当时是**显式** `--task HOMA` 运行的。补训后重跑时若直接用缺省值，会因 lunci10 口径变化而覆盖现有 HOMA 基线记录，需注意。

---

## 3. 仓库外路径依赖（待相对化项）

这些脚本在运行时依赖本 publication 仓库**之外**的工作目录路径，属已知技术债，**本次迁移按规则未改动**。

### 3.1 外部依赖清单
| 外部路径 | 作用 | 被哪些脚本使用 |
|---|---|---|
| `/home/ubuntu/aroma-dps-code/best_model_package/` | 提供 `graph_utils.build_graph`（SMILES+`atom_on_ring`→60 维节点特征图）与 `model_arch.build_model`（MPNN 架构），以及 Step 3 的**旧基线权重** | `eval_old_baseline.py`、`train_merged.py`、`train_homa_l10val.py`、`train_task_l10val.py`、`resave_homa_l10val_preds.py`、`saliency_merged.py`、`saliency_l10val.py`（7 个，均 `sys.path.insert(0, …)`） |
| `/home/ubuntu/aroma-dps-code/code_end/data1_end/` | collet 三任务原始标注 CSV（`collet_homa_0716.csv` 等），Step 1 / Step A 的输入 | `prepare_merged_data.py`、`prepare_homa_l10val.py`、`prepare_task_l10val.py`、`saliency_merged.py`（4 个） |
| `/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv` | lunci10 统一标注表，Step A/B 输入与环名/取代基元信息反查源 | `prepare_merged_data.py`、`prepare_homa_l10val.py`、`prepare_task_l10val.py`、`resave_homa_l10val_preds.py`、`train_homa_l10val.py`、`train_task_l10val.py`（6 个） |

### 3.2 需要相对化的 `.py` 数量：**10 / 13**

- **含仓库外绝对路径（10 个）**：`eval_old_baseline.py`、`prepare_homa_l10val.py`、`prepare_merged_data.py`、`prepare_task_l10val.py`、`resave_homa_l10val_preds.py`、`saliency_merged.py`、`saliency_l10val.py`、`train_homa_l10val.py`、`train_merged.py`、`train_task_l10val.py`
- **不含绝对路径（3 个，但依赖 `ROOT` 目录约定）**：`plot_homa_figures.py`、`plot_merged_figures.py`、`saliency_homa_driver.py`

### 3.3 相对化时还需注意的目录约定（重要）

所有脚本用 `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`，即假定自己位于 `<ROOT>/scripts/`，并在 `ROOT` 下寻找 `data/`、`models/`、`results/`、`predictions/`、`figures/`、`attention_*/`。

迁移后本目录的 `ROOT` 变成 `/home/ubuntu/aroma-dps/experiments/fig3_model/interpretability/`，而本仓库按用户框架把三类产物**分置**到：

| 脚本预期的 `ROOT/…` | 本仓库实际位置 |
|---|---|
| `ROOT/data/` | `/home/ubuntu/aroma-dps/data/interpretability/` |
| `ROOT/models/` | `/home/ubuntu/aroma-dps/models/interpretability/` |
| `ROOT/{results,predictions,figures,attention_*}/` | `/home/ubuntu/aroma-dps/experiments/fig3_model/interpretability/results/…`（见 `../results/README.md`） |

因此**脚本现状不可直接在新位置重跑**；相对化时必须统一引入一个可配置的数据/权重根路径（建议读环境变量或 `configs/default.yaml`），或在新位置建立到 `data/interpretability`、`models/interpretability` 的符号链接。此为后续独立任务，本次未执行。

---

## 4. ring index（1-based vs 0-based）自查结论

**方法**：对 13 个 `.py` 逐个执行
`grep -n "for idx in atom_on_ring\|idx - 1"` → **零命中**。
再扩大模式（`- 1` / `-1` 出现在 `aor|atom_on_ring|ring` 同一行、列表推导 `-1 for`、`1-based/0-based/索引` 关键词、以及全量 `--\s*1\b` 原始扫描）复核。

**结论：13 个文件全部通过，无一含 ring index 1-based→0-based 的减一处理，因此本目录未排除任何文件（"未复制：含未修复 ring index bug"清单为空）。**

扫描中出现的 `-1` 均属合法用途，与环索引无关，逐条如下：
- `.reshape(-1)` / `sum(dim=-1)` / `squeeze(-1)` / `squeeze(dim=-1)`：张量形状操作（`eval_old_baseline.py:86`、`resave_homa_l10val_preds.py:52`、`saliency_l10val.py:137,144`、`saliency_merged.py:195`、`train_homa_l10val.py:108,118,143`、`train_merged.py:12,150`、`train_task_l10val.py:114,124,149`）。
- `saliency_l10val.py:53`：`"attention_l10val_nics", "NICS(1)zz", -1.0` —— NICS(1)zz 越负越芳香，故对归因值**整体乘 −1** 以统一配色方向（红=增强芳香性），是符号翻转不是索引运算。
- `saliency_l10val.py:103`：`z = 2.0 * np.clip(value, 0, 1) - 1.0` —— colormap 归一化公式。
- `ring_over_sub` 计算式（`saliency_l10val.py:231`、`saliency_merged.py:345`）中匹配到的 `1` 实为防除零偏置 `1e-10`。

**索引约定的一致性证据**：`saliency_merged.py:95` 注释明确写「atom_on_ring 取自当前 0716 数据, **0-based** 与 AddHs 后原子索引对齐」，`:218` 写「在 AddHs 分子上找到与 target_atoms 完全匹配的 RDKit 环（**0-based**）」。即本批脚本从数据准备到归因可视化**全程按 0-based 传递 `atom_on_ring`**，未做任何 1-based↔0-based 转换，与 `p0_verification/audit_atom_on_ring.py` 的 0-based 约定一致；因此 `attention_viz/` 与 `attention_l10val*/` 的环内/环外原子着色归属可信。

**仍需独立复核之处（诚实标注）**：`graph_utils.build_graph`（位于仓库外 `best_model_package/`，本次未复制）如何处理传入的 `atom_on_ring` 不在本自查覆盖范围内。若其内部隐含 1-based 假设，则节点特征 `ring_flag` 的错位不会体现在本目录脚本文本上。建议后续把 `graph_utils`/`model_arch` 一并纳入 `src/aroma_dps/` 后，用 `p0_verification/atom_on_ring_audit_*.csv` 对 `models/interpretability/*.pt` 的训练输入做一次端到端复核。
