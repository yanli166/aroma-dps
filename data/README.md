# data — 仓库内数据（Data Availability）

- **迁移日期**：2026-09-08
- **总原则**：本目录只放**成品数据表**（训练集注册表、外部基准表、人工反应表）。
  **原始量子化学中间产物（Gaussian `.chk/.fchk/.log`、Multiwfn 输入输出、xTB 逐分子目录、`.xyz` 构象库）不随仓库分发**，
  全部留存于实验工作目录，来源路径逐条列于下文与 `experiments/fig2_dataset/` 各 README。
- 复制方式为逐字节 `cp`（已用 `cmp` 与工作目录原件核对一致）。

## 各数据集 → 论文章节对应关系

| 子目录 / 文件 | 论文位置 | 作用 | 来源绝对路径 |
|---|---|---|---|
| `collet/collet_homa_0716.csv` | Methods 数据 / Fig.2 / Fig.3 训练标签 | HOMA 任务训练-测试注册表（6858 行，`New_ID,smiles,Ring_ID,Ring_Size,atom_on_ring,homa_value`） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_homa_0716.csv` |
| `collet/collet_nics_0716.csv` | 同上 | NICS(1)zz 任务注册表（6820 行，目标列 `NICS_value`；表头含一个空的尾列，原样保留） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_nics_0716.csv` |
| `collet/collet_mbco_0716.csv` | 同上 | MBCO 任务注册表（6927 行，目标列 `mbco_value`） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_mbco_0716.csv` |
| `lunci10/lunci10_unified.csv` | Fig.2 / Fig.4 外部基准 | 38 环骨架 × 30 取代基统一表（2153 行，含 `ring_name/sub_name/sub_type/ring_pos` 与 HOMA/MBCO/NICS_iso/NICS_ZZ） | `/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv` |
| `lunci10/lunci10-test-corrected.csv` | Fig.4 外部基准（修正版） | 同上的修正版（2153 行，`New_ID,SMILES,Ring_ID,Ring_Size,Ring_Atoms,...`） | `/home/ubuntu/aroma-dps-code/lunci10/lunci10-test-corrected.csv` |
| `lunci_external/lunci6-test.csv` | Fig.4 / SI 外部测试 | 早期取代基外部集（312 行，无 NICS_iso 列） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/lunci6-test.csv` |
| `lunci_external/lunci78-test.csv` | Fig.4 / SI 外部测试 | lunci7+8 外部集（610 行） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/lunci78-test.csv` |
| `lunci_external/lunci10-test.csv` | Fig.4 外部测试（扩展版） | lunci10 扩展测试集（2465 行；与 `lunci10/` 两表的 2153 行版本为不同切分，勿混用） | `/home/ubuntu/aroma-dps-code/code_end/data1_end/lunci10-test.csv` |
| `reactions/汇总.xlsx` | Fig.5 | 83 对策划去芳构化反应手工表（反应物/产物 SMILES + 环索引），立体描述符脚本的输入 | `/home/ubuntu/aroma-dps-code/汇总.xlsx` |
| `reactions/汇总_stereo.xlsx` | Fig.5 / Methods | 上表 + 11 维立体描述符（ETKDG+MMFF 构象）扩展版 | `/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx` |

## md5（用于与工作目录核对）

```
a8521cf4cc49422a6983fc492fe5b8fd  collet/collet_homa_0716.csv
2c5829ac817d212a43a65b61315375bf  collet/collet_nics_0716.csv
e7afaa8b572e73790cb07ea2997fbfb7  collet/collet_mbco_0716.csv
0c0a6f2db7d716f06dbf9a0f61509201  lunci10/lunci10_unified.csv
0c7d0549d9afe72a57813128b9ee83d6  lunci10/lunci10-test-corrected.csv
0d4a8db276b120a32ef8e6980c0e2441  lunci_external/lunci6-test.csv
283ddc757e77bc212965d4b8c8033711  lunci_external/lunci78-test.csv
b835da9001b7d6ecf94fa0f57541fb75  lunci_external/lunci10-test.csv
f6e24a1b7616f487110881e4e7dd931d  reactions/汇总.xlsx
68fc2fb01db1794a8a48c021dc6cd60f  reactions/汇总_stereo.xlsx
```
核对命令：在 `/home/ubuntu/aroma-dps/data` 下执行 `md5sum -c` 上述清单，或
`md5sum /home/ubuntu/aroma-dps-code/code_end/data1_end/collet_homa_0716.csv` 与工作目录逐一比对。

## 数据口径提示

- `atom_on_ring` / `Ring_Atoms` **均为 0-based** 索引（RDKit `AddHs` 后的原子序），与
  `models/best_model_package/graph_utils.py` 及 `experiments/si_baselines/common/features.py`（修复版）一致；
  历史上按 1-based 处理（`[idx - 1 for idx in atom_on_ring]`）的代码一律未入仓（规则 2）。
- 行数与 `models/best_model_package/metrics.json` 的 `n_total` 略有差异（HOMA 6858→6854、MBCO 6927→6923、
  NICS 6820→6820），原因是 `train_final.load_data()` 对 `target` 与 `smiles` 做了 `dropna`。数据表本身未做任何删改。

## 未复制 / 暂置清单（数据侧）

- 未复制：`nics_pipeline` 各批次的量子化学计算目录 —— `/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci7/s1/`、
  `/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/`、`/home/ubuntu/aroma-dps-code/lunci8/gaussian_inputs/`、
  `/home/ubuntu/aroma-dps-code/lunci8-3/gaussian_inputs/`、`/home/ubuntu/aroma-dps-code/lunci9/gaussian_inputs/`、
  `/home/ubuntu/aroma-dps-code/lunci9_out/`。原因：原始中间产物不随仓库分发（规则 4）。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/gaussian_inputs/`（4174 文件 4.6 GB，含 60 个 >5 MB 的 `.chk`）、
  `lunci10/xtb_output/`（1389 目录 121 MB）、`lunci10/xyz_files/`（1389 文件 5.5 MB）——
  见 `experiments/fig2_dataset/lunci10_construction/README.md`。
- 未复制：基线三层逐折/逐种子权重与预测目录（`baseline_*/results/`、`new5zhecv/results/` 共约 440 MB）——
  见 `experiments/si_baselines/*/README.md`，仓库仅保留各层汇总 CSV。
- 暂置未复制（清单未指定去向）：`/home/ubuntu/aroma-dps-code/lunci10/lunci10-expanded-test.csv`（216 KB，
  `lunci10-test.csv` 的上游同名版本）、`/home/ubuntu/aroma-dps-code/code_end/data1_end/` 中除已列 6 个 CSV 外无其他文件。
- 未复制：`/home/ubuntu/aroma-dps-code/code_end/data1_end/` 的旧版 0702 表（在 `/home/ubuntu/data_90/alldata_in_3090/model1/`，
  含 `collet_homa_0702.csv`、`nics-nics1zz-out-no3.csv`、`outcsv/lunci2-mbcout.csv`）——SI 三层基线的 `common/tasks.py`
  仍按这些名字寻址，若要复现基线原值需另行取回；风险已记在 `experiments/si_baselines/common/README.md`。
- 本目录不含 Python 代码，故**无**"仍需路径相对化的 `.py` 文件"；引用这些数据的脚本待相对化项记录在
  `experiments/fig2_dataset/*/README.md`、`experiments/si_baselines/*/README.md`、`models/best_model_package/README.md`。
- ⚠️ git 层面待办（本次未改配置）：仓库根 `.gitignore` 含 `*.csv`、`*.xlsx`、`*.pt`，
  本目录数据表与模型权重虽已落盘，但**不会被 git 跟踪**。投稿前需为 `data/`、`models/` 增加例外规则
  （或按 README 建议把权重改走 Zenodo/Release、git 内只留 manifest + md5）。

## `interpretability/` — Fig.3f 可解释性 / 归因分析数据集

- **面板归属**：Fig.3f（可解释性 / 归因分析）
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/0908-end-code/data/`（由可解释性迁移任务复制，本任务仅补全其在本 README 中的登记）
- 10 个文件（6 `.csv` + 4 `.json`），全部 <5 MB。列语义统一为 `smiles, atom_on_ring, y, mol_id, source, split`；
  `atom_on_ring` 全程 **0-based**。

两条实验线：

1. **`merged_*`（并入 lunci10 的同测试集对比线）**：`split` ∈ {`dev`, `test`, `l10`}。
   由 `collet_*_0716` 去掉 e- 前缀行（`n_e_dropped` 224/228/224）得 `collet_ab`，再并入 lunci10 环级记录；
   `merge_summary.json` 记录各步计数。
2. **`*_l10val`（lunci10 分子级留出线）**：`split` ∈ {`train`, `l10_val`}。
   三任务共享同一留出集：**272 个 lunci10 验证分子 / 436 条验证记录**，训练侧含 1087 个 lunci10 分子
   （HOMA 1686 条、MBCO/NICS 各 1687 条，另含 `extra_l10_train_records` 29 条），`split_seed = 2026`，`l10_test_frac = 0.2`。

| 文件 | 行数（数据行） | split 分布 |
|---|---|---|
| `merged_HOMA.csv` | 8752 | dev 5295 / test 1335 / l10 2122 |
| `merged_MBCO.csv` | 8851 | dev 5360 / test 1339 / l10 2152 |
| `merged_NICS_1zz.csv` | 8744 | dev 5311 / test 1281 / l10 2152 |
| `homa_l10val.csv` | 8752 | train 8316 / l10_val 436 |
| `mbco_l10val.csv` | 8822 | train 8386 / l10_val 436 |
| `nics_1zz_l10val.csv` | 8715 | train 8279 / l10_val 436 |

md5：

```
d21c507f034185665020cb51f17b76bc  interpretability/merged_HOMA.csv
8894c28cc3242286211e270a9e54841b  interpretability/merged_MBCO.csv
9217d6e8d59c44d42ed8aab5e3dcfb02  interpretability/merged_NICS_1zz.csv
9c61f840d28d6c3cf174b411a3b22522  interpretability/homa_l10val.csv
84a498a567cb00c669723aff4fb7be70  interpretability/mbco_l10val.csv
2f7c2e03b0f74a624ec59d75a847b4d4  interpretability/nics_1zz_l10val.csv
c3fd67339e95cc58e2bd7f1c9b7ca7b3  interpretability/merge_summary.json
4e099f69445593e53dd3b89be06a4ca6  interpretability/homa_l10val_manifest.json
dd617d079a12fb4a2478d37c32503dff  interpretability/mbco_l10val_manifest.json
907b9e83d706f8e5696592f49564003d  interpretability/nics_1zz_l10val_manifest.json
```

配套代码 `experiments/fig3_model/interpretability/scripts/`、结果 `experiments/fig3_model/interpretability/results/`、
权重 `models/interpretability/`；细节另见 `interpretability/README.md`。

> 备注：本节由 fig2/si_baselines 归档任务（2026-09-08）重写——本 README 的上一版在并发写入时被覆盖，
> 上表事实全部由 `md5sum` 与逐行统计重新核验；若可解释性任务持有更详细的列语义说明，请以该任务版本为准。
