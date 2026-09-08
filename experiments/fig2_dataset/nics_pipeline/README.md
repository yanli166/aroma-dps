# nics_pipeline — 量子化学描述符后处理管线（HOMA / MBCO / NICS(1)zz）

- **论文面板归属**：Methods（芳香性标签的计算定义）＋ Fig.2 数据来源；本管线是三任务标签值的唯一生成路径。
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/nics_pipeline/`（原样复制，未改动任何代码、路径或格式）

## 已复制文件清单

| 文件 | 职责 |
|---|---|
| `config.py` | 数据集注册表（`DATASETS`：lunci7 / lunci8 / lunci8-2 / lunci8-3 / lunci9）与派生路径规则 |
| `01_prepare_inputs.py` | Step 1：从 SMILES 提取环信息 → `ring_info.csv` |
| `02_run_formchk.sh` | Step 2：批量 `chk → fchk` |
| `03_run_multiwfn.sh` | Step 3：Multiwfn 计算 HOMA 或 MBCO |
| `04_parse_homa_mbco.py` | Step 4：解析 Multiwfn HOMA/MBCO 输出 → `*-homa-mbco-summary.csv` |
| `05_gen_nics_multiwfn_input.py` | Step 5：从 NMR log 提取磁屏蔽张量，生成 NICS 输入 txt |
| `06_run_multiwfn_nics.sh` | Step 6：Multiwfn NICS_ZZ（function 25 option 4） |
| `07_parse_nics.py` | Step 7：解析 NICS_ZZ → 最终汇总 CSV |
| `run_all.sh` | 01→07 串行驱动 |

## 数据集注册表与训练集 CSV 的关系（重要）

管线的注册单位是**批次**（lunci7 / lunci8 / lunci8-2 / lunci8-3 / lunci9），每批次产出
`{out_dir}/{dataset}-homa-mbco-nics-final.csv`。这些批次表经合并整理后，构成三任务训练集注册表：

- `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_homa_0716.csv`
- `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_nics_0716.csv`
- `/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_mbco_0716.csv`

这三个 CSV **已随本次归档迁入仓库** `data/collet/`（见 `data/README.md`）。也就是说：本目录只保留"计算 → 解析"的代码，
批次级中间表与最终训练集的对应关系由 `collet_*_0716.csv` 承载。

## 未复制 / 暂置清单

- 未复制：`/home/ubuntu/aroma-dps-code/nics_pipeline/__pycache__/config.cpython-313.pyc`（垃圾文件，规则 4）
- 未复制：各批次 Gaussian `chk/gjf/log/fchk` 与 Multiwfn 输入输出目录，来源
  `/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/`、`/home/ubuntu/aroma-dps-code/lunci8/gaussian_inputs/`、
  `/home/ubuntu/aroma-dps-code/lunci8-3/gaussian_inputs/`、`/home/ubuntu/aroma-dps-code/lunci9/gaussian_inputs/`、
  `/home/ubuntu/aroma-dps-code/lunci9_out/`、`/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci7/s1/`
  ——原因：巨型量子化学中间产物目录（规则 4），不随仓库分发；数值结论以 `data/collet/` 的 CSV 为准。
- 未复制：管线运行日志（`*.log`）——原因：`.gitignore` 排除且非交付物。
- 本目录不含 `common/features.py`；未复制任何含 1-based ring index bug 的文件（规则 2）。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `config.py`（`DATASETS` 中每个批次的 `input_dir` / `csv_path` / `out_dir`，共 15 处，例如
  `/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci7/s1`、`/home/ubuntu/aroma-dps-code/lunci9_out`）

附带（非 `.py`，同一批需处理）：`03_run_multiwfn.sh`（`Multiwfnpath=/home/ubuntu/data_90/.../Multiwfn`）、
`06_run_multiwfn_nics.sh`（`MULTIWFN`、`Multiwfnpath`）、`run_all.sh`（注释中的 Multiwfn 路径）。
迁移时**未修改**这些路径（规则 1）。
