# lunci10_construction — 外部基准 lunci10 的构建流程（38 环骨架 × 30 取代基）

- **论文面板归属**：Fig.2 数据来源 / 外部基准构建（Methods 中"外部验证集如何生成"一节）；
  其**评估**用法属 Fig.4（脚本在 `0901-end-code/fig4_lunci10/data/`，由 Fig.4 迁移任务归档）。
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/lunci10/`（流程脚本与小体积输入/版本表）

## 构建链条（xTB → Gaussian 16 → 描述符解析）

1. SMILES → 加氢 + ETKDG/MMFF 三维构象 → `xyz`；
2. `xtb`（GFN2）几何优化 → `xtbopt.xyz`；
3. 由优化结构生成 Gaussian `.gjf`（HOMA/MBCO 用几何、NMR-NICS 用磁屏蔽计算）；
4. `run_g16_parallel.py` 并行提交 G16（改写 `%mem`/`%nprocshared`、独立 `GAUSS_SCRDIR`、失败重试一次）；
5. 计算结果解析为标签表（→ `lunci10_unified.csv` / `lunci10-test-corrected.csv`）。

## 已复制文件清单

| 文件 | 来源 | 说明 |
|---|---|---|
| `run_g16_parallel.py` | `/home/ubuntu/aroma-dps-code/lunci10/run_g16_parallel.py` | G16 并行运行器 |
| `1_xtb.py` | `/home/ubuntu/aroma-dps-code/lunci9/1_xtb.py` | xTB 优化 + gjf 生成脚本（**见下方来源说明**） |
| `lunci10-begin.csv` | `/home/ubuntu/aroma-dps-code/lunci10/lunci10-begin.csv` | 构建注册表：1389 条，列 `no,SMILES,ring_name,sub_name,sub_type,ring_pos` |
| `lunci10-begin-simple.csv` | `/home/ubuntu/aroma-dps-code/lunci10/lunci10-begin-simple.csv` | 同上的 `no,SMILES` 精简版（1389 条） |

**关于 `1_xtb.py` 的来源**：`/home/ubuntu/aroma-dps-code/lunci10/` 目录内**没有** xTB/gjf 生成脚本
（仅 `run_g16_parallel.py`，其 `__pycache__` 也只有该模块的 `.pyc`），第 1–3 步沿用的是同批次通用模板
`/home/ubuntu/aroma-dps-code/lunci9/1_xtb.py`（lunci8 下另有同名脚本）。为使本目录的构建流程自成一体，
按迁移清单要求复制该"1_xtb.py 类构建脚本"，并在此明确它**不是 lunci10 目录内的原件**；
若需要严格逐字节还原本批次，请以 `lunci10/xtb_output/`、`lunci10/gaussian_inputs/`（留存工作目录）为准核对。

## 未复制 / 暂置清单（量化中间产物不入库）

- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/gaussian_inputs/`（4174 个文件，4.6 GB；G16 `chk/gjf/log`，
  其中 **60 个 `.chk` 单文件 >5 MB**，最大 `e1078.chk` 6.7 MB）——规则 4 巨型中间产物目录 + 单文件 >5 MB。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/xtb_output/`（1389 个逐分子目录，121 MB）——规则 4（`xtb_output/`）。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/xyz_files/`（1389 个文件，5.5 MB）——规则 4（`xyz_files/`）。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/g16_run.log`、`xtb_run.log`——运行日志，`.gitignore` 排除 `*.log`。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/__pycache__/run_g16_parallel.cpython-313.pyc`——垃圾文件。
- 未复制到此目录（改由 data 承载）：`lunci10_unified.csv`、`lunci10-test-corrected.csv`
  → 已迁入 `data/lunci10/`（它们是构建流程的**成品结果表**，不是流程脚本）。
- 暂置未复制：`/home/ubuntu/aroma-dps-code/lunci10/lunci10-expanded-test.csv`（216 KB，2465 行，
  含骨架扩展行的早期版本结果表）——本次迁移清单未指定其去向，且已被 `lunci10-test-corrected.csv` 取代，留存工作目录。
- 本目录不含 `common/features.py`；未复制任何含 1-based ring index bug 的文件（规则 2）。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 或 `/home/ubuntu/data_90` 的 `.py` 文件：

- `run_g16_parallel.py`（默认参数 `GJF_DIR=/home/ubuntu/aroma-dps-code/lunci10/gaussian_inputs`、
  `SCRATCH_BASE=/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci10/scratch`，另有 `G16_BIN=/home/ubuntu/apps/g16/g16`）

`1_xtb.py` 不含上述两类绝对路径（`xtb` 依赖 `$PATH`，目录全部由命令行参数传入），但相对化时仍需确认
其对 `xyz_files/xtb_output/gaussian_inputs` 三级目录名的默认约定与仓库布局一致。
