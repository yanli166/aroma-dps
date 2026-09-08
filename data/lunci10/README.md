# data/lunci10 — 外部基准（38 环骨架 × 30 取代基）

- **论文面板归属**：Fig.2（外部基准构成）/ Fig.4（泛化与 Δ 预测的主评测集）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/lunci10/`（原样 `cp`，已 `cmp` 核对一致）

## 已复制文件清单

| 文件 | 行数 | md5 | 说明 |
|---|---|---|---|
| `lunci10_unified.csv` | 2153 | `0c0a6f2db7d716f06dbf9a0f61509201` | 统一表：环/取代基元信息 + `HOMA,MBCO,NICS_iso,NICS_ZZ` |
| `lunci10-test-corrected.csv` | 2153 | `0c7d0549d9afe72a57813128b9ee83d6` | 修正版：`New_ID,SMILES,Ring_ID,Ring_Size,Ring_Atoms,...` |

两表同源同行数，字段命名不同（`ring_atoms` vs `Ring_Atoms`）；Fig.4 的 manifest 构建脚本引用其一，
迁移时未指定权威版本，故两版均入仓，由 `0901-end-code/fig4_lunci10/data/` 的构建脚本决定使用哪一版。
构建流程见 `experiments/fig2_dataset/lunci10_construction/README.md`。

## 未复制 / 暂置清单

- 未复制：`lunci10-expanded-test.csv`（216 KB，2465 行早期扩展版）——去向未在迁移清单内，留存工作目录。
- 未复制：`/home/ubuntu/aroma-dps-code/lunci10/gaussian_inputs/`（4174 文件 4.6 GB，含 60 个 >5 MB `.chk`）、
  `xtb_output/`（1389 目录 121 MB）、`xyz_files/`（1389 文件 5.5 MB）、`g16_run.log`、`xtb_run.log`——
  原始量子化学中间产物与日志，不随仓库分发。
- 本目录无 `.py` 文件，无路径相对化项。
