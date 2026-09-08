# data/lunci_external — 早期外部测试集（lunci6 / lunci7+8 / lunci10 扩展版）

- **论文面板归属**：Fig.4 / SI（外部泛化测试，早期批次）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/code_end/data1_end/`（原样 `cp`，已 `cmp` 核对一致）

## 已复制文件清单

| 文件 | 行数 | md5 | 说明 |
|---|---|---|---|
| `lunci6-test.csv` | 312 | `0d4a8db276b120a32ef8e6980c0e2441` | 取代基外部集，列无 `NICS_iso` |
| `lunci78-test.csv` | 610 | `283ddc757e77bc212965d4b8c8033711` | lunci7+lunci8 合并外部集 |
| `lunci10-test.csv` | 2465 | `b835da9001b7d6ecf94fa0f57541fb75` | lunci10 扩展版测试表（与 `data/lunci10/` 的 2153 行版本为不同切分，勿混用） |

## 未复制 / 暂置清单

- `code_end/data1_end/` 中除已列 6 个 CSV（3 个 `collet_*_0716` + 本目录 3 个）外无其他文件；
  对应批次的高斯/Multiwfn 原始计算目录（`lunci7_8_out_nics/`、`lunci8*/gaussian_inputs/`、`/home/ubuntu/data_90/...`）
  一律未复制——原始量子化学中间产物不随仓库分发。
- 本目录无 `.py` 文件，无路径相对化项。
