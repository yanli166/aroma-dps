# data/collet — 三任务训练集注册表（0716 版）

- **论文面板归属**：Methods 数据来源 / Fig.2 / Fig.3（训练与测试标签的唯一来源）
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/code_end/data1_end/`（三个 CSV 原样 `cp`，已 `cmp` 核对一致）

## 已复制文件清单

| 文件 | 行数 | md5 |
|---|---|---|
| `collet_homa_0716.csv` | 6858 | `a8521cf4cc49422a6983fc492fe5b8fd` |
| `collet_nics_0716.csv` | 6820 | `2c5829ac817d212a43a65b61315375bf` |
| `collet_mbco_0716.csv` | 6927 | `e7afaa8b572e73790cb07ea2997fbfb7` |

列：`New_ID,smiles,Ring_ID,Ring_Size,atom_on_ring,<target>`（HOMA→`homa_value`、NICS→`NICS_value`、MBCO→`mbco_value`）；
`atom_on_ring` 为 **0-based**；NICS 表头含一个空的尾列，按原件保留未删。
上游生成管线见 `experiments/fig2_dataset/nics_pipeline/README.md`；完整口径与 md5 总表见 `../README.md`。

## 未复制 / 暂置清单

- 未复制：0702 旧表（`collet_homa_0702.csv` 等，位于 `/home/ubuntu/data_90/alldata_in_3090/model1/`）——
  SI 三层基线的 `experiments/si_baselines/common/tasks.py` 仍按旧表名寻址，如需复现基线原值需另行取回。
- 未复制：各批次 Gaussian/Multiwfn 中间产物——原始量子化学计算文件不随仓库分发。
- 本目录无 `.py` 文件，无路径相对化项。
