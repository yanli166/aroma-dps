# data/reactions — 83 对策划去芳构化反应手工表

- **论文面板归属**：Fig.5（策划反应验证）；`汇总_stereo.xlsx` 亦支撑 Methods 的立体描述符与 SI 立体化学讨论
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/汇总.xlsx`、`/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx`
  （原样 `cp`，已 `cmp` 核对一致）

## 已复制文件清单

| 文件 | md5 | 说明 |
|---|---|---|
| `汇总.xlsx` | `f6e24a1b7616f487110881e4e7dd931d` | 反应物/产物 SMILES + 目标环索引的人工整理表（立体描述符脚本的输入） |
| `汇总_stereo.xlsx` | `68fc2fb01db1794a8a48c021dc6cd60f` | 上表 + 11 维立体描述符（ETKDG + MMFF 构象）扩展版 |

生成 `汇总_stereo.xlsx` 的代码：`experiments/fig2_dataset/stereo_descriptors/compute_stereo_descriptors.py`
（其 `XLSX_PATH` / `OUTPUT_PATH` 仍硬编码工作目录绝对路径，待相对化到本目录）。
出图脚本（`generate_*_figures.py`、`lunci8/predict_lunci8.py`）属 Fig.5 迁移任务范围，本任务未复制。

## 未复制 / 暂置清单

- 未复制：`/home/ubuntu/aroma-dps-code/lunci8/`、`lunci8-3/`、`lunci9_out/` 等反应批次的计算中间产物——不随仓库分发。
- 本目录无 `.py` 文件，无路径相对化项。
