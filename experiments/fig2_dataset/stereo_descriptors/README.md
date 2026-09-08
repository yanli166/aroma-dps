# stereo_descriptors — 11 维立体描述符计算（ETKDG + MMFF 构象）

- **论文面板归属**：Methods / Fig.2 描述符体系中的立体（3D/构象）维度；同时服务 Fig.5 去芳构化反应表的立体化学标注。
- **迁移日期**：2026-09-08
- **来源绝对路径**：`/home/ubuntu/aroma-dps-code/compute_stereo_descriptors.py`
- **目标位置**：`/home/ubuntu/aroma-dps/experiments/fig2_dataset/stereo_descriptors/compute_stereo_descriptors.py`（内容与来源逐字节一致）

## 已复制文件清单

- `compute_stereo_descriptors.py`（单一脚本，无附属模块）

## 输入 / 输出与仓库内对应数据

脚本读入 `汇总.xlsx`、写出 `汇总_stereo.xlsx`（83 对去芳构化反应的手工表及其立体描述符扩展版）。
两份表已迁入本仓库：

- `data/reactions/汇总.xlsx`（md5 `f6e24a1b7616f487110881e4e7dd931d`）
- `data/reactions/汇总_stereo.xlsx`（md5 `68fc2fb01db1794a8a48c021dc6cd60f`）

## 未复制 / 暂置清单

- 无其他来源文件被排除：`/home/ubuntu/aroma-dps-code/` 下与立体描述符相关的产物仅上述两个 xlsx（均已迁入 `data/reactions/`）。
- 本目录不含 `common/features.py`；未复制任何含 1-based ring index bug 的文件（规则 2）。

## 仍需路径相对化的文件列表

含绝对路径 `/home/ubuntu/aroma-dps-code` 的 `.py` 文件：

- `compute_stereo_descriptors.py`（`XLSX_PATH = '/home/ubuntu/aroma-dps-code/汇总.xlsx'`、
  `OUTPUT_PATH = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'`，第 30–31 行）
  → 相对化后应指向仓库内 `data/reactions/汇总.xlsx` 与 `data/reactions/汇总_stereo.xlsx`（本次未改，规则 1）。
