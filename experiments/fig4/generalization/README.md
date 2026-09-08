# Fig.4 泛化实验驱动脚本（工作目录顶层 fig4_*.py）

**归属面板**：Fig.4a–4d（外部零样本泛化 / internal↔external 对比 / 环族 LOO 热图 / 暴露度曲线）
与 Fig.4 的 transferability、cold-start 消融、SI 附表。

**迁移日期**：2026-09-08
**来源绝对路径**：`/home/ubuntu/aroma-dps-code/fig4_*.py`（6 个文件，逐字节 `cmp` 已核对一致，未改动一行代码）

| 文件 | 行数 | 写入的结果目录 | 角色 |
|------|------|----------------|------|
| `fig4_rerun_seed11.py` | 497 | `0901-end-code/results/fig4_lunci10_final_v2/` | **当前 Fig.4a–4d 出图主脚本**（fig4_combined.pdf/png + fig4a–4d CSV） |
| `fig4_retrain_v2.py` | 797 | `0901-end-code/results/fig4_generalization_retrain_v4{,_noe}/` | 主重训管线，`--noe` 切换无 E* 变体 |
| `fig4_generalization_retrain.py` | 900 | `.../fig4_generalization_retrain_v4/` | 上一代重训管线（同目录，二者存其一为准，见下"待确认"） |
| `fig4_coldstart_ablation.py` | 95 | `.../06_tables/coldstart_ablation_base_wo_e.csv` | 冷启动消融；`from fig4_retrain_v2 import ...` |
| `fig4_descriptors_transferability.py` | 831 | `0901-end-code/results/fig4_transferability_v2/` | 描述符可迁移性（HOMA/NICS/MBCO 互传） |
| `fig4_ring_family_audit.py` | — | `.../00_audit/ring_family_mapping.csv` | 环族分类器原始脚本；其算法已抽取为 `src/aroma_dps/chemistry/ring_family.py`（含两处修正，见该模块 docstring） |

## 为什么需要这一批脚本（重要）

仓库原有的 Fig.4 代码树 `0901-end-code/fig4_lunci10/` 把结果写到
`0901-end-code/results/fig4_lunci10_final/`，而该目录中
`02_novelty/`、`03_pairwise/`、`06_position/` **是空的**，`05_hammett/` **根本不存在**；
论文当前使用的 Fig.4 图件位于 `fig4_lunci10_final_v2/` 与
`fig4_generalization_retrain_v4_noe/`（2026-09-08 生成），由本目录脚本产出。
即：**`0901-end-code/fig4_lunci10/` 与当前 Fig.4 结果不是同一代**，
`experiments/fig4/launcher.py` 只覆盖前者，尚不指向本目录。

## 运行前必须改的路径（本次未改，保持逐字节可追溯）

6 个脚本全部硬编码工作目录，需改为仓库根：

1. `AROMA_PROJ = Path("/home/ubuntu/aroma-dps-code")` /
   `PROJ_ROOT = Path("/home/ubuntu/aroma-dps-code")` → 仓库根 `Path(__file__).resolve().parents[2]`
2. `BEST_PKG`（由 `AROMA_PROJ / "best_model_package"` 推出）→ 仓库中权重包在
   **`models/best_model_package/`**，目录层级不同，不能只改根路径
3. `fig4_coldstart_ablation.py:21-22` 的 `sys.path.insert(0, "/home/ubuntu/aroma-dps-code…")`
4. 结果目录：仓库内 `0901-end-code/results/` 已有同名结果，重跑会覆盖既有产物

## 待确认（不替用户决定）

- `fig4_generalization_retrain_v4/` 与 `..._v4_noe/` 哪个是正文版本；`_noe` 更新（09-08 16:14）。
- `fig4_generalization_retrain.py` 与 `fig4_retrain_v2.py` 写同一目录，谁是权威。
- 这批脚本是否正式并入 `experiments/fig4/launcher.py`（需要面板→脚本映射改写与 manifest 同步）。
