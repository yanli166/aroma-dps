# Archive: Deprecated Code

This directory contains deprecated scripts that are retained for historical
traceability but should NOT be used for new runs.

## What physically lives here (moved on 2026-09-08)

| Path here | Original path | Reason | Replacement |
|-----------|---------------|--------|-------------|
| `0831_v1_generation/` | `0831-end-code/{multiseed,lunci_test}/` + `orchestrate*.py/.sh`, `run_all.sh`, `run_all_parallel.sh`, `launch_stage4_parallel.sh`, `merge_stage{3,4}_parallel.py`, `resume_stage3_missing.py`, `rerun_ring_mask_pretrain.py` | Pre-v2 generation layer: test set derived from the model seed, results written to `results/` instead of `results_v2/` | `run_all_v2_parallel.sh` + the six `stage*` v2 scripts |
| `0831_v1_generation/run_pretrain_eval.py` | `0831-end-code/stage3_mask_pretraining/code/` | canonical_splits | `run_mask_pretrain_v2.py` |
| `0831_v1_generation/cross_arch_eval.py` | `0831-end-code/stage4_cross_architecture/code/` | canonical_splits | `run_stage4_v2.py` |
| `0831_v1_generation/gnn_baseline.py` | `0831-end-code/stage1_representation_comparison/code/` | Pseudo-code draft, cannot run | `run_stage1_v2.py` |
| `0901_v1_evaluation/run_external_absolute.py` | `0901-end-code/fig4_lunci10/evaluation/` | Full manifest includes exact_seen molecules | `run_external_absolute_v2.py` |
| `ml_cross_task.py` | `unified_models/` | PCA fitted before split (data leakage) | `ml_cross_task_v2.py` |
| `code_end/` | repo root | Superseded three-layer experiment infrastructure (`p0_verification/legacy_to_publication_mapping.csv` marks every file archive/superseded) | `0831-end-code/` (Fig.3), `0901-end-code/` (Fig.4), `src/aroma_dps/` |

`code_end/common/features.py` here is the **fixed** 0-based version (commit
d63f22c); the working directory at `/home/ubuntu/aroma-dps-code/` still carries
the 1-based variant, which is deliberately not published.

Related fix made at the same time: `0831-end-code/run_all_v2_parallel.sh`
(Stage 3) called `run_pretrain_eval.py`, so the documented one-command Fig.3
reproduction could emit non-publication Fig.3f numbers. It now calls
`run_mask_pretrain_v2.py`. **Verify which script produced the existing
`results_v2/stage3/` before re-running it.**

Still un-ported: `paired_delta_train.py` (v4) in the working directory
`/home/ubuntu/aroma-dps-code/code_end/`, superseded by
`paired_delta_v5_multimodel.py`.

## ⚠️ 这个归档不是"死代码"：publication 代码在运行时依赖它

移动 `code_end/` 后实测发现，**Fig.3 与 Fig.4 的 publication 脚本都在运行时依赖
`code_end/`**（`experiments/fig4/launcher.py` 与 `0831-end-code` 均间接依赖）：

| 依赖方 | 需要的 `code_end` 内容 | 用途 |
|--------|------------------------|------|
| `0901-end-code/fig4_lunci10/**/*.py`（20 个模块把 `code_end` 加进 `sys.path`） | `code_end/common/`（constants/tasks/graph_data/**features.py**） | 节点特征与图构建 |
| `0901-end-code/fig4_lunci10/{data,training}/` | `code_end/data1_end/*.csv` | collet 三任务训练/内部集输入 |
| `0901-end-code/fig4_lunci10/{analysis,audit}/` | `code_end/layer4_substituent/code/hammett_constants.py` | Hammett σ 表 |
| `0901-end-code/fig4_lunci10/training/train_l10_exposure.py` | `code_end/results/layer3_ring_fixed/seed_*/**/best_model.pth` | 冻结权重（重跑时） |
| `0831-end-code/common/tasks.py` | `code_end/data1_end/` 作为 `DATA1_END_DIR` 回退值 | 三任务数据根 |

因此本次（2026-09-08）做了两件事，使归档后这些依赖仍然成立：

1. **路径改指**：所有 `code_end` 路径常量改为 `archive/deprecated/code_end`
   （21 处 `PROJ_ROOT / "code_end"` / `os.path.join(...)` + 8 处 `f"{PROJ_ROOT}/code_end"`），
   并同步 `0831-end-code/common/tasks.py` 的回退目录。**只改路径常量，未改任何算法。**
2. **补回归档内被排除的数据件**（原移动时按"不复制 CSV"规则漏掉）：
   - `code_end/data1_end/` 6 个 CSV / 1.9 MB，`cmp` 与来源一致
     （其中 `collet_*_0716.csv` 与 `data/collet/` 下的副本字节相同；
     **canonical 输入位置是 `data/`，这里是归档自洽副本**）
   - `code_end/results/{layer2_gnn,layer3_ring_fixed}/` 的 **1192 个 CSV / 17 MB**

**未迁入（外部依赖）**：`layer3_ring_fixed` 的 375 个 `best_model.pth`（291 MB）仍留在
`/home/ubuntu/aroma-dps-code/code_end/results/layer3_ring_fixed/`，只有重跑
`train_l10_exposure.py` 才需要。

**后续应当做的（本轮未做，属于重构而非搬运）**：把 Fig.3/Fig.4 的 `common.*`
导入改到 `src/aroma_dps/`（featurization/data 已有等价实现），数据读取改指 `data/`，
才能让这个归档真正变成只读历史。注意 `code_end/common/features.py` 是**已修复的 0-based 版本**，
迁移时不得从工作目录取回 1-based 变体。

## Policy

- **Do NOT modify** deprecated scripts
- **Do NOT delete** deprecated scripts
- Historical results from these scripts are labeled "non-publication" in
  `p0_verification/active_run_protocol_audit.csv`
- `canonical_splits()` in `src/aroma_dps/data/splits.py` is kept as a
  deprecation wrapper with `DeprecationWarning` for 6-12 months
- Deprecated code is moved here as a *whole dependency cluster*: the moved
  scripts import each other (e.g. `run_multiseed.py` imports
  `run_pretrain_eval.py`), so splitting one file out of the cluster would
  break the rest
