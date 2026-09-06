# Fig.4 lunci10 — External Generalization Package

评估在 *lunci10* (strict OOD 配对测试集) 上对 HOMA / NICS_1zz / MBCO 三个
芳香性 / 磁化率 / 键级指标的 Siamese Δ-learning vs absolute-subtraction
差异。配套 Phase 14–16 与 Run Registry。

---

## 项目目标

1. 验证 Siamese Δ-MLP 在 strict OOD (lunci10) 上是否仍优于
   absolute subtraction 基线 (ΔA = A_i − A_j)。
2. 量化 pretraining (Stage-I checkpoint) 对 Siamese 收敛速度 / 数据效率
   的影响 (Phase 14)。
3. 检验结论对 Hammett anchor 选择 (F / Cl / OMe / 3-anchor avg) 的稳健性
   (Phase 15)。
4. 在 Siamese 协议下评估 layer4 的 M3 / M4 / M8 三种结构变体在 OOD 上的
   表现 (Phase 16)。

---

## 目录结构

```
fig4_lunci10/
├── __init__.py
├── README.md                       ← 本文件
├── configs/
│   └── fig4_lunci10.yaml           ← 单一 source of truth 的 config
├── data/
│   ├── build_manifest.py           ← Phase 1 input manifest 构造
│   ├── build_statistics.py         ← Phase 1 statistics 报告
│   └── build_pairs.py              ← Phase 2 配对构造 (lunci10 + internal)
├── audit/                          ← Phase 1: repository audit
│   ├── audit_repository.py
│   └── audit_overlap.py
├── evaluation/                     ← Phase 3, 4: lunci10 推理
│   ├── load_frozen_models.py
│   ├── run_external_absolute.py    ← 一次性绝对预测
│   └── run_external_delta.py       ← ΔA = A_i − A_j
├── training/                       ← Phase 9, 14: 训练 Siamese (内部)
│   ├── train_internal_pair_models.py    ← Phase 9
│   └── train_pretraining_control.py     ← Phase 14 (NEW)
├── analysis/                       ← Phase 5–8, 10–13, 15, 16
│   ├── scaffold_bias.py            ← Phase 5 / 10
│   ├── novelty_analysis.py         ← Phase 6 / 11
│   ├── position_analysis.py        ← Phase 7 / 12
│   ├── hammett_analysis.py         ← Phase 8 / 13
│   ├── anchor_analysis.py          ← Phase 15 (NEW)
│   └── oov_si_experiments.py       ← Phase 16 (NEW — M3/M4/M8)
└── reporting/                      ← Run registry + 主入口
    ├── run_registry.py             ← register_run / final_summary (NEW)
    ├── generate_fig4_report.py     ← Phase final
    └── run_all.py                  ← 主入口, 仅 plan / dry-run (NEW)
```

---

## Phase 执行顺序 (依赖 DAG)

线性 topological order:

1. `phase_01_audit`              — `audit/audit_repository.py` + `audit_overlap.py`
2. `phase_02_pairs`              — `data/build_pairs.py`
3. `phase_03_external_absolute`  — `evaluation/run_external_absolute.py`
4. `phase_04_external_delta`     — `evaluation/run_external_delta.py`
5. `phase_05_to_08_diagnostics`  — `analysis/{scaffold_bias, novelty, position, hammett}.py`
6. `phase_09_internal_pair_train`— `training/train_internal_pair_models.py`
7. `phase_10_to_13_diagnostics`  — `analysis/{scaffold_bias, novelty, position, hammett}.py` (二次)
8. `phase_14_pretraining_control`— `training/train_pretraining_control.py` ⭐ NEW
9. `phase_15_anchor_robustness`  — `analysis/anchor_analysis.py` ⭐ NEW
10. `phase_16_oov_si_methods`    — `analysis/oov_si_experiments.py` ⭐ NEW
11. `phase_final_report`         — `reporting/generate_fig4_report.py`

详细依赖图见 `python -m reporting.run_all --show-dag`。

依赖大致是:
-  14 → 9
-  15 → 14 + 3
-  16 → 9
-  final → 15 + 16

---

## 产物布局

```
/home/ubuntu/aroma-dps-code/0901-end-code/results/fig4_lunci10_final/
├── 00_audit/                                ← Phase 1
├── 01_external_absolute/                    ← Phase 3
├── 02_novelty/                              ← Phase 6 / 11
├── 03_pairwise/
│   ├── lunci10_pair_manifest.csv
│   ├── internal_pair_manifest.csv
│   ├── internal_pair_feasibility.json
│   ├── lunci10_pair_predictions.csv
│   └── fig4c_pairwise_summary.csv
├── 04_bias/                                 ← Phase 5 / 10
├── si_pretraining/
│   └── pretraining_transfer.csv             ← Phase 14 ⭐
├── si_anchor/
│   └── anchor_robustness.csv                ← Phase 15 ⭐
├── si_methods/
│   └── m3_m4_m8_summary.csv                 ← Phase 16 ⭐
├── run_registry.csv                         ← 所有 phases ⭐
└── run_registry_summary.md                  ← 自动生成
```

---

## RED FLAGS — 失败/可疑信号清单

1. **lunci10_pair_manifest.csv 为空**
   → 没有 (sub_i, sub_j) 配对可建 → 无法做 strict OOD。
   → 替代: within-lunci10 scaffold split (已 Phase 14 自动 fallback)。

2. **internal_pair_manifest.csv 不足 / 仅 lunci10 可建对**
   → `internal_pair_feasibility.json` 中
     `can_build_internal_pair = false` 是预期信号, 不应隐瞒。
   → Phase 9 不应回头用 lunci10 选超参; 若 *已* 选了, 标 status=violation。

3. **Stage-I pretraining checkpoint 缺失**
   → Phase 14 random_init vs stage_i_pretrained 第二个分支会标
     `factory_imported = False, status = missing_checkpoint`,
     *不要* 默默回退到 random_init。

4. **M3/M4/M8 (layer4) factory 导入失败**
   → `oov_si_experiments.py` 会标 status = `missing_layer4_module`。
   → *不要* 把 NaN metrics 当 OK 提交。

5. **scaler / checkpoint 在 lunci10 上 fit**
   → 触发 zero-shot 协议违例。Run Registry status = `violation`,
     不能进入 final summary。

6. **git commit 在 registry 上是空字符串**
   → 通常是 git 安装异常或仓库损坏。保留行, 但在评审中标注 "git_probe_failed"。

7. **run_id 撞**
   → UUID12 在 16M 次以内无撞概率; 若真撞, 第二次 register_run 会覆盖行
     (失败 runs 仍保留 — 见 registry.append 行为)。

8. **MAE 突然 ≈ 0**
   → 多半是泄漏 / 数据错配。Phase 14 会强制要求 split_seed != model_seed
     + 显式 evidence_kind = `internal_pair_fraction` vs
     `within_lunci10_adaptation` 来排查。

---

## 如何运行各 Phase

每个 Python 文件均带 `if __name__ == "__main__":` 入口, 可独立运行。

### Phase 14 — pretraining transfer
```bash
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/training/train_pretraining_control.py \
    --fractions 1.0 0.5 0.2 0.1 \
    --seeds 42 123 456 \
    --tasks HOMA NICS_1zz MBCO
```
产物:
```
results/fig4_lunci10_final/si_pretraining/pretraining_transfer.csv
```
若 only-lunci10 可配对, evidence_kind = `within_lunci10_adaptation`,
strict_external = False。脚本会读
`03_pairwise/internal_pair_feasibility.json` 自动切换。

### Phase 15 — anchor robustness
```bash
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/analysis/anchor_analysis.py
```
产物:
```
results/fig4_lunci10_final/si_anchor/anchor_robustness.csv
```
列含 `Q1_siamese_vs_subtraction`, `Q2_delta_learning_helps_HOMA_MBCO`,
`Q3_NICS_favors_subtraction`, `Q4_conclusion_anchor_sensitive` 四问答复。

### Phase 16 — OOV SI methods (M3/M4/M8)
```bash
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/analysis/oov_si_experiments.py
```
产物:
```
results/fig4_lunci10_final/si_methods/m3_m4_m8_summary.csv
```
⚠ 由于仓库中无 M3/M4/M8 Siamese 头与 Stage-I checkpoint,
脚本会标 status=`missing_layer4_module` 或
`needs_more_implementation`, 不假装跑成功。

---

## Run Registry

### 注册一个 run
```python
from run_registry import register_run
register_run(
    task="HOMA",
    model="RC_MPNN",
    dataset="lunci10",
    protocol="external_absolute",
    checkpoint="rc_mpnn_homa_best.pt",
    split_seed=42, model_seed=42,
    pretrained=False, feature_mode="atom_bond",
    train_N=0, test_N=10,
    MAE=0.07, RMSE=0.10, R2=0.85,
    status="ok",
    notes="first lunci10 absolute evaluation",
)
```

### 生成 summary
```bash
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/reporting/run_registry.py --summary
```

### 主入口 (不实际执行训练)
```bash
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/reporting/run_all.py --show-dag
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/reporting/run_all.py --plan-all      # dry-run + registry
python /home/ubuntu/aroma-dps-code/0901-end-code/fig4_lunci10/reporting/run_all.py --summary
```

### 失败 runs 不删除
任何 phase 触发异常, run_registry.register_run 应当仍然写入 status=`fail`
(失败原因进入 notes)。手写脚本时切勿用 try/except 静默吃掉。

---

## 约束 / 协议约定

- **strict_external** = True 仅在 protocol 标签包含 `external_*`,
  且未在 lunci10 上做 fit / 超参选择 / 早停的情况下才成立。
- 任何 pretraining fraction 必须仅由 internal_train_<task>.csv 提供。
- lunci10 manifest 只能用作 test-only, 不得进入 *fit* 任何变换。
- Run registry 是 audit trail — *不*允许重写历史; 只能追加。
