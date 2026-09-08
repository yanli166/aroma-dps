# Fig.6 v2 — DATA_AUDIT

## 1. Target pair 数
- Tier-A **target** pairs total: **4443** (file: dearom_ring_pairs_A_tierA/ring_pairs_ml.csv)
- Tier-A **complete** R/P predictions (used in Fig.5): **4377** (file: ring_pair_aromaticity_predictions_complete.csv)
- **All ring pairs** (across all reactions, all tiers): **10221**
- Stage-2 tier distribution: {'R': 5722, 'A': 4443, 'C': 40, 'B': 16}

## 2. Unique reaction 数
- Tier-A reactions: **4306**
- All-rings reactions (same 4306): **4306**

## 3. 每反应 target ring 数
- Reactions with multiple target rings: **137** of 4306 (3.2%)
- Distribution: {1: 4169, 2: 137}
→ All downstream clustered bootstrap 必须以 reaction_id 为 cluster.

## 4. Spectator ring 数
- Tier=R spectator rings: **5722**
- Spectator rings with existing prediction (intersection with predictions file): **0**
- **Spectator rings WITHOUT prediction (must be predicted before Panel B):** **5722**

## 5. Target / Spectator 预测覆盖
- Pairs with both R/P PASS: 4383
- Tier-A target pairs with both PASS: 4383
- Spectator pairs with both PASS (will be enriched in Panel B prep): 0

## 6. NaN / inf in descriptors
- `HOMA_pred_reactant`: {'nan': 0, 'inf': 0}
- `HOMA_pred_product`: {'nan': 0, 'inf': 0}
- `nMCBO_pred_reactant`: {'nan': 0, 'inf': 0}
- `nMCBO_pred_product`: {'nan': 0, 'inf': 0}
- `NICS_1zz_pred_reactant`: {'nan': 0, 'inf': 0}
- `NICS_1zz_pred_product`: {'nan': 0, 'inf': 0}

## 7. pair_id 唯一性
- duplicate pair_id rows in complete: **0** (expected 0)

## 8. reaction_id 可用于 clustered analysis
- reactions with at least one Tier-A target: **4306**
- reactions with at least one complete R/P prediction: **4243**

## 9. Canonicalization / index mismatch
- samples REINDEXED: 0 / 8886 (0.0%)
- pairs touched by reindexing: 0
→ `fix_ring_indices.py` has corrected all of them; current predictions are safe to use.

## 10. Ring family n per family
- n families: 16
- counts (top 10):
  - quinoline_or_isoquinoline: 962
  - pyridine: 696
  - indole: 512
  - other N-heteroarene: 447
  - other carbocycle: 347
  - benzene: 322
  - other O-heteroarene: 247
  - phenol-type: 210
  - other S-heteroarene: 163
  - benzofuran: 145

## Excluded records
- See `excluded_records.csv` (pairs excluded from 'complete' due to incomplete R/P prediction).
- Total excluded: **66** pairs (these are Tier-A target pairs but one side failed model prediction).

## Verdict
Data is internally consistent. **Action required before Panel B:**
1. Predict **5722** missing spectator rings with frozen model.
2. Maintain target/spectator pair_id distinction in the prediction cache.
3. Use reaction_id for clustered bootstrap in all panel statistics.