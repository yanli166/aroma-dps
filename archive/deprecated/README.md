# Archive: Deprecated Code

This directory contains deprecated scripts that are retained for historical
traceability but should NOT be used for new runs.

## Deprecated Scripts (not modified, not deleted)

| Script | Reason | Replacement |
|--------|--------|-------------|
| `canonical_splits()` | Split seed = model seed (variable test set) | `get_final_splits(SPLIT_SEED=2026)` |
| `run_pretrain_eval.py` | Uses canonical_splits | `run_mask_pretrain_v2.py` |
| `cross_arch_eval.py` | Uses canonical_splits | `run_stage4_v2.py` |
| `run_multiseed.py` | Uses canonical_splits | Individual v2 stage scripts |
| `run_external_absolute.py` (v1) | Full manifest (includes exact_seen) | `run_external_absolute_v2.py` |
| `ml_cross_task.py` (v1) | PCA data leakage | `ml_cross_task_v2.py` |
| `paired_delta_train.py` (v4) | scaffold_group_split v4 flaw | `paired_delta_v5_multimodel.py` |
| `gnn_baseline.py` | Pseudo-code draft, cannot run | `run_stage1_v2.py` |

## Policy

- **Do NOT modify** deprecated scripts
- **Do NOT delete** deprecated scripts
- Historical results from these scripts are labeled "non-publication" in
  `p0_verification/active_run_protocol_audit.csv`
- `canonical_splits()` in `src/aroma_dps/data/splits.py` is kept as a
  deprecation wrapper with `DeprecationWarning` for 6-12 months
