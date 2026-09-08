#!/usr/bin/env python3
"""
Generate legacy_to_publication_mapping.csv

逐文件标注每个 .py 文件的迁移策略。
不移动任何文件，仅生成 mapping。
"""
import csv
from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/aroma-dps/p0_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each row: (legacy_path, role, manuscript_figure, action, new_path, publication_required, numerical_behavior_change)
ROWS = []

def add(legacy, role, fig, action, new, pub, num_change):
    ROWS.append({
        "legacy_path": legacy,
        "role": role,
        "manuscript_figure": fig,
        "action": action,
        "new_path": new,
        "publication_required": pub,
        "numerical_behavior_change": num_change,
    })

# ============================================================================
# 0831-end-code/ — Fig.3 publication code
# ============================================================================

# --- common/ (shared infrastructure) ---
add("0831-end-code/common/constants.py", "paths_constants", "Shared", "migrate_to_src",
    "src/aroma_dps/config.py", "yes", "no")
add("0831-end-code/common/estar_pipeline.py", "training_estar", "Shared", "migrate_to_src",
    "src/aroma_dps/training/estar.py", "yes", "no")
add("0831-end-code/common/features.py", "ring_descriptors", "Shared", "migrate_to_src",
    "src/aroma_dps/featurization/features.py", "yes", "no")
add("0831-end-code/common/graph_data.py", "graph_construction", "Shared", "migrate_to_src",
    "src/aroma_dps/featurization/graph_data.py", "yes", "no")
add("0831-end-code/common/__init__.py", "package_init", "Shared", "archive",
    "archive/legacy_0831/common/__init__.py", "no", "no")
add("0831-end-code/common/protocol.py", "split_logic", "Shared", "migrate_to_src",
    "src/aroma_dps/data/splits.py", "yes", "no")
add("0831-end-code/common/smoke_test.py", "smoke_test", "Shared", "archive",
    "archive/legacy_0831/common/smoke_test.py", "no", "no")
add("0831-end-code/common/tasks.py", "dataset_metrics_split", "Shared", "migrate_to_src",
    "src/aroma_dps/data/datasets.py + src/aroma_dps/evaluation/metrics.py", "yes", "no")
add("0831-end-code/common/train_eval.py", "training_loop", "Shared", "migrate_to_src",
    "src/aroma_dps/training/trainer.py", "yes", "no")

# --- models/ (model definitions) ---
add("0831-end-code/models/__init__.py", "package_init", "Shared", "archive",
    "archive/legacy_0831/models/__init__.py", "no", "no")
add("0831-end-code/models/pyg_models.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/mpnn.py", "yes", "no")
add("0831-end-code/models/ring_conditioned_gnn.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/ring_conditioned.py", "yes", "no")
add("0831-end-code/models/ring_readout.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/ring_readout.py", "yes", "no")

# --- stage1/ (Fig.3: Representation comparison) ---
add("0831-end-code/stage1_representation_comparison/code/run_stage1_v2.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage1_representation/launcher.py", "yes", "no")
add("0831-end-code/stage1_representation_comparison/code/traditional_ml.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage1_representation/traditional_ml.py", "yes", "no")
add("0831-end-code/stage1_representation_comparison/code/gnn_baseline.py", "broken_draft", "Fig.3", "archive",
    "archive/legacy_0831/stage1/gnn_baseline.py", "no", "no")
add("0831-end-code/stage1_representation_comparison/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage1/__init__.py", "no", "no")
add("0831-end-code/stage1_representation_comparison/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage1_outer/__init__.py", "no", "no")

# --- stage2/ (Fig.3: 2x2 factorial ablation) ---
add("0831-end-code/stage2_ring_conditioning/code/ring_conditioning_ablation.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage2_factorial_ablation/launcher.py", "yes", "no")
add("0831-end-code/stage2_ring_conditioning/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage2/__init__.py", "no", "no")
add("0831-end-code/stage2_ring_conditioning/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage2_outer/__init__.py", "no", "no")

# --- stage3/ (Fig.3: Mask pretraining) ---
add("0831-end-code/stage3_mask_pretraining/code/run_mask_pretrain_v2.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage3_mask_pretraining/launcher.py", "yes", "no")
add("0831-end-code/stage3_mask_pretraining/code/mask_pretrain.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/masked_autoencoder.py", "yes", "no")
add("0831-end-code/stage3_mask_pretraining/code/run_pretrain_eval.py", "deprecated_launcher", "Fig.3", "archive",
    "archive/legacy_0831/stage3/run_pretrain_eval.py", "no", "no")
add("0831-end-code/stage3_mask_pretraining/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage3/__init__.py", "no", "no")
add("0831-end-code/stage3_mask_pretraining/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage3_outer/__init__.py", "no", "no")

# --- stage4/ (Fig.3: Cross-architecture) ---
add("0831-end-code/stage4_cross_architecture/code/run_stage4_v2.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage4_cross_architecture/launcher.py", "yes", "no")
add("0831-end-code/stage4_cross_architecture/code/aggregate_stage4_v2.py", "aggregation", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage4_cross_architecture/aggregate.py", "yes", "no")
add("0831-end-code/stage4_cross_architecture/code/cross_arch_eval.py", "deprecated_launcher", "Fig.3", "archive",
    "archive/legacy_0831/stage4/cross_arch_eval.py", "no", "no")
add("0831-end-code/stage4_cross_architecture/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage4/__init__.py", "no", "no")
add("0831-end-code/stage4_cross_architecture/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stage4_outer/__init__.py", "no", "no")

# --- stage5/ (Fig.3: Ring flag sensitivity) ---
add("0831-end-code/stage5_ring_flag_sensitivity/code/run_ring_flag_sensitivity.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage5_ring_flag_sensitivity/launcher.py", "yes", "no")

# --- stage6/ (Fig.3: Final membership) ---
add("0831-end-code/stage6_final_membership/code/run_final_membership.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/stage6_final_membership/launcher.py", "yes", "no")

# --- stats/ ---
add("0831-end-code/stats/significance_test.py", "statistical_test", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/aggregation/significance_test.py", "yes", "no")
add("0831-end-code/stats/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/stats/__init__.py", "no", "no")

# --- orchestration / multiseed / utilities ---
add("0831-end-code/orchestrate.py", "orchestration", "Fig.3", "archive",
    "archive/legacy_0831/orchestrate.py", "no", "no")
add("0831-end-code/multiseed/run_multiseed.py", "deprecated_launcher", "Fig.3", "archive",
    "archive/legacy_0831/multiseed/run_multiseed.py", "no", "no")
add("0831-end-code/multiseed/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_0831/multiseed/__init__.py", "no", "no")
add("0831-end-code/lunci_test/run_lunci_test.py", "deprecated_launcher", "Fig.4", "archive",
    "archive/legacy_0831/lunci_test/run_lunci_test.py", "no", "no")
add("0831-end-code/rerun_ring_mask_pretrain.py", "one_time_script", "Fig.3", "archive",
    "archive/legacy_0831/rerun_ring_mask_pretrain.py", "no", "no")
add("0831-end-code/resume_stage3_missing.py", "one_time_script", "Fig.3", "archive",
    "archive/legacy_0831/resume_stage3_missing.py", "no", "no")
add("0831-end-code/merge_stage3_parallel.py", "one_time_script", "Fig.3", "archive",
    "archive/legacy_0831/merge_stage3_parallel.py", "no", "no")
add("0831-end-code/merge_stage4_parallel.py", "one_time_script", "Fig.3", "archive",
    "archive/legacy_0831/merge_stage4_parallel.py", "no", "no")
add("0831-end-code/__init__.py", "package_init", "Shared", "archive",
    "archive/legacy_0831/__init__.py", "no", "no")

# ============================================================================
# 0901-end-code/ — Fig.4 publication code
# ============================================================================

# --- data/ ---
add("0901-end-code/fig4_lunci10/data/build_clean_split.py", "data_preparation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/data/build_clean_split.py", "yes", "no")
add("0901-end-code/fig4_lunci10/data/build_manifest.py", "deprecated_data_prep", "Fig.4", "archive",
    "archive/legacy_0901/data/build_manifest.py", "no", "no")
add("0901-end-code/fig4_lunci10/data/build_manifest_v2.py", "data_preparation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/data/build_manifest.py", "yes", "no")
add("0901-end-code/fig4_lunci10/data/build_pairs.py", "data_preparation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/data/build_pairs.py", "yes", "no")
add("0901-end-code/fig4_lunci10/data/build_scaffold_splits.py", "data_preparation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/data/build_scaffold_splits.py", "yes", "no")
add("0901-end-code/fig4_lunci10/data/build_statistics.py", "data_preparation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/data/build_statistics.py", "yes", "no")
add("0901-end-code/fig4_lunci10/data/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/data/__init__.py", "no", "no")

# --- training/ ---
add("0901-end-code/fig4_lunci10/training/train_scaffold_ood.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/scaffold_ood.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/train_exposure_curve.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/exposure_curve.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/train_l10_exposure.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/ring_family_ood.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/train_internal_pair_models.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/internal_pairs.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/train_pretraining_control.py", "stub_placeholder", "Fig.4", "archive",
    "archive/legacy_0901/training/train_pretraining_control.py", "no", "no")
add("0901-end-code/fig4_lunci10/training/retrain_frozen.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/retrain_frozen.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/retrain_frozen_mpnn_label.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/retrain_frozen_mpnn.py", "yes", "no")
add("0901-end-code/fig4_lunci10/training/check_internal_pair.py", "one_time_check", "Fig.4", "archive",
    "archive/legacy_0901/training/check_internal_pair.py", "no", "no")
add("0901-end-code/fig4_lunci10/training/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/training/__init__.py", "no", "no")

# --- evaluation/ ---
add("0901-end-code/fig4_lunci10/evaluation/load_frozen_models.py", "frozen_model_loading", "Shared", "migrate_to_src",
    "src/aroma_dps/inference/frozen_models.py", "yes", "no")
add("0901-end-code/fig4_lunci10/evaluation/merge_predictions.py", "thin_wrapper", "Fig.4", "archive",
    "archive/legacy_0901/evaluation/merge_predictions.py", "no", "no")
add("0901-end-code/fig4_lunci10/evaluation/run_external_absolute.py", "deprecated_evaluation", "Fig.4", "archive",
    "archive/legacy_0901/evaluation/run_external_absolute.py", "no", "no")
add("0901-end-code/fig4_lunci10/evaluation/run_external_absolute_v2.py", "evaluation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/evaluation/external_absolute.py", "yes", "no")
add("0901-end-code/fig4_lunci10/evaluation/run_external_delta.py", "evaluation", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/evaluation/external_delta.py", "yes", "no")
add("0901-end-code/fig4_lunci10/evaluation/run_external_smoke.py", "smoke_test", "Fig.4", "archive",
    "archive/legacy_0901/evaluation/run_external_smoke.py", "no", "no")
add("0901-end-code/fig4_lunci10/evaluation/run_external_v2_subset.py", "parallelization_wrapper", "Fig.4", "archive",
    "archive/legacy_0901/evaluation/run_external_v2_subset.py", "no", "no")
add("0901-end-code/fig4_lunci10/evaluation/scan_frozen_provenance.py", "provenance_scan", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/audit/scan_provenance.py", "yes", "no")
add("0901-end-code/fig4_lunci10/evaluation/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/evaluation/__init__.py", "no", "no")

# --- analysis/ ---
add("0901-end-code/fig4_lunci10/analysis/anchor_analysis.py", "analysis", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/analysis/anchor_analysis.py", "yes", "no")
add("0901-end-code/fig4_lunci10/analysis/hammett_analysis.py", "analysis", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/analysis/hammett_analysis.py", "yes", "no")
add("0901-end-code/fig4_lunci10/analysis/novelty_analysis.py", "analysis", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/analysis/novelty_analysis.py", "yes", "no")
add("0901-end-code/fig4_lunci10/analysis/oov_si_experiments.py", "stub_placeholder", "Fig.4", "archive",
    "archive/legacy_0901/analysis/oov_si_experiments.py", "no", "no")
add("0901-end-code/fig4_lunci10/analysis/position_analysis.py", "analysis", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/analysis/position_analysis.py", "yes", "no")
add("0901-end-code/fig4_lunci10/analysis/scaffold_bias.py", "analysis", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/analysis/scaffold_bias.py", "yes", "no")
add("0901-end-code/fig4_lunci10/analysis/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/analysis/__init__.py", "no", "no")

# --- audit/ ---
add("0901-end-code/fig4_lunci10/audit/audit_frozen_provenance.py", "audit", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/audit/frozen_provenance.py", "yes", "no")
add("0901-end-code/fig4_lunci10/audit/audit_mbco_extreme.py", "audit", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/audit/mcbo_extreme.py", "yes", "no")
add("0901-end-code/fig4_lunci10/audit/audit_multiplicity.py", "audit", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/audit/multiplicity.py", "yes", "no")
add("0901-end-code/fig4_lunci10/audit/audit_overlap.py", "audit", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/audit/overlap.py", "yes", "no")
add("0901-end-code/fig4_lunci10/audit/audit_repository.py", "audit", "Fig.4", "archive",
    "archive/legacy_0901/audit/audit_repository.py", "no", "no")
add("0901-end-code/fig4_lunci10/audit/audit_substituent_feasibility.py", "audit", "Fig.4", "archive",
    "archive/legacy_0901/audit/audit_substituent_feasibility.py", "no", "no")
add("0901-end-code/fig4_lunci10/audit/audit_target_ring_id_semantics.py", "audit", "Fig.4", "archive",
    "archive/legacy_0901/audit/audit_target_ring_id_semantics.py", "no", "no")
add("0901-end-code/fig4_lunci10/audit/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/audit/__init__.py", "no", "no")

# --- reporting/ ---
add("0901-end-code/fig4_lunci10/reporting/generate_fig4_report.py", "reporting", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/reporting/generate_report.py", "yes", "no")
add("0901-end-code/fig4_lunci10/reporting/molecule_cluster_bootstrap.py", "statistical_utility", "Shared", "migrate_to_src",
    "src/aroma_dps/evaluation/bootstrap.py", "yes", "no")
add("0901-end-code/fig4_lunci10/reporting/run_all.py", "reporting", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/reporting/run_all.py", "yes", "no")
add("0901-end-code/fig4_lunci10/reporting/run_registry.py", "provenance_registry", "Shared", "migrate_to_src",
    "src/aroma_dps/training/checkpoint.py", "yes", "no")
add("0901-end-code/fig4_lunci10/reporting/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/reporting/__init__.py", "no", "no")

# --- configs/ ---
add("0901-end-code/fig4_lunci10/configs/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/configs/__init__.py", "no", "no")

# --- root ---
add("0901-end-code/fig4_lunci10/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_0901/__init__.py", "no", "no")

# ============================================================================
# code_end/ — Legacy, mostly archive
# ============================================================================

# --- common/ (duplicates of 0831 common) ---
add("code_end/common/constants.py", "duplicate_infrastructure", "Shared", "archive",
    "archive/legacy_code_end/common/constants.py", "no", "no")
add("code_end/common/features.py", "duplicate_infrastructure", "Shared", "archive",
    "archive/legacy_code_end/common/features.py", "no", "no")
add("code_end/common/graph_data.py", "duplicate_infrastructure", "Shared", "archive",
    "archive/legacy_code_end/common/graph_data.py", "no", "no")
add("code_end/common/tasks.py", "duplicate_infrastructure", "Shared", "archive",
    "archive/legacy_code_end/common/tasks.py", "no", "no")
add("code_end/common/__init__.py", "package_init", "Shared", "archive",
    "archive/legacy_code_end/common/__init__.py", "no", "no")

# --- baseline_gnn/ (superseded by 0831) ---
add("code_end/baseline_gnn/code/gnn_train_eval.py", "superseded_training", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_gnn/gnn_train_eval.py", "no", "no")
add("code_end/baseline_gnn/code/pyg_models.py", "superseded_model", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_gnn/pyg_models.py", "no", "no")
add("code_end/baseline_gnn/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_gnn/__init__.py", "no", "no")
add("code_end/baseline_gnn/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_gnn_outer/__init__.py", "no", "no")

# --- baseline_traditional_ml/ (superseded by 0831) ---
add("code_end/baseline_traditional_ml/code/ml_train_eval.py", "superseded_training", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_traditional_ml/ml_train_eval.py", "no", "no")
add("code_end/baseline_traditional_ml/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_traditional_ml/__init__.py", "no", "no")
add("code_end/baseline_traditional_ml/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/baseline_traditional_ml_outer/__init__.py", "no", "no")

# --- generalization_test/ (superseded by 0901) ---
add("code_end/generalization_test/code/splits.py", "superseded_split", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/splits.py", "no", "no")
add("code_end/generalization_test/code/train_eval.py", "superseded_training", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/train_eval.py", "no", "no")
add("code_end/generalization_test/code/ring_utils.py", "ring_family_identification", "Shared", "migrate_to_src",
    "src/aroma_dps/chemistry/ring_family.py", "yes", "no")
add("code_end/generalization_test/code/fixed_ood_benchmark.py", "superseded_launcher", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/fixed_ood_benchmark.py", "no", "no")
add("code_end/generalization_test/code/fixed_ood_multiseed.py", "superseded_launcher", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/fixed_ood_multiseed.py", "no", "no")
add("code_end/generalization_test/code/run_learning_curve_rerun.py", "superseded_launcher", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/run_learning_curve_rerun.py", "no", "no")
add("code_end/generalization_test/code/step1_audit.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step1_audit.py", "no", "no")
add("code_end/generalization_test/code/step2_build_pair_dataset.py", "superseded_data_prep", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step2_build_pair_dataset.py", "no", "no")
add("code_end/generalization_test/code/step3_subtraction_baseline.py", "superseded_baseline", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step3_subtraction_baseline.py", "no", "no")
add("code_end/generalization_test/code/step4_delta_fingerprint.py", "superseded_baseline", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step4_delta_fingerprint.py", "no", "no")
add("code_end/generalization_test/code/step5_siamese_mpnn.py", "superseded_model", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step5_siamese_mpnn.py", "no", "no")
add("code_end/generalization_test/code/step6_pretraining_transfer.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step6_pretraining_transfer.py", "no", "no")
add("code_end/generalization_test/code/step7_bias_decomposition.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step7_bias_decomposition.py", "no", "no")
add("code_end/generalization_test/code/step8_delta_magnitude.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step8_delta_magnitude.py", "no", "no")
add("code_end/generalization_test/code/step9_error_analysis.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/step9_error_analysis.py", "no", "no")
add("code_end/generalization_test/code/anchor_step1_selection.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step1_selection.py", "no", "no")
add("code_end/generalization_test/code/anchor_step2_build_dataset.py", "superseded_data_prep", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step2_build_dataset.py", "no", "no")
add("code_end/generalization_test/code/anchor_step3_run_models.py", "superseded_launcher", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step3_run_models.py", "no", "no")
add("code_end/generalization_test/code/anchor_step7_robustness.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step7_robustness.py", "no", "no")
add("code_end/generalization_test/code/anchor_step8_pretraining.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step8_pretraining.py", "no", "no")
add("code_end/generalization_test/code/anchor_step8_pretraining_gpu2test.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/anchor_step8_pretraining_gpu2test.py", "no", "no")
add("code_end/generalization_test/code/hard_ring_type_table.py", "superseded_utility", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/hard_ring_type_table.py", "no", "no")
add("code_end/generalization_test/code/hard_ring_type_v2.py", "superseded_utility", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/hard_ring_type_v2.py", "no", "no")
add("code_end/generalization_test/code/lunci10_three_layer_analysis.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/lunci10_three_layer_analysis.py", "no", "no")
add("code_end/generalization_test/code/ood_difficulty_analysis.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/ood_difficulty_analysis.py", "no", "no")
add("code_end/generalization_test/code/p1c_leakage_ablation.py", "superseded_analysis", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/p1c_leakage_ablation.py", "no", "no")
add("code_end/generalization_test/code/plot_exp_comparison.py", "superseded_plotting", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/plot_exp_comparison.py", "no", "no")
add("code_end/generalization_test/code/run_all.py", "superseded_orchestration", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/run_all.py", "no", "no")
add("code_end/generalization_test/code/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test/__init__.py", "no", "no")
add("code_end/generalization_test/__init__.py", "package_init", "Fig.4", "archive",
    "archive/legacy_code_end/generalization_test_outer/__init__.py", "no", "no")

# --- aromatic_split/ (superseded by 0831 stage6) ---
add("code_end/aromatic_split/code/run_aromatic_split.py", "superseded_launcher", "Fig.3", "archive",
    "archive/legacy_code_end/aromatic_split/run_aromatic_split.py", "no", "no")
add("code_end/aromatic_split/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/aromatic_split/__init__.py", "no", "no")
add("code_end/aromatic_split/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/aromatic_split_outer/__init__.py", "no", "no")

# --- ring_encoding_ablation/ (superseded by 0831 stage2) ---
add("code_end/ring_encoding_ablation/code/combined_ragcn.py", "superseded_model", "Fig.3", "archive",
    "archive/legacy_code_end/ring_encoding_ablation/combined_ragcn.py", "no", "no")
add("code_end/ring_encoding_ablation/code/ring_train_eval.py", "superseded_training", "Fig.3", "archive",
    "archive/legacy_code_end/ring_encoding_ablation/ring_train_eval.py", "no", "no")
add("code_end/ring_encoding_ablation/code/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/ring_encoding_ablation/__init__.py", "no", "no")
add("code_end/ring_encoding_ablation/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/ring_encoding_ablation_outer/__init__.py", "no", "no")

# --- layer4_substituent/ (SI experiments, not main pipeline) ---
for f in ["__init__.py", "code/__init__.py", "code/hammett_constants.py",
          "code/m1_hammett_embedding.py", "code/m2_monotonicity_loss.py",
          "code/m3_hierarchical_attention.py", "code/m4_dual_channel.py",
          "code/m5_ssl_pretrain.py", "code/m6_position_encoding.py",
          "code/m7_perturbation.py", "code/m8_multi_scale_pooling.py",
          "code/rebuild_summary.py", "code/run_all.py", "code/train_eval.py"]:
    add(f"code_end/layer4_substituent/{f}", "si_experiment", "Fig.4", "archive",
        f"archive/legacy_code_end/layer4_substituent/{f.replace('/', '_')}", "no", "no")

# --- optuna_search/ (hyperparameter search, not publication) ---
add("code_end/optuna_search/code/optuna_search.py", "hyperparameter_search", "N/A", "archive",
    "archive/legacy_code_end/optuna_search/optuna_search.py", "no", "no")
add("code_end/optuna_search/code/__init__.py", "package_init", "N/A", "archive",
    "archive/legacy_code_end/optuna_search/__init__.py", "no", "no")
add("code_end/optuna_search/__init__.py", "package_init", "N/A", "archive",
    "archive/legacy_code_end/optuna_search_outer/__init__.py", "no", "no")

# --- multiseed/ (deprecated) ---
add("code_end/multiseed/run_multiseed.py", "deprecated_launcher", "N/A", "archive",
    "archive/legacy_code_end/multiseed/run_multiseed.py", "no", "no")
add("code_end/multiseed/__init__.py", "package_init", "N/A", "archive",
    "archive/legacy_code_end/multiseed/__init__.py", "no", "no")

# --- other code_end/ ---
add("code_end/generate_final_report.py", "superseded_reporting", "N/A", "archive",
    "archive/legacy_code_end/generate_final_report.py", "no", "no")
add("code_end/generate_report.py", "superseded_reporting", "N/A", "archive",
    "archive/legacy_code_end/generate_report.py", "no", "no")
add("code_end/layer1_lunci6_test.py", "superseded_test", "Fig.4", "archive",
    "archive/legacy_code_end/layer1_lunci6_test.py", "no", "no")
add("code_end/lunci6_substituent_diag.py", "superseded_diagnostic", "Fig.4", "archive",
    "archive/legacy_code_end/lunci6_substituent_diag.py", "no", "no")
add("code_end/paired_delta_train.py", "deprecated_v4", "Fig.4", "archive",
    "archive/legacy_code_end/paired_delta_train.py", "no", "no")
add("code_end/paired_delta_v5_multimodel.py", "experiment_launcher", "Fig.4", "migrate_to_experiments",
    "experiments/fig4/training/paired_delta.py", "yes", "no")
add("code_end/part6_gate_check.py", "superseded_check", "Fig.4", "archive",
    "archive/legacy_code_end/part6_gate_check.py", "no", "no")
add("code_end/test_attentivefp_lunci6.py", "superseded_test", "Fig.4", "archive",
    "archive/legacy_code_end/test_attentivefp_lunci6.py", "no", "no")
add("code_end/test_gat_lunci6.py", "superseded_test", "Fig.4", "archive",
    "archive/legacy_code_end/test_gat_lunci6.py", "no", "no")
add("code_end/stats/significance_test.py", "duplicate_stats", "Fig.3", "archive",
    "archive/legacy_code_end/stats/significance_test.py", "no", "no")
add("code_end/stats/__init__.py", "package_init", "Fig.3", "archive",
    "archive/legacy_code_end/stats/__init__.py", "no", "no")

# ============================================================================
# unified_models/ — Upstream source, model definitions migrate to src/
# ============================================================================

# --- common/ (core infrastructure) ---
add("unified_models/common/graphs.py", "graph_construction", "Shared", "migrate_to_src",
    "src/aroma_dps/featurization/graph.py", "yes", "no")
add("unified_models/common/utils.py", "training_utilities", "Shared", "migrate_to_src",
    "src/aroma_dps/training/trainer.py", "yes", "no")

# --- Model definitions (5 backbones) ---
add("unified_models/gnn/model.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/gnn.py", "yes", "no")
add("unified_models/gin/model.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/gin.py", "yes", "no")
add("unified_models/gat/model.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/gat.py", "yes", "no")
add("unified_models/mpnn/model.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/mpnn_base.py", "yes", "no")
add("unified_models/graphsage/model.py", "model_definition", "Shared", "migrate_to_src",
    "src/aroma_dps/models/graphsage.py", "yes", "no")

# --- Feature/model utilities ---
add("unified_models/ml_models.py", "feature_computation", "Shared", "migrate_to_src",
    "src/aroma_dps/featurization/features.py", "yes", "no")

# --- Training/optimization (superseded by 0831) ---
add("unified_models/train.py", "superseded_training", "Fig.3", "archive",
    "archive/legacy_unified_models/train.py", "no", "no")
add("unified_models/three_task_eval.py", "superseded_evaluation", "Fig.3", "archive",
    "archive/legacy_unified_models/three_task_eval.py", "no", "no")
add("unified_models/optimize.py", "hyperparameter_search", "N/A", "archive",
    "archive/legacy_unified_models/optimize.py", "no", "no")
add("unified_models/optimize_all_15.py", "hyperparameter_search", "N/A", "archive",
    "archive/legacy_unified_models/optimize_all_15.py", "no", "no")
add("unified_models/optimize_gnn_label.py", "hyperparameter_search", "N/A", "archive",
    "archive/legacy_unified_models/optimize_gnn_label.py", "no", "no")
add("unified_models/optimize_gnn_label_fast.py", "hyperparameter_search", "N/A", "archive",
    "archive/legacy_unified_models/optimize_gnn_label_fast.py", "no", "no")

# --- Cross-task ML ---
add("unified_models/ml_cross_task.py", "deprecated_v1_leakage", "Fig.3", "archive",
    "archive/legacy_unified_models/ml_cross_task.py", "no", "no")
add("unified_models/ml_cross_task_v2.py", "experiment_launcher", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/cross_task_ml.py", "yes", "no")

# --- Fig.2: Chemical space / dataset analysis ---
add("unified_models/chemical_space_analysis.py", "analysis", "Fig.2", "migrate_to_experiments",
    "experiments/fig2/chemical_space.py", "yes", "no")
add("unified_models/chemical_space_advanced.py", "analysis", "Fig.2", "migrate_to_experiments",
    "experiments/fig2/chemical_space_advanced.py", "yes", "no")
add("unified_models/improved_ring_distribution.py", "analysis", "Fig.2", "migrate_to_experiments",
    "experiments/fig2/ring_distribution.py", "yes", "no")
add("unified_models/substituent_coverage_analysis.py", "analysis", "Fig.2", "migrate_to_experiments",
    "experiments/fig2/substituent_coverage.py", "yes", "no")

# --- Visualization ---
add("unified_models/nice_density_scatter.py", "plotting", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/aggregation/density_scatter.py", "yes", "no")
add("unified_models/enhanced_plots.py", "plotting", "Fig.3", "migrate_to_experiments",
    "experiments/fig3/aggregation/enhanced_plots.py", "yes", "no")

# --- Results analysis ---
add("unified_models/analyze_results.py", "superseded_analysis", "N/A", "archive",
    "archive/legacy_unified_models/analyze_results.py", "no", "no")
add("unified_models/show_best_results.py", "superseded_analysis", "N/A", "archive",
    "archive/legacy_unified_models/show_best_results.py", "no", "no")


def main():
    print("=" * 70)
    print("Legacy → Publication Mapping")
    print("=" * 70)

    output_csv = OUTPUT_DIR / "legacy_to_publication_mapping.csv"
    fieldnames = [
        "legacy_path", "role", "manuscript_figure", "action",
        "new_path", "publication_required", "numerical_behavior_change",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)

    print(f"\n  Output: {output_csv}")
    print(f"  Total files mapped: {len(ROWS)}")

    # Summary
    from collections import Counter
    actions = Counter(r["action"] for r in ROWS)
    figs = Counter(r["manuscript_figure"] for r in ROWS)
    pub = Counter(r["publication_required"] for r in ROWS)
    num = Counter(r["numerical_behavior_change"] for r in ROWS)

    print(f"\n  --- Action Distribution ---")
    for a, c in sorted(actions.items()):
        print(f"    {a}: {c}")

    print(f"\n  --- Figure Distribution ---")
    for f, c in sorted(figs.items()):
        print(f"    {f}: {c}")

    print(f"\n  --- Publication Required ---")
    for p, c in sorted(pub.items()):
        print(f"    {p}: {c}")

    print(f"\n  --- Numerical Behavior Change ---")
    for n, c in sorted(num.items()):
        print(f"    {n}: {c}")

    # Show migrate_to_src files
    print(f"\n  --- Files migrating to src/aroma_dps/ ---")
    for r in ROWS:
        if r["action"] == "migrate_to_src":
            print(f"    {r['legacy_path']}  →  {r['new_path']}")

    # Show migrate_to_experiments files
    print(f"\n  --- Files migrating to experiments/ ---")
    for r in ROWS:
        if r["action"] == "migrate_to_experiments":
            print(f"    {r['legacy_path']}  →  {r['new_path']}")


if __name__ == "__main__":
    main()
