#!/usr/bin/env python3
"""
P0-3: Paper-Result Dependency Manifest

明确 Fig.3/Fig.4 每个正式 panel 来源于哪个脚本、result directory、protocol 和代码版本。
任何 old/sample-level split/v1 external 结果不得进入 main-text。
"""
import csv
import json
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/aroma-dps/p0_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Git commit of the aroma-dps repo
REPO_ROOT = Path("/home/ubuntu/aroma-dps")
try:
    GIT_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()[:12]
    GIT_DIRTY = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip())
except Exception:
    GIT_COMMIT = "unknown"
    GIT_DIRTY = None

# --- Manifest rows ---
# Each row: fig, panel, description, source_script, result_dir, protocol,
#           split_seed, model_seeds, publication, notes

MANIFEST = [
    # ==================== Fig.3: Representation ====================
    {
        "fig": "Fig.3",
        "panel": "3a/3b",
        "description": "Fingerprint vs vanilla GNN vs RC-GNN comparison (HOMA/NICS/MCBO)",
        "source_script": "0831-end-code/stage1_representation_comparison/code/run_stage1_v2.py",
        "result_dir": "0831-end-code/results_v2/stage1/",
        "protocol": "PROTOCOL_SPEC Stage1 (P0-1 v2 + P0-2)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard + explicit_aromaticity_ablated",
        "publication": True,
        "notes": "5-seed CV + fixed test. Group-aware split via get_final_splits(SPLIT_SEED=2026).",
    },
    {
        "fig": "Fig.3",
        "panel": "3c/3d",
        "description": "2x2 factorial ablation (Base / Membership / LearnableReadout / Joint)",
        "source_script": "0831-end-code/stage2_ring_conditioning/code/ring_conditioning_ablation.py",
        "result_dir": "0831-end-code/results_v2_backup_pre_bugfix/stage2/",
        "protocol": "PROTOCOL_SPEC Stage2 (P0-1 v2 + P0-2)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard + explicit_aromaticity_ablated",
        "publication": True,
        "notes": "Factorial design preserved: ring_flag=10 for Membership/Joint. Do NOT unify with final model.",
    },
    {
        "fig": "Fig.3",
        "panel": "3e",
        "description": "Explicit aromaticity ablation",
        "source_script": "0831-end-code/stage1_representation_comparison/code/run_stage1_v2.py + stage2",
        "result_dir": "0831-end-code/results_v2/stage1/ (explicit_aromaticity_ablated mode)",
        "protocol": "PROTOCOL_SPEC Stage1/2 (P0-1 v2 + P0-2)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "explicit_aromaticity_ablated",
        "publication": True,
        "notes": "Comparison: standard features (with aromaticity) vs ablated (without).",
    },
    {
        "fig": "Fig.3",
        "panel": "3f",
        "description": "Mask pretraining comparison (direct / random_mask / ring_mask)",
        "source_script": "0831-end-code/stage3_mask_pretraining/code/run_mask_pretrain_v2.py",
        "result_dir": "0831-end-code/results_v2_backup_pre_bugfix/stage3/",
        "protocol": "PROTOCOL_SPEC Stage3 (P0-1 v2 + P0-2)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "v2 protocol. OLD results in results/stage3_mask_pretraining/ are NON-publication (canonical_splits).",
    },
    {
        "fig": "Fig.3",
        "panel": "3g",
        "description": "Cross-architecture validation (GNN/GIN/GAT/MPNN/GraphSAGE/DMPNN)",
        "source_script": "0831-end-code/stage4_cross_architecture/code/run_stage4_v2.py",
        "result_dir": "0831-end-code/results_v2_backup_pre_bugfix/stage4/ + results_v2/stage4/",
        "protocol": "PROTOCOL_SPEC Stage4 (P0-1 v2 + P0-2)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard + explicit_aromaticity_ablated",
        "publication": True,
        "notes": "v2 protocol. OLD cross_arch_eval.py uses canonical_splits — NON-publication.",
    },
    {
        "fig": "Fig.3",
        "panel": "3h",
        "description": "Ring flag sensitivity (0/1/5/10) -> binary membership decision",
        "source_script": "0831-end-code/stage5_ring_flag_sensitivity/code/run_ring_flag_sensitivity.py",
        "result_dir": "0831-end-code/results_v2/ring_flag_sensitivity/ (in progress)",
        "protocol": "PROTOCOL_SPEC ring_flag_sensitivity (P1-1 expanded)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard + explicit_aromaticity_ablated",
        "publication": True,
        "notes": "Currently running (seed 11=42, explicit_aromaticity_ablated). Determines binary membership + learnable projection.",
    },
    {
        "fig": "Fig.3",
        "panel": "3i",
        "description": "Final membership model (binary 0/1 + nn.Embedding(2, hidden_dim))",
        "source_script": "0831-end-code/stage6_final_membership/code/run_final_membership.py",
        "result_dir": "0831-end-code/results_v2_backup_pre_bugfix/final_membership/",
        "protocol": "PROTOCOL_SPEC final_membership (P1-1 follow-up)",
        "split_seed": 2026,
        "model_seeds": "[42]",
        "feature_modes": "standard + explicit_aromaticity_ablated",
        "publication": True,
        "notes": "HOMA uses membership_proj (binary + learnable projection). NICS/MCBO use membership_1 (binary).",
    },
    {
        "fig": "Fig.3",
        "panel": "SI",
        "description": "Significance testing (paired t-test / Wilcoxon)",
        "source_script": "0831-end-code/stats/significance_test.py",
        "result_dir": "N/A (consumes per_seed_results.csv from stages above)",
        "protocol": "Depends on input",
        "split_seed": 2026,
        "model_seeds": "N/A",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Must only consume v2 protocol results.",
    },

    # ==================== Fig.4: Generalization ====================
    {
        "fig": "Fig.4",
        "panel": "4a",
        "description": "Random split baseline (same as Fig.3 test set)",
        "source_script": "0831-end-code/stage4/run_stage4_v2.py (test set metrics)",
        "result_dir": "0831-end-code/results_v2_backup_pre_bugfix/stage4/",
        "protocol": "get_final_splits(SPLIT_SEED=2026)",
        "split_seed": 2026,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Random split reference for comparison with OOD.",
    },
    {
        "fig": "Fig.4",
        "panel": "4b",
        "description": "Scaffold OOD (Experiment A: E* protocol)",
        "source_script": "0901-end-code/fig4_lunci10/training/train_scaffold_ood.py",
        "result_dir": "0901-end-code/fig4_lunci10/training/ (scaffold_ood outputs)",
        "protocol": "global_scaffold_split_map.csv (80/20 Murcko scaffold)",
        "split_seed": 42,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Three tasks share same global_scaffold_split_map. E*=median(5 best epochs).",
    },
    {
        "fig": "Fig.4",
        "panel": "4c",
        "description": "Ring-family OOD (Experiment C: zero-shot + adaptation)",
        "source_script": "0901-end-code/fig4_lunci10/training/train_l10_exposure.py",
        "result_dir": "0901-end-code/fig4_lunci10/training/ (l10_exposure outputs)",
        "protocol": "Ring-family split (5 split seeds)",
        "split_seed": "[42, 123, 456, 789, 2024]",
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "0% = frozen zero-shot. >0% = fine-tune from frozen checkpoint. 5x5x5=375 fine-tunes + 75 zero-shot.",
    },
    {
        "fig": "Fig.4",
        "panel": "4d",
        "description": "Exposure/adaptation curve (Experiment B: 20/40/60/80/100%)",
        "source_script": "0901-end-code/fig4_lunci10/training/train_exposure_curve.py",
        "result_dir": "0901-end-code/fig4_lunci10/training/ (exposure_curve outputs)",
        "protocol": "global_dev_scaffold_order.csv (nested Train20 ⊂ Train40 ⊂ ... ⊂ Train100)",
        "split_seed": 42,
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Reuses E* from Experiment A. No new epoch selection.",
    },
    {
        "fig": "Fig.4",
        "panel": "4e",
        "description": "External absolute prediction (clean manifest)",
        "source_script": "0901-end-code/fig4_lunci10/evaluation/run_external_absolute_v2.py",
        "result_dir": "0901-end-code/fig4_lunci10/evaluation/",
        "protocol": "lunci10_clean_manifest.csv (frozen checkpoints, no training)",
        "split_seed": "N/A (inference only)",
        "model_seeds": "[42, 123, 456, 789, 2024] (RC_MPNN/GAT), [42] (Base_GNN)",
        "feature_modes": "standard",
        "publication": True,
        "notes": "v2 uses clean manifest. v1 (run_external_absolute.py) uses full manifest — NON-publication.",
    },
    {
        "fig": "Fig.4",
        "panel": "4f",
        "description": "External delta prediction (pairwise aromaticity loss)",
        "source_script": "0901-end-code/fig4_lunci10/evaluation/run_external_delta.py",
        "result_dir": "0901-end-code/fig4_lunci10/evaluation/",
        "protocol": "Frozen checkpoints (no training), clean manifest",
        "split_seed": "N/A (inference only)",
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": True,
        "notes": "DeltaA = A_R - A_P. HOMA: ΔHOMA=HOMA_R-HOMA_P. MCBO: ΔMCBO=MCBO_R-MCBO_P. NICS: ΔNICS*=NICS_P-NICS_R.",
    },
    {
        "fig": "Fig.4",
        "panel": "4g",
        "description": "Hammett analysis (σm/σp vs ΔA)",
        "source_script": "0901-end-code/fig4_lunci10/analysis/hammett_analysis.py",
        "result_dir": "0901-end-code/fig4_lunci10/analysis/",
        "protocol": "Consumes frozen predictions from v2",
        "split_seed": "N/A",
        "model_seeds": "N/A",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Within-context analysis. ortho separated. No oracle correction.",
    },
    {
        "fig": "Fig.4",
        "panel": "4h",
        "description": "Novelty analysis (A/B/C/D categories)",
        "source_script": "0901-end-code/fig4_lunci10/analysis/novelty_analysis.py",
        "result_dir": "0901-end-code/fig4_lunci10/analysis/",
        "protocol": "Consumes frozen predictions from v2",
        "split_seed": "N/A",
        "model_seeds": "N/A",
        "feature_modes": "standard",
        "publication": True,
        "notes": "Category A (seen scaffold+substituent) through D (both unseen).",
    },

    # ==================== Excluded from main text ====================
    {
        "fig": "EXCLUDED",
        "panel": "-",
        "description": "OLD stage3 mask pretraining (canonical_splits, seed=model_seed)",
        "source_script": "0831-end-code/stage3_mask_pretraining/code/run_pretrain_eval.py",
        "result_dir": "0831-end-code/results/stage3_mask_pretraining/",
        "protocol": "canonical_splits(seed=model_seed) — OLD",
        "split_seed": "model_seed (variable)",
        "model_seeds": "[42]",
        "feature_modes": "standard",
        "publication": False,
        "notes": "NON-PUBLICATION: split changes with model seed. Superseded by run_mask_pretrain_v2.py.",
    },
    {
        "fig": "EXCLUDED",
        "panel": "-",
        "description": "OLD stage4 cross-arch (canonical_splits)",
        "source_script": "0831-end-code/stage4_cross_architecture/code/cross_arch_eval.py",
        "result_dir": "N/A (superseded by v2)",
        "protocol": "canonical_splits(seed=model_seed) — OLD",
        "split_seed": "model_seed (variable)",
        "model_seeds": "[42]",
        "feature_modes": "standard",
        "publication": False,
        "notes": "NON-PUBLICATION: split changes with model seed. Superseded by run_stage4_v2.py.",
    },
    {
        "fig": "EXCLUDED",
        "panel": "-",
        "description": "multiseed (canonical_splits with model seed)",
        "source_script": "0831-end-code/multiseed/run_multiseed.py",
        "result_dir": "N/A (exploratory)",
        "protocol": "canonical_splits(seed=model_seed) — OLD",
        "split_seed": "model_seed (variable)",
        "model_seeds": "[42, 123, 456, 789, 2024]",
        "feature_modes": "standard",
        "publication": False,
        "notes": "NON-PUBLICATION: different test sets per seed. Exploratory only.",
    },
    {
        "fig": "EXCLUDED",
        "panel": "-",
        "description": "v1 external absolute (full manifest, includes exact_seen)",
        "source_script": "0901-end-code/fig4_lunci10/evaluation/run_external_absolute.py",
        "result_dir": "N/A (superseded by v2)",
        "protocol": "lunci10_manifest.csv (full, includes exact_seen) — DEPRECATED",
        "split_seed": "N/A (inference only)",
        "model_seeds": "[42]",
        "feature_modes": "standard",
        "publication": False,
        "notes": "NON-PUBLICATION: includes exact_seen molecules. Superseded by run_external_absolute_v2.py.",
    },
    {
        "fig": "EXCLUDED",
        "panel": "-",
        "description": "ml_cross_task v1 (PCA data leakage)",
        "source_script": "unified_models/ml_cross_task.py",
        "result_dir": "N/A (superseded by v2)",
        "protocol": "train_test_split + PCA on full data — DATA LEAKAGE",
        "split_seed": 42,
        "model_seeds": "[42]",
        "feature_modes": "standard",
        "publication": False,
        "notes": "NON-PUBLICATION: PCA fit on full data before split. Superseded by ml_cross_task_v2.py.",
    },
]


def main():
    print("=" * 70)
    print("P0-3: Paper-Result Dependency Manifest")
    print("=" * 70)
    print(f"  Git commit: {GIT_COMMIT}")
    print(f"  Git dirty: {GIT_DIRTY}")

    output_csv = OUTPUT_DIR / "paper_result_dependency_manifest.csv"
    fieldnames = [
        "fig", "panel", "description", "source_script", "result_dir",
        "protocol", "split_seed", "model_seeds", "feature_modes",
        "publication", "notes", "git_commit", "git_dirty",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in MANIFEST:
            row["git_commit"] = GIT_COMMIT
            row["git_dirty"] = GIT_DIRTY
            writer.writerow(row)

    print(f"\n  Output: {output_csv}")
    print(f"  Total entries: {len(MANIFEST)}")

    pub = [r for r in MANIFEST if r["publication"]]
    excluded = [r for r in MANIFEST if not r["publication"]]
    print(f"  Publication panels: {len(pub)}")
    print(f"  Excluded (non-publication): {len(excluded)}")

    print(f"\n  --- Publication Panels ---")
    for r in pub:
        print(f"    {r['fig']} {r['panel']}: {r['description'][:70]}")
        print(f"      script: {r['source_script']}")
        print(f"      split_seed: {r['split_seed']}")

    print(f"\n  --- Excluded (DO NOT use in main text) ---")
    for r in excluded:
        print(f"    {r['source_script']}")
        print(f"      reason: {r['notes'][:80]}")

    # Save as JSON too
    json_path = OUTPUT_DIR / "paper_result_dependency_manifest.json"
    with open(json_path, "w") as f:
        json.dump({
            "git_commit": GIT_COMMIT,
            "git_dirty": GIT_DIRTY,
            "manifest": MANIFEST,
        }, f, indent=2)
    print(f"\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
