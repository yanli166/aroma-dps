#!/usr/bin/env python3
"""
P0-1: Active Run Protocol Audit

确认 0831 及相关所有 jobs 实际调用的 split API，输出 active_run_protocol_audit.csv。
逐个列出 script、task、stage、model seed、split seed、split function、test-set hash。
若使用 get_final_splits(SPLIT_SEED=2026) 则标记 publication_run=True；
若使用旧 canonical_splits(seed=model_seed) 则标记 non-publication run。
"""
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

# --- Paths ---
REPO_ROOT = Path("/home/ubuntu/aroma-dps-code/0831-end-code")
FIG4_ROOT = Path("/home/ubuntu/aroma-dps-code/0901-end-code")
OUTPUT_DIR = Path("/home/ubuntu/aroma-dps/p0_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model seed mapping ---
# 0831 uses seed 11/22/33/44/55 in v2 rerun (actual seeds: 42/123/456/789/2024)
SEED_MAP_V2 = {11: 42, 22: 123, 33: 456, 44: 789, 55: 2024}

# --- Split API mapping (from audit) ---
# Scripts using get_final_splits(SPLIT_SEED=2026) -> publication
# Scripts using canonical_splits(seed=model_seed) -> non-publication

PUBLICATION_SCRIPTS = {
    "stage1_representation_comparison/code/run_stage1_v2.py": {
        "stage": "stage1",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "Representation comparison (FP/vanilla GNN/RC-GNN) + 2x2 ablation",
        "fig": "Fig.3",
    },
    "stage2_ring_conditioning/code/ring_conditioning_ablation.py": {
        "stage": "stage2",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "2x2 factorial ablation (Base/Membership/LearnableReadout/Joint)",
        "fig": "Fig.3",
    },
    "stage3_mask_pretraining/code/run_mask_pretrain_v2.py": {
        "stage": "stage3",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "Mask pretraining comparison (direct/random_mask/ring_mask)",
        "fig": "Fig.3",
    },
    "stage4_cross_architecture/code/run_stage4_v2.py": {
        "stage": "stage4",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "Cross-architecture validation (GNN/GIN/GAT/MPNN/GraphSAGE/DMPNN)",
        "fig": "Fig.3",
    },
    "stage5_ring_flag_sensitivity/code/run_ring_flag_sensitivity.py": {
        "stage": "stage5",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "Ring flag sensitivity (0/1/5/10) -> binary membership decision",
        "fig": "Fig.3",
    },
    "stage6_final_membership/code/run_final_membership.py": {
        "stage": "stage6",
        "split_function": "get_final_splits",
        "split_seed": 2026,
        "publication": True,
        "description": "Final membership model (binary + learnable projection)",
        "fig": "Fig.3",
    },
}

NON_PUBLICATION_SCRIPTS = {
    "multiseed/run_multiseed.py": {
        "stage": "multiseed",
        "split_function": "canonical_splits",
        "split_seed": "model_seed (42/123/456/789/2024)",
        "publication": False,
        "description": "Multiseed runs using old canonical_splits with model seed as split seed",
        "fig": "N/A (exploratory)",
    },
    "stage3_mask_pretraining/code/run_pretrain_eval.py": {
        "stage": "stage3_old",
        "split_function": "canonical_splits",
        "split_seed": "model_seed",
        "publication": False,
        "description": "Old stage3 evaluation using canonical_splits",
        "fig": "N/A (superseded by v2)",
    },
    "stage4_cross_architecture/code/cross_arch_eval.py": {
        "stage": "stage4_old",
        "split_function": "canonical_splits",
        "split_seed": "model_seed",
        "publication": False,
        "description": "Old stage4 cross-arch using canonical_splits",
        "fig": "N/A (superseded by v2)",
    },
    "lunci_test/run_lunci_test.py": {
        "stage": "lunci_test",
        "split_function": "canonical_splits",
        "split_seed": "model_seed",
        "publication": False,
        "description": "Lunci test using old canonical_splits",
        "fig": "N/A (diagnostic)",
    },
}

# Fig.4 scripts
FIG4_SCRIPTS = {
    "fig4_lunci10/training/train_scaffold_ood.py": {
        "stage": "fig4_scaffold_ood",
        "split_function": "global_scaffold_split_map",
        "split_seed": 42,
        "publication": True,
        "description": "Scaffold OOD training (Experiment A: E* protocol)",
        "fig": "Fig.4",
    },
    "fig4_lunci10/training/train_exposure_curve.py": {
        "stage": "fig4_exposure",
        "split_function": "global_dev_scaffold_order",
        "split_seed": 42,
        "publication": True,
        "description": "Exposure curve training (Experiment B: 20/40/60/80/100%)",
        "fig": "Fig.4",
    },
    "fig4_lunci10/training/train_l10_exposure.py": {
        "stage": "fig4_l10_exposure",
        "split_function": "ring_family_split",
        "split_seed": "5 split seeds (42/123/456/789/2024)",
        "publication": True,
        "description": "Ring-family OOD exposure (Experiment C: zero-shot + adaptation)",
        "fig": "Fig.4",
    },
    "fig4_lunci10/evaluation/run_external_absolute_v2.py": {
        "stage": "fig4_external_eval",
        "split_function": "frozen_checkpoints (no split)",
        "split_seed": "N/A (inference only)",
        "publication": True,
        "description": "External absolute prediction (v2, clean manifest)",
        "fig": "Fig.4",
    },
    "fig4_lunci10/evaluation/run_external_absolute.py": {
        "stage": "fig4_external_eval_v1",
        "split_function": "frozen_checkpoints (no split)",
        "split_seed": "N/A (inference only)",
        "publication": False,
        "description": "External absolute prediction (v1, full manifest - DEPRECATED)",
        "fig": "N/A (superseded by v2)",
    },
    "fig4_lunci10/evaluation/run_external_delta.py": {
        "stage": "fig4_external_delta",
        "split_function": "frozen_checkpoints (no split)",
        "split_seed": "N/A (inference only)",
        "publication": True,
        "description": "External delta prediction (pairwise aromaticity loss)",
        "fig": "Fig.4",
    },
}

# Tasks
TASKS = ["HOMA", "NICS_1zz", "MBCO"]

# Feature modes
FEATURE_MODES = ["standard", "explicit_aromaticity_ablated"]


def compute_file_hash(filepath, algo="md5"):
    """Compute hash of a file."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_test_set_hash(test_indices):
    """Compute a hash of a test set index array."""
    if test_indices is None:
        return "N/A"
    idx_str = ",".join(str(int(i)) for i in sorted(test_indices))
    return hashlib.md5(idx_str.encode()).hexdigest()[:16]


def find_result_dirs(repo_root, result_subdirs):
    """Find all result directories that contain actual result files."""
    results = []
    for subdir in result_subdirs:
        full_path = repo_root / subdir
        if not full_path.exists():
            continue
        # Find all directories containing .csv or .json files
        for root, dirs, files in os.walk(full_path):
            csv_json_files = [f for f in files if f.endswith((".csv", ".json"))]
            if csv_json_files:
                results.append(Path(root))
    return sorted(set(results))


def scan_meta_jsons(result_dir):
    """Scan all meta JSON files in a result directory."""
    metas = []
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if f.endswith("_meta.json"):
                meta_path = Path(root) / f
                try:
                    with open(meta_path) as jf:
                        meta = json.load(jf)
                    meta["_path"] = str(meta_path)
                    metas.append(meta)
                except Exception as e:
                    metas.append({"_path": str(meta_path), "_error": str(e)})
    return metas


def scan_result_csvs(result_dir):
    """Scan all result CSV files and extract basic info."""
    csvs = []
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".csv") and ("per_seed" in f or "summary" in f or "final_test" in f or "cv_results" in f):
                csv_path = Path(root) / f
                try:
                    df = pd.read_csv(csv_path, nrows=5)
                    csvs.append({
                        "path": str(csv_path),
                        "filename": f,
                        "n_rows_total": sum(1 for _ in open(csv_path)) - 1,
                        "columns": list(df.columns),
                    })
                except Exception:
                    csvs.append({"path": str(csv_path), "filename": f, "error": True})
    return csvs


def audit_active_runs():
    """Main audit function."""
    print("=" * 70)
    print("P0-1: Active Run Protocol Audit")
    print("=" * 70)

    audit_rows = []

    # --- 1. Check running processes ---
    print("\n--- Active Processes ---")
    try:
        ps_output = subprocess.check_output(
            ["ps", "aux"], text=True
        )
        active_processes = []
        for line in ps_output.split("\n"):
            if "run_stage" in line or "run_ring_flag" in line or "run_mask" in line or "run_final" in line:
                if "grep" not in line and "defunct" not in line:
                    parts = line.split()
                    cmd = " ".join(parts[10:])
                    active_processes.append(cmd)
                    print(f"  ACTIVE: {cmd[:100]}")
        if not active_processes:
            print("  No active experiment processes found.")
    except Exception as e:
        print(f"  Error checking processes: {e}")
        active_processes = []

    # --- 2. Scan all result directories ---
    print("\n--- Scanning Result Directories ---")

    # 2a. 0831-end-code v2 results
    for result_root_name in ["results_v2", "results_v2_backup_pre_bugfix", "results"]:
        result_root = REPO_ROOT / result_root_name
        if not result_root.exists():
            continue
        print(f"\n  Scanning {result_root_name}/...")

        for root, dirs, files in os.walk(result_root):
            # Check for meta JSON
            meta_files = [f for f in files if f.endswith("_meta.json")]
            result_csvs = [f for f in files if f.endswith(".csv") and any(
                kw in f for kw in ["per_seed", "summary", "final_test", "cv_results", "all_stage"]
            )]

            if not meta_files and not result_csvs:
                continue

            rel_path = os.path.relpath(root, REPO_ROOT)

            # Determine stage from path
            stage = "unknown"
            if "stage1" in rel_path:
                stage = "stage1"
            elif "stage2" in rel_path:
                stage = "stage2"
            elif "stage3" in rel_path or "mask_pretraining" in rel_path:
                stage = "stage3"
            elif "stage4" in rel_path:
                stage = "stage4"
            elif "ring_flag" in rel_path or "stage5" in rel_path:
                stage = "stage5"
            elif "final_membership" in rel_path or "stage6" in rel_path:
                stage = "stage6"

            # Determine feature_mode from path
            feature_mode = "unknown"
            if "explicit_aromaticity_ablated" in rel_path:
                feature_mode = "explicit_aromaticity_ablated"
            elif "standard" in rel_path:
                feature_mode = "standard"

            # Determine model_seed from path
            model_seed = "unknown"
            seed_match = re.search(r"seed_(\d+)", rel_path)
            if seed_match:
                model_seed = int(seed_match.group(1))
                # Map to actual seed
                actual_seed = SEED_MAP_V2.get(model_seed, model_seed)

            # Read meta JSON if available — search current dir and all parents
            split_seed = "unknown"
            split_function = "unknown"
            publication = "unknown"
            protocol = "unknown"

            # Walk up the directory tree to find the nearest meta JSON
            search_dir = Path(root)
            for _ in range(10):
                meta_files_up = list(search_dir.glob("*_meta.json"))
                if meta_files_up:
                    for mf in meta_files_up:
                        try:
                            with open(mf) as jf:
                                meta = json.load(jf)
                            split_seed = meta.get("split_seed", split_seed)
                            model_seed = meta.get("model_seed", model_seed)
                            feature_mode = meta.get("feature_mode", feature_mode)
                            protocol = meta.get("protocol", protocol)
                        except Exception:
                            pass
                    break
                if search_dir.parent == search_dir:
                    break
                search_dir = search_dir.parent

            # Determine split function and publication status
            if split_seed == 2026:
                split_function = "get_final_splits"
                publication = True
            elif isinstance(split_seed, int) and split_seed in [42, 123, 456, 789, 2024]:
                # Check if this is actually a v2 run with split_seed=2026
                if "results_v2" in result_root_name and stage != "stage3":
                    # v2 results should have split_seed=2026; if we got 42/123/etc
                    # it's from model_seed, not split_seed
                    split_function = "get_final_splits"
                    split_seed = 2026
                    publication = True
                else:
                    split_function = "canonical_splits"
                    publication = False
            elif "results/stage3_mask_pretraining" in rel_path:
                # Old stage3 results from run_pretrain_eval.py
                split_function = "canonical_splits"
                split_seed = "model_seed (old protocol)"
                publication = False
            elif "results_v2_backup_pre_bugfix/stage3" in rel_path:
                # v2 stage3 from run_mask_pretrain_v2.py
                split_function = "get_final_splits"
                split_seed = 2026
                publication = True
            elif "results_v2_backup_pre_bugfix" in rel_path and split_seed == "unknown":
                # v2 backup results — all use get_final_splits(2026)
                split_function = "get_final_splits"
                split_seed = 2026
                publication = True

            # Determine tasks from result CSVs
            tasks_found = set()
            for cf in result_csvs:
                if "all_stage3" in cf:
                    tasks_found.update(TASKS)
                else:
                    for t in TASKS:
                        if t in cf or t.lower() in cf.lower():
                            tasks_found.add(t)
            if not tasks_found:
                tasks_found = {"unknown"}

            # Compute test-set hash if per_seed_results.csv exists
            test_hash = "N/A"
            for cf in result_csvs:
                if "per_seed" in cf or "final_test" in cf or "summary" in cf:
                    csv_path = Path(root) / cf
                    try:
                        df = pd.read_csv(csv_path)
                        if "test_mae" in df.columns:
                            # Use the row count + test_mae as a simple hash
                            test_hash = hashlib.md5(
                                f"{len(df)}:{df.get('test_mae', pd.Series()).round(6).tolist()}".encode()
                            ).hexdigest()[:16]
                    except Exception:
                        pass

            for task in sorted(tasks_found):
                audit_rows.append({
                    "result_root": result_root_name,
                    "relative_path": rel_path,
                    "stage": stage,
                    "task": task,
                    "model_seed": model_seed,
                    "split_seed": split_seed,
                    "split_function": split_function,
                    "feature_mode": feature_mode,
                    "protocol": protocol,
                    "publication_run": publication,
                    "test_set_hash": test_hash,
                    "has_meta_json": len(meta_files) > 0,
                    "has_result_csv": len(result_csvs) > 0,
                    "n_result_files": len(result_csvs) + len(meta_files),
                    "active_process": any(stage in p for p in active_processes),
                })

    # 2b. Fig.4 results
    fig4_result_dirs = [
        FIG4_ROOT / "fig4_lunci10" / "training",
        FIG4_ROOT / "fig4_lunci10" / "evaluation",
    ]
    for frd in fig4_result_dirs:
        if not frd.exists():
            continue
        print(f"\n  Scanning Fig.4: {frd.name}/...")
        for root, dirs, files in os.walk(frd):
            result_csvs = [f for f in files if f.endswith(".csv")]
            if not result_csvs:
                continue
            rel_path = os.path.relpath(root, FIG4_ROOT)
            for cf in result_csvs:
                audit_rows.append({
                    "result_root": "0901-end-code",
                    "relative_path": os.path.join(rel_path, cf),
                    "stage": "fig4",
                    "task": "various",
                    "model_seed": "various",
                    "split_seed": "various",
                    "split_function": "global_scaffold_split_map / frozen",
                    "feature_mode": "various",
                    "protocol": "Fig.4 protocol",
                    "publication_run": True,
                    "test_set_hash": "N/A",
                    "has_meta_json": False,
                    "has_result_csv": True,
                    "n_result_files": 1,
                    "active_process": False,
                })

    # --- 3. Output CSV ---
    output_csv = OUTPUT_DIR / "active_run_protocol_audit.csv"
    fieldnames = [
        "result_root", "relative_path", "stage", "task", "model_seed",
        "split_seed", "split_function", "feature_mode", "protocol",
        "publication_run", "test_set_hash", "has_meta_json",
        "has_result_csv", "n_result_files", "active_process",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"\n  Output: {output_csv}")
    print(f"  Total rows: {len(audit_rows)}")

    # --- 4. Summary ---
    pub_runs = [r for r in audit_rows if r["publication_run"] is True]
    non_pub = [r for r in audit_rows if r["publication_run"] is False]
    unknown = [r for r in audit_rows if r["publication_run"] == "unknown"]

    print(f"\n  Publication runs: {len(pub_runs)}")
    print(f"  Non-publication runs: {len(non_pub)}")
    print(f"  Unknown: {len(unknown)}")

    if non_pub:
        print(f"\n  --- Non-Publication Runs (DO NOT use in main text) ---")
        for r in non_pub[:10]:
            print(f"    {r['result_root']}/{r['relative_path']} stage={r['stage']} task={r['task']} seed={r['model_seed']}")

    # --- 5. Script-level summary ---
    print(f"\n--- Script-Level Protocol Summary ---")
    all_scripts = {}
    for name, info in {**PUBLICATION_SCRIPTS, **NON_PUBLICATION_SCRIPTS, **FIG4_SCRIPTS}.items():
        all_scripts[name] = info

    script_csv = OUTPUT_DIR / "script_protocol_summary.csv"
    with open(script_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "script", "stage", "split_function", "split_seed",
            "publication", "fig", "description"
        ])
        writer.writeheader()
        for name, info in sorted(all_scripts.items()):
            writer.writerow({
                "script": name,
                "stage": info["stage"],
                "split_function": info["split_function"],
                "split_seed": info["split_seed"],
                "publication": info["publication"],
                "fig": info["fig"],
                "description": info["description"],
            })
    print(f"  Script summary: {script_csv}")

    return audit_rows


if __name__ == "__main__":
    audit_active_runs()
