"""Fig.4 Scaffold OOD & Exposure Curve: split generation (NO training).

Generates and reports split statistics for three experiments:
  A. Internal scaffold OOD (UNIFIED Murcko scaffold 80/20, 5-fold Group CV)
  B. Internal scaffold exposure curve (fixed 20% OOD, 20/40/60/80/100% nested)
  C. lunci10 ring-family exposure curve (5 split seeds, 20% permanent OOD, nested fractions)

Key rule (user revision):
  - Same Murcko scaffold gets SAME assignment (development or OOD_test) across HOMA/NICS/MBCO.
  - Missing-label samples ignored per task, but don't change scaffold assignment.

Strict rules:
  - Same molecule's all target-ring records MUST stay in one split.
  - Test scaffolds NEVER enter training.
  - Exposure curve test set is FIXED across fractions.
  - Train20 ⊂ Train40 ⊂ ... ⊂ Train100 (nested).
  - NO training here; this script only generates splits + reports stats.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"
DATA1_END = CODE_END / "data1_end"
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
SPLIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/02_scaffold_splits"
SPLIT_OUT.mkdir(parents=True, exist_ok=True)

L10_CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"

INTERNAL_TASKS = [
    {"name": "HOMA", "csv": DATA1_END / "collet_homa_0716.csv", "smiles_col": "smiles", "target_col": "homa_value"},
    {"name": "NICS_1zz", "csv": DATA1_END / "collet_nics_0716.csv", "smiles_col": "smiles", "target_col": "NICS_value"},
    {"name": "MBCO", "csv": DATA1_END / "collet_mbco_0716.csv", "smiles_col": "smiles", "target_col": "mbco_value"},
]

MODEL_SEEDS = [42, 123, 456, 789, 2024]
L10_SPLIT_SEEDS = [42, 123, 456, 789, 2024]
INTERNAL_SPLIT_SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.2
EXPOSURE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


# --------------------------- RDKit helpers ---------------------------

def murcko_scaffold(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        if scaff is None or scaff.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaff)
    except Exception:
        return ""


def canonical_smiles(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


# --------------------------- Unified internal scaffold split ---------------------------

def build_internal_splits() -> Dict[str, Any]:
    """Experiment A/B: UNIFIED 80/20 Murcko scaffold split across all 3 tasks.

    Same scaffold gets same assignment (development or OOD_test) across HOMA/NICS/MBCO.
    Missing-label samples are ignored per task but don't change scaffold assignment.
    """
    # Step 1: load all 3 tasks, collect all molecules
    task_dfs: Dict[str, pd.DataFrame] = {}
    all_smiles: Set[str] = set()
    for task in INTERNAL_TASKS:
        df = pd.read_csv(task["csv"], encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.dropna(subset=[task["smiles_col"]]).reset_index(drop=True)
        df["canon_smiles"] = df[task["smiles_col"]].apply(canonical_smiles)
        task_dfs[task["name"]] = df
        all_smiles.update(df["canon_smiles"].unique())

    # Step 2: compute Murcko scaffold for ALL unique molecules (union of 3 tasks)
    smi_list = sorted(all_smiles)
    smi_to_scaffold: Dict[str, str] = {smi: murcko_scaffold(smi) for smi in smi_list}

    # Step 3: unique scaffolds across all tasks
    unique_scaffolds = sorted(set(smi_to_scaffold.values()))

    # Step 4: unified 80/20 split on scaffolds (seed=42)
    rng = np.random.RandomState(INTERNAL_SPLIT_SEED)
    scaffold_order = list(unique_scaffolds)
    rng.shuffle(scaffold_order)
    n_test = int(TEST_FRAC * len(scaffold_order))
    test_scaffolds = set(scaffold_order[:n_test])
    dev_scaffolds = set(scaffold_order[n_test:])

    # Step 5: global scaffold split map
    global_map = pd.DataFrame([
        {"scaffold": s, "split": "OOD_test" if s in test_scaffolds else "development"}
        for s in unique_scaffolds
    ])
    global_map_path = SPLIT_OUT / "global_scaffold_split_map.csv"
    global_map.to_csv(global_map_path, index=False)

    # scaffold -> molecules mapping (global)
    scaffold_to_mols: Dict[str, Set[str]] = {}
    for smi, scaff in smi_to_scaffold.items():
        scaffold_to_mols.setdefault(scaff, set()).add(smi)

    # Step 6: 5-fold Group CV on development scaffolds (same for all tasks)
    dev_scaffolds_list = sorted(dev_scaffolds)
    rng2 = np.random.RandomState(INTERNAL_SPLIT_SEED)
    rng2.shuffle(dev_scaffolds_list)
    cv_folds = np.array_split(dev_scaffolds_list, N_FOLDS)

    # Save global dev scaffold order (for Experiment B nested exposure consistency)
    dev_order_df = pd.DataFrame({"order": range(len(dev_scaffolds_list)),
                                 "scaffold": dev_scaffolds_list})
    dev_order_df.to_csv(SPLIT_OUT / "global_dev_scaffold_order.csv", index=False)

    scaffold_to_fold: Dict[str, int] = {}
    for fi, fold_scaffs in enumerate(cv_folds):
        for s in fold_scaffs:
            scaffold_to_fold[s] = fi

    # molecule -> fold (via scaffold)
    mol_to_fold: Dict[str, int] = {
        smi: scaffold_to_fold[scaff]
        for smi, scaff in smi_to_scaffold.items()
        if scaff in scaffold_to_fold
    }

    # Step 7: exposure fractions (nested, same ordering for all tasks)
    dev_order = list(dev_scaffolds_list)  # already shuffled

    report: Dict[str, Any] = {"tasks": {}, "global_map_path": str(global_map_path)}

    for task in INTERNAL_TASKS:
        name = task["name"]
        df = task_dfs[name].dropna(subset=[task["target_col"]]).reset_index(drop=True)
        df["scaffold"] = df["canon_smiles"].map(smi_to_scaffold)

        # apply global assignment
        df["split"] = df["scaffold"].apply(
            lambda s: "OOD_test" if s in test_scaffolds else "development"
        )
        df["cv_fold"] = df["canon_smiles"].map(mol_to_fold).fillna(-1).astype(int)

        df_test = df[df["split"] == "OOD_test"].reset_index(drop=True)
        df_dev = df[df["split"] == "development"].reset_index(drop=True)

        test_mols = set(df_test["canon_smiles"])
        dev_mols = set(df_dev["canon_smiles"])

        # exposure fractions
        exposure_splits: Dict[float, Dict[str, Any]] = {}
        for frac in EXPOSURE_FRACTIONS:
            n_sel = int(len(dev_order) * frac)
            exposed_scaffolds = set(dev_order[:n_sel])
            exposed_mols_global = set()
            for s in exposed_scaffolds:
                exposed_mols_global.update(scaffold_to_mols[s])
            # only molecules present in THIS task
            exposed_mols_task = exposed_mols_global & dev_mols
            exposure_splits[frac] = {
                "n_scaffolds": len(exposed_scaffolds),
                "n_molecules": len(exposed_mols_task),
                "n_records": int(df[df["canon_smiles"].isin(exposed_mols_task)].shape[0]),
            }

        # leakage check
        mol_leakage = test_mols & dev_mols
        scaffold_leakage = test_scaffolds & dev_scaffolds

        # save per-task split
        split_df = df[["canon_smiles", "scaffold", task["smiles_col"], "split", "cv_fold"]].drop_duplicates("canon_smiles").copy()
        split_path = SPLIT_OUT / f"internal_{name}_scaffold_split.csv"
        split_df.to_csv(split_path, index=False)

        # exposure curve manifest
        exp_rows = []
        for frac in EXPOSURE_FRACTIONS:
            exp_rows.append({
                "task": name,
                "exposure_fraction": frac,
                "n_train_scaffolds": exposure_splits[frac]["n_scaffolds"],
                "n_train_molecules": exposure_splits[frac]["n_molecules"],
                "n_train_records": exposure_splits[frac]["n_records"],
                "n_test_scaffolds": len(test_scaffolds),
                "n_test_molecules": len(test_mols),
                "n_test_records": len(df_test),
            })
        exp_df = pd.DataFrame(exp_rows)
        exp_path = SPLIT_OUT / f"internal_{name}_exposure_curve_split.csv"
        exp_df.to_csv(exp_path, index=False)

        report["tasks"][name] = {
            "total_records": len(df),
            "total_molecules": df["canon_smiles"].nunique(),
            "total_scaffolds": df["scaffold"].nunique(),
            "test_scaffolds": len(test_scaffolds),
            "test_molecules": len(test_mols),
            "test_records": len(df_test),
            "dev_scaffolds": len(dev_scaffolds),
            "dev_molecules": len(dev_mols),
            "dev_records": len(df_dev),
            "cv_fold_sizes": [int(len(f)) for f in cv_folds],
            "molecule_leakage": len(mol_leakage),
            "scaffold_leakage": len(scaffold_leakage),
            "exposure_splits": {f: exposure_splits[f] for f in EXPOSURE_FRACTIONS},
            "split_path": str(split_path),
            "exposure_path": str(exp_path),
        }

    # global stats
    report["global"] = {
        "total_unique_molecules": len(smi_list),
        "total_unique_scaffolds": len(unique_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "dev_scaffolds": len(dev_scaffolds),
        "global_map_path": str(global_map_path),
    }
    return report


# --------------------------- lunci10 ring-family split ---------------------------

def build_l10_splits() -> Dict[str, Any]:
    """Experiment C: lunci10 ring-family split with 5 split seeds.

    ring_name is the scaffold-family unit.
    ~20% ring families = permanent OOD test (fixed per split seed).
    Remaining 80% = exposure pool, nested 20/40/60/80/100%.
    Molecule-level: if any record of a molecule is in test, all its records go to test.
    """
    df = pd.read_csv(L10_CLEAN_MANIFEST)
    mol_to_rings: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        mol = row["canonical_smiles"]
        rn = row["ring_name"]
        mol_to_rings.setdefault(mol, set()).add(rn)

    ring_to_mols: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        ring_to_mols.setdefault(row["ring_name"], set()).add(row["canonical_smiles"])

    ring_names = sorted(ring_to_mols.keys())
    n_rings = len(ring_names)

    report: Dict[str, Any] = {
        "total_records": len(df),
        "total_molecules": len(mol_to_rings),
        "total_ring_families": n_rings,
        "ring_family_sizes": {r: len(ring_to_mols[r]) for r in ring_names},
        "splits": {},
    }

    all_split_manifest_rows: List[Dict[str, Any]] = []

    for split_seed in L10_SPLIT_SEEDS:
        rng = np.random.RandomState(split_seed)
        ring_order = list(ring_names)
        rng.shuffle(ring_order)

        n_test_fam = int(TEST_FRAC * n_rings)
        test_families = set(ring_order[:n_test_fam])
        pool_families = set(ring_order[n_test_fam:])

        test_mols = set()
        for fam in test_families:
            test_mols.update(ring_to_mols[fam])
        for mol, rings in mol_to_rings.items():
            if rings & test_families:
                test_mols.add(mol)
        pool_mols = set(mol_to_rings.keys()) - test_mols

        pool_order = list(ring_order[n_test_fam:])
        exposure: Dict[float, Dict[str, Any]] = {}
        for frac in EXPOSURE_FRACTIONS:
            n_sel = int(len(pool_order) * frac)
            exposed_families = set(pool_order[:n_sel])
            exposed_mols = set()
            for fam in exposed_families:
                exposed_mols.update(ring_to_mols[fam])
            exposed_mols = exposed_mols - test_mols
            exposure[frac] = {
                "n_exposed_families": len(exposed_families),
                "n_exposed_molecules": len(exposed_mols),
                "n_exposed_records": int(df[df["canonical_smiles"].isin(exposed_mols)].shape[0]),
                "exposed_families": sorted(exposed_families),
            }

        mol_leakage = test_mols & pool_mols
        test_df = df[df["canonical_smiles"].isin(test_mols)]

        for frac in EXPOSURE_FRACTIONS + [0.0]:
            if frac == 0.0:
                all_split_manifest_rows.append({
                    "split_seed": split_seed,
                    "exposure_fraction": 0.0,
                    "n_exposed_families": 0,
                    "n_exposed_molecules": 0,
                    "n_exposed_records": 0,
                    "n_test_families": len(test_families),
                    "n_test_molecules": len(test_mols),
                    "n_test_records": len(test_df),
                    "phase": "ZERO_SHOT",
                })
            else:
                all_split_manifest_rows.append({
                    "split_seed": split_seed,
                    "exposure_fraction": frac,
                    "n_exposed_families": exposure[frac]["n_exposed_families"],
                    "n_exposed_molecules": exposure[frac]["n_exposed_molecules"],
                    "n_exposed_records": exposure[frac]["n_exposed_records"],
                    "n_test_families": len(test_families),
                    "n_test_molecules": len(test_mols),
                    "n_test_records": len(test_df),
                    "phase": "L10_ADAPTED",
                })

        report["splits"][split_seed] = {
            "n_test_families": len(test_families),
            "test_families": sorted(test_families),
            "n_pool_families": len(pool_families),
            "n_test_molecules": len(test_mols),
            "n_pool_molecules": len(pool_mols),
            "n_test_records": len(test_df),
            "molecule_leakage": len(mol_leakage),
            "exposure": {f: {k: v for k, v in exposure[f].items() if k != "exposed_families"} for f in EXPOSURE_FRACTIONS},
        }

    exp_manifest = pd.DataFrame(all_split_manifest_rows)
    exp_path = SPLIT_OUT / "l10_exposure_split_manifest.csv"
    exp_manifest.to_csv(exp_path, index=False)
    report["manifest_path"] = str(exp_path)
    return report


# --------------------------- Model count (revised) ---------------------------

def count_models() -> Dict[str, Any]:
    """Estimate total training runs (revised per user).

    Experiment A: 3 tasks x 2 models x 5 seeds x 6 (5-fold CV + final E*) = 180
    Experiment B: 3 tasks x 1 model x 5 seeds x 5 fractions = 75 (reuses E* from A)
    Experiment C: 3 tasks x 5 split_seeds x 5 model_seeds x 5 non-zero fractions = 375
    Zero-shot (0%): 3 x 5 x 5 = 75 predictions (no training)
    """
    return {
        "experiment_A_internal_ood": {
            "tasks": 3, "models": 2, "seeds": 5, "runs_per_config": 6,
            "total_trainings": 3 * 2 * 5 * 6,
        },
        "experiment_B_internal_curve": {
            "tasks": 3, "models": 1, "seeds": 5, "fractions": 5,
            "total_trainings": 3 * 1 * 5 * 5,
            "note": "reuses E* from Experiment A, no new CV",
        },
        "experiment_C_l10_curve": {
            "tasks": 3, "split_seeds": 5, "model_seeds": 5, "non_zero_fractions": 5,
            "total_fine_tunes": 3 * 5 * 5 * 5,
            "zero_shot_evaluations": 3 * 5 * 5,
        },
        "grand_total_trainings": 180 + 75 + 375,
        "grand_total_zero_shot_evaluations": 75,
    }


# --------------------------- Main ---------------------------

def main() -> None:
    print("=" * 70)
    print("Fig.4 Scaffold OOD & Exposure Curve — SPLIT GENERATION (no training)")
    print("=" * 70)

    # ---- Internal scaffold split (UNIFIED) ----
    print("\n[1] Internal UNIFIED scaffold OOD split (Experiment A/B)...")
    internal_report = build_internal_splits()
    g = internal_report["global"]
    print(f"\n  GLOBAL: {g['total_unique_molecules']} unique molecules, {g['total_unique_scaffolds']} unique scaffolds")
    print(f"  OOD test: {g['test_scaffolds']} scaffolds | Development: {g['dev_scaffolds']} scaffolds")
    print(f"  global_map: {g['global_map_path']}")
    for name, info in internal_report["tasks"].items():
        print(f"\n  --- {name} ---")
        print(f"  total: {info['total_records']} records, {info['total_molecules']} molecules, {info['total_scaffolds']} scaffolds")
        print(f"  OOD test: {info['test_scaffolds']} scaffolds, {info['test_molecules']} molecules, {info['test_records']} records")
        print(f"  development: {info['dev_scaffolds']} scaffolds, {info['dev_molecules']} molecules, {info['dev_records']} records")
        print(f"  5-fold Group CV sizes (scaffolds): {info['cv_fold_sizes']}")
        print(f"  molecule leakage: {info['molecule_leakage']}, scaffold leakage: {info['scaffold_leakage']}")
        print(f"  Exposure fractions (Experiment B):")
        for f in EXPOSURE_FRACTIONS:
            e = info["exposure_splits"][f]
            print(f"    {int(f*100)}%: {e['n_scaffolds']} scaffolds, {e['n_molecules']} mols, {e['n_records']} records")

    # ---- lunci10 ring-family split ----
    print("\n[2] lunci10 ring-family exposure split (Experiment C)...")
    l10_report = build_l10_splits()
    print(f"\n  lunci10 clean: {l10_report['total_records']} records, {l10_report['total_molecules']} molecules, {l10_report['total_ring_families']} ring families")
    for split_seed, info in l10_report["splits"].items():
        print(f"\n  --- split_seed={split_seed} ---")
        print(f"  permanent OOD test: {info['n_test_families']} families, {info['n_test_molecules']} molecules, {info['n_test_records']} records")
        print(f"  test families: {info['test_families']}")
        print(f"  exposure pool: {info['n_pool_families']} families, {info['n_pool_molecules']} molecules")
        print(f"  molecule leakage: {info['molecule_leakage']}")
        print(f"  Exposure fractions:")
        for f in EXPOSURE_FRACTIONS:
            e = info["exposure"][f]
            print(f"    {int(f*100)}%: {e['n_exposed_families']} families, {e['n_exposed_molecules']} mols, {e['n_exposed_records']} records")

    # ---- Model count ----
    print("\n[3] Total model run estimate (revised)...")
    counts = count_models()
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # ---- Save full report ----
    full_report = {
        "internal": internal_report,
        "lunci10": l10_report,
        "model_counts": counts,
        "config": {
            "internal_split_seed": INTERNAL_SPLIT_SEED,
            "l10_split_seeds": L10_SPLIT_SEEDS,
            "model_seeds": MODEL_SEEDS,
            "n_folds": N_FOLDS,
            "test_fraction": TEST_FRAC,
            "exposure_fractions": EXPOSURE_FRACTIONS,
            "unified_scaffold_split": True,
        },
    }
    report_path = SPLIT_OUT / "scaffold_split_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n[done] full report saved to {report_path}")


if __name__ == "__main__":
    main()
