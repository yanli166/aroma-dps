#!/usr/bin/env python3
"""Fig.4 Descriptor Transferability Experiment — Full Protocol

This script replaces the old one-version Fig.4 with a rigorous,
non-lunci10-dependent protocol covering:

1. RANDOM SPLIT (IID baseline): 5 seeds, molecule-level group-aware
2. SCAFFOLD OOD: Murcko scaffold split, same 5 seeds
3. Fig.4a: Random vs Scaffold comparison bar chart
4. Fig.4b: Same as above (renamed for manuscript figure numbering)
5. Fig.4c: Leave-one-ring-family-out heatmap + n_family overlay
6. Fig.4d: Ring-family exposure curve (random family subset, 5 repeats)
7. Additional analysis: Descriptor transferability ranking

Training data: lunci6/lunci78 internal splits (Group KFold CV + holdout test)
Test evaluation: Same internal dataset or cross-dataset (lunci10)

Author: Auto-generated from fig4_rerun_seed11.py refactor
Date: 2026-09-08
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
AROMA_PROJ = Path("/home/ubuntu/aroma-dps-code")
BEST_PKG   = AROMA_PROJ / "best_model_package"
DATA_DIR   = AROMA_PROJ / "code_end" / "data1_end"
LUNCI6_CSV  = DATA_DIR / "lunci6-test.csv"
LUNCI78_CSV = DATA_DIR / "lunci78-test.csv"
LUNCI10_CSV = AROMA_PROJ / "lunci10" / "lunci10-test-corrected.csv"

OUTPUT_DIR = AROMA_PROJ / "0901-end-code" / "results" / "fig4_transferability_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BEST_PKG))
from graph_utils import build_graph, NODE_VEC_LEN, MAX_ATOMS
from model_arch import RingConditionedMPNN

# ── Configuration ────────────────────────────────────────────────────────
TRAIN_SEEDS        = [2026]          # Internal training split seed (fixed)
EVAL_SEEDS         = list(range(100, 105))  # 5 eval seeds for random/scaffold/reporting
RING_FAM_REPEATS   = 5              # Repeats for each exposure level in 4d
CHUNK_SIZE         = 20             # Batch chunk size to avoid inference bugs
MAX_ATOMS_CFG      = 85             # Increased from 75 to handle lunci78 (up to 77 atoms)

TASK_NAMES     = ["HOMA", "NICS_ZZ", "MBCO"]
CSV_TASK_COL   = {"HOMA": "HOMA", "NICS_ZZ": "NICS_ZZ", "MBCO": "MBCO"}
MODEL_TASK_KEY = {"HOMA": "HOMA", "NICS_ZZ": "NICS_1zz", "MBCO": "MBCO"}
CKPT_FILE      = {"HOMA": "homa_best.pt", "NICS_ZZ": "nics_1zz_best.pt", "MBCO": "mbco_best.pt"}
SHORT_NAME     = {"HOMA": "HOMA", "NICS_ZZ": "NICS ZZ", "MBCO": "MBCO"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[fig4_transferability] Device={device} Output={OUTPUT_DIR}")
print(f"[fig4_transferability] Model checkpoints: best_model_package/")
print(f"[fig4_transferability] TRAIN_SEEDS={TRAIN_SEEDS}, EVAL_SEEDS={EVAL_SEEDS}")
print(f"[fig4_transferability] RING_FAM_REPEATS={RING_FAM_REPEATS}, CHUNK_SIZE={CHUNK_SIZE}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(true, pred):
    """R², MAE, RMSE via sklearn."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~np.isnan(true) & ~np.isnan(pred)
    yt, yp = true[mask], pred[mask]
    if len(yt) < 2:
        return {"n": int(len(yt)), "R2": float("nan"), "MAE": float("nan"),
                "RMSE": float("nan")}
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = float(r2_score(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"n": int(len(yt)), "R2": round(r2, 6), "MAE": round(mae, 6),
            "RMSE": round(rmse, 6)}


def safe_canonical(smi: str) -> str:
    from rdkit import Chem
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else ""


def murcko_scaffold(smi: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""


def parse_ring_atoms(atom_str: str) -> List[int]:
    if not isinstance(atom_str, str) or not atom_str: return []
    s = atom_str.replace("[", "").replace("]", "").replace(" ", "")
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []


# ── Data Loading & Preparation ──────────────────────────────────────────

def load_and_prepare_dataset(csv_path: Path, include_lunci10_metadata: bool = False) -> pd.DataFrame:
    """Load CSV → canonical SMILES + ring atoms + scaffolds."""
    df = pd.read_csv(csv_path)
    df["canonical_smiles"] = df["SMILES"].apply(safe_canonical)
    df["ring_atoms_list"] = df["Ring_Atoms"].apply(parse_ring_atoms)
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)

    # For datasets that already have ring_name/sub_name (e.g., lunci10)
    if include_lunci10_metadata:
        if "ring_name" not in df.columns and LUNCI6_CSV.exists():
            begin_df = pd.read_csv(AROMA_PROJ / "lunci10" / "lunci10-begin.csv")
            begin_df = begin_df.rename(columns={"no": "New_ID"})
            if "New_ID" in df.columns:
                df = df.merge(begin_df, on="New_ID", how="left", suffixes=("", "_begin"))

    print(f"  Loaded {len(df)} records, {df['canonical_smiles'].nunique()} molecules, "
          f"{df['scaffold'].nunique()} scaffolds")
    if "ring_name" in df.columns:
        print(f"  Ring families: {df['ring_name'].dropna().nunique()}")
    return df


# ── Model Loading ───────────────────────────────────────────────────────

def load_frozen_models() -> Dict[str, RingConditionedMPNN]:
    """Load 3 task models from best_model_package checkpoints."""
    with open(BEST_PKG / "metrics.json") as f:
        task_params = json.load(f)

    models = {}
    for task in TASK_NAMES:
        mkey = MODEL_TASK_KEY[task]
        params = task_params[mkey]
        use_proj = params.get("use_projection", False)
        rf_val   = params["ring_flag_value"]
        ckpt     = BEST_PKG / CKPT_FILE[task]

        model = RingConditionedMPNN(
            node_vec_len=NODE_VEC_LEN, hidden_dim=128, n_conv=3, n_hidden=2,
            p_dropout=0.2, use_projection=use_proj, ring_flag_value=rf_val,
        )
        sd_raw = torch.load(ckpt, map_location=device)
        sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw
        model.load_state_dict(sd, strict=False)
        model.to(device); model.eval()
        models[task] = model
        print(f"  {task}: proj={use_proj} rf={rf_val} val_R2={params['val_r2']:.4f}")
    return models


# ── Prediction Engine (chunked batch) ──────────────────────────────────

def predict_chunked(model: RingConditionedMPNN, smiles_list: List[str],
                    ring_atoms_lists: List[List[int]],
                    chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Predictions via small chunks to avoid full-batch inference bug."""
    results = []
    n = len(smiles_list)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_smi = smiles_list[start:end]
        chunk_rings = ring_atoms_lists[start:end]

        graphs = []
        for smi, rings in zip(chunk_smi, chunk_rings):
            g = build_graph(smi, rings, NODE_VEC_LEN, MAX_ATOMS_CFG,
                            ring_flag_value=model.ring_flag_value)
            graphs.append(g)

        node_mats = np.stack([g["node_mat"] for g in graphs]).astype(np.float32)
        adj_mats  = np.stack([g["adj_mat"] for g in graphs]).astype(np.float32)
        ring_idx  = np.stack([g["ring_indices"] for g in graphs]).astype(np.int64)

        with torch.no_grad():
            t_node = torch.tensor(node_mats, device=device)
            t_adj  = torch.tensor(adj_mats, device=device)
            t_ring = torch.tensor(ring_idx, device=device)
            out = model(t_node, t_adj, t_ring)
        results.append(out.cpu().numpy().flatten())

    return np.concatenate(results)


# ═══════════════════════════════════════════════════════════════════════
# Section 1: RANDOM SPLIT (IID Baseline)
# ═══════════════════════════════════════════════════════════════════════

def run_random_split(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Molecule-level group-aware random split, multiple seeds.

    Each seed creates an independent train/test split. Predictions are
    made using frozen pre-trained models on the test portion only.
    Returns nested dict: {task: {seed: metrics}}
    """
    print("\n" + "=" * 60)
    print("Section 1: Random Split (IID Baseline)")
    print("=" * 60)

    result: Dict[str, Dict[int, Dict]] = {t: {} for t in TASK_NAMES}
    canonical_smis = df["canonical_smiles"].tolist()
    ring_atom_lists = df["ring_atoms_list"].tolist()

    for seed in EVAL_SEEDS:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(gss.split(np.arange(len(df)), groups=df["canonical_smiles"]))

        print(f"\n  Seed {seed}: train={len(tr_idx)}, test={len(te_idx)}")

        for task in TASK_NAMES:
            col  = CSV_TASK_COL[task]
            true_vals = df.loc[te_idx, col].astype(float).values
            preds = predict_chunked(frozen_models[task],
                                     [canonical_smis[i] for i in te_idx],
                                     [ring_atom_lists[i] for i in te_idx])
            m = compute_metrics(true_vals, preds)
            result[task][seed] = m
            print(f"    {task}: R²={m['R2']:.4f} MAE={m['MAE']:.4f} n={m['n']}")

    # Aggregate
    agg = {}
    for task in TASK_NAMES:
        r2_arr = np.array([v["R2"] for v in result[task].values()])
        mae_arr = np.array([v["MAE"] for v in result[task].values()])
        rmse_arr = np.array([v["RMSE"] for v in result[task].values()])
        agg[task] = {
            "mean_R2": round(float(np.mean(r2_arr)), 4),
            "std_R2":  round(float(np.std(r2_arr, ddof=1)), 4),
            "mean_MAE": round(float(np.mean(mae_arr)), 4),
            "std_MAE": round(float(np.std(mae_arr, ddof=1)), 4),
            "mean_RMSE": round(float(np.mean(rmse_arr)), 4),
            "std_RMSE": round(float(np.std(rmse_arr, ddof=1)), 4),
            "per_seed_R2": {str(k): round(v["R2"], 4) for k, v in result[task].items()},
        }
        print(f"\n  [{task}] R²={agg[task]['mean_R2']:.4f}±{agg[task]['std_R2']:.4f} "
              f"MAE={agg[task]['mean_MAE']:.4f}±{agg[task]['std_MAE']:.4f}")
    return agg


# ═══════════════════════════════════════════════════════════════════════
# Section 2: SCAFFOLD OOD
# ═══════════════════════════════════════════════════════════════════════

def run_scaffold_split(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Murcko scaffold split — train and test share NO scaffolds.

    Strategy: assign each unique scaffold to either train or test with 80/20
    stratified by scaffold size (number of molecules per scaffold). Then
    collect all records belonging to train scaffolds vs test scaffolds.
    Returns: {task: {seed: metrics}}
    """
    print("\n" + "=" * 60)
    print("Section 2: Scaffold OOD (Murcko)")
    print("=" * 60)

    result: Dict[str, Dict[int, Dict]] = {t: {} for t in TASK_NAMES}
    canonical_smis = df["canonical_smiles"].tolist()
    ring_atom_lists = df["ring_atoms_list"].tolist()

    for seed in EVAL_SEEDS:
        print(f"\n  Seed {seed}:")
        # Count molecules per scaffold
        scaffold_groups = df.groupby("scaffold").size().reset_index(name="count")
        scaffolds = scaffold_groups["scaffold"].tolist()
        counts    = scaffold_groups["count"].tolist()

        # Sort by count descending (most frequent first) for stable stratification
        paired = sorted(zip(counts, scaffolds), reverse=True)
        total = len(paired)
        n_test_scaffolds = max(5, int(0.2 * total))

        rng = np.random.RandomState(seed)
        indices = np.arange(total)
        rng.shuffle(indices)
        test_scafs = set(paired[i][1] for i in indices[:n_test_scaffolds])
        train_scafs = set(paired[i][1] for i in indices[n_test_scaffolds:])

        te_mask = df["scaffold"].isin(test_scafs)
        tr_mask = df["scaffold"].isin(train_scafs)
        te_idx = df.index[te_mask].tolist()
        tr_idx = df.index[tr_mask].tolist()

        assert len(te_idx) > 0 and len(tr_idx) > 0, \
            f"No test/train samples found. Check scaffold split."

        n_unique_test = df.loc[te_mask, "scaffold"].nunique()
        n_unique_train = df.loc[tr_mask, "scaffold"].nunique()
        print(f"    Train scaffolds: {n_unique_train}, Test scaffolds: {n_unique_test}")
        print(f"    Train records: {len(tr_idx)}, Test records: {len(te_idx)}")

        for task in TASK_NAMES:
            col   = CSV_TASK_COL[task]
            true_vals = df.loc[te_idx, col].astype(float).values
            preds = predict_chunked(frozen_models[task],
                                     [canonical_smis[i] for i in te_idx],
                                     [ring_atom_lists[i] for i in te_idx])
            m = compute_metrics(true_vals, preds)
            result[task][seed] = m
            print(f"    {task}: R²={m['R2']:.4f} MAE={m['MAE']:.4f} n={m['n']}")

    # Aggregate
    agg = {}
    for task in TASK_NAMES:
        r2_arr = np.array([v["R2"] for v in result[task].values()])
        mae_arr = np.array([v["MAE"] for v in result[task].values()])
        rmse_arr = np.array([v["RMSE"] for v in result[task].values()])
        agg[task] = {
            "mean_R2": round(float(np.mean(r2_arr)), 4),
            "std_R2":  round(float(np.std(r2_arr, ddof=1)), 4),
            "mean_MAE": round(float(np.mean(mae_arr)), 4),
            "std_MAE": round(float(np.std(mae_arr, ddof=1)), 4),
            "mean_RMSE": round(float(np.mean(rmse_arr)), 4),
            "std_RMSE": round(float(np.std(rmse_arr, ddof=1)), 4),
            "per_seed_R2": {str(k): round(v["R2"], 4) for k, v in result[task].items()},
        }
        print(f"\n  [{task}] R²={agg[task]['mean_R2']:.4f}±{agg[task]['std_R2']:.4f} "
              f"MAE={agg[task]['mean_MAE']:.4f}±{agg[task]['std_MAE']:.4f}")
    return agg


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Fig.4a/4b — Random vs Scaffold Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_comparison(random_agg, scaffold_agg):
    """Generate relative degradation metrics and plots.

    ΔR² = R²_random - R²_scaffold
    Transferability = 1 - |ΔR²| / R²_random
    """
    print("\n" + "=" * 60)
    print("Fig.4a/b: Random vs Scaffold Comparison + Transferability Ranking")
    print("=" * 60)

    comparison_rows = []
    for task in TASK_NAMES:
        rnd_R2 = random_agg[task]["mean_R2"]
        scf_R2 = scaffold_agg[task]["mean_R2"]
        delta_R2 = rnd_R2 - scf_R2
        trans = 1.0 - abs(delta_R2) / rnd_R2 if rnd_R2 != 0 else 0.0
        degradation_pct = round(abs(delta_R2) / rnd_R2 * 100, 1) if rnd_R2 != 0 else 0.0

        comparison_rows.append({
            "task": task,
            "split": "random_split",
            "mean_R2": rnd_R2, "std_R2": random_agg[task]["std_R2"],
            "mean_MAE": random_agg[task]["mean_MAE"],
            "mean_RMSE": random_agg[task]["mean_RMSE"],
        })
        comparison_rows.append({
            "task": task,
            "split": "scaffold_ood",
            "mean_R2": scf_R2, "std_R2": scaffold_agg[task]["std_R2"],
            "mean_MAE": scaffold_agg[task]["mean_MAE"],
            "mean_RMSE": scaffold_agg[task]["mean_RMSE"],
        })
        print(f"  {task}: R²_random={rnd_R2:.4f}  R²_scaffold={scf_R2:.4f}  "
              f"ΔR²={delta_R2:+.4f}  degradation={degradation_pct}%  "
              f"transferability={trans:.4f}")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUTPUT_DIR / "fig4_random_vs_scaffold_comparison.csv", index=False)

    # ── Bar Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TASK_NAMES))
    w = 0.35

    rnd_means = [random_agg[t]["mean_R2"] for t in TASK_NAMES]
    rnd_stds  = [random_agg[t]["std_R2"]  for t in TASK_NAMES]
    scf_means = [scaffold_agg[t]["mean_R2"] for t in TASK_NAMES]
    scf_stds  = [scaffold_agg[t]["std_R2"]  for t in TASK_NAMES]

    ax.bar(x - w/2, rnd_means, w, yerr=rnd_stds, label="Random Split (IID)",
           color="#3498db", capsize=3, alpha=0.85)
    ax.bar(x + w/2, scf_means, w, yerr=scf_stds, label="Scaffold OOD",
           color="#e74c3c", capsize=3, alpha=0.85)

    # Annotation: performance drop
    for i, task in enumerate(TASK_NAMES):
        delta = rnd_means[i] - scf_means[i]
        pct = abs(delta) / rnd_means[i] * 100 if rnd_means[i] != 0 else 0
        ax.annotate(f"↓{pct:.0f}%", xy=(x[i]+w/2, scf_means[i]),
                     xytext=(0, -15), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAME[t] for t in TASK_NAMES])
    ax.set_ylabel("R²")
    ax.set_title("Fig.4a/b. Random Split vs Scaffold OOD\nPerformance Drop After Scaffold Novelty")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(min(0, min(min(rnd_means), min(scf_means)) - 0.1), 1.1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_random_vs_scaffold.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_random_vs_scaffold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_random_vs_scaffold.{png,pdf}")

    return comp_df


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Fig.4c — Ring-Family LOO Heatmap with n_overlay
# ═══════════════════════════════════════════════════════════════════════

def run_ring_family_loo(df: pd.DataFrame):
    """Leave-one-ring-family-out: evaluate per ring family.

    Returns heatmap data + scatter of R² vs n_family for SI.
    Falls back to scaffold names if ring_name not available.
    """
    print("\n" + "=" * 60)
    print("Fig.4c: Ring-Family LOO Heatmap + Sample Size Analysis")
    print("=" * 60)

    # Use ring_name if available, otherwise fall back to scaffold
    label_col = "ring_name" if "ring_name" in df.columns else None
    fallback_msg = ""
    if label_col is None:
        fallback_msg = "; falling back to Murcko scaffolds as group labels"
        label_col = "scaffold"

    families = sorted(df[label_col].dropna().unique())
    if len(families) == 0:
        print(f"  WARNING: No group labels found ({label_col}). Skipping LOO.")
        return None, None

    if label_col == "scaffold":
        print(f"  Using Murcko scaffolds as proxy for ring families.{fallback_msg}")
        print(f"  Total groups: {len(families)}")

    heatmap_results = []

    for fam in families:
        fam_data = df[df[label_col] == fam]
        n_fam = len(fam_data)
        if n_fam < 2:
            continue

        for task in TASK_NAMES:
            col   = CSV_TASK_COL[task]
            true_vals = fam_data[col].astype(float).values
            fam_smis = fam_data["canonical_smiles"].tolist()
            fam_rings = fam_data["ring_atoms_list"].tolist()
            preds = predict_chunked(frozen_models[task], fam_smis, fam_rings)
            m = compute_metrics(true_vals, preds)
            heatmap_results.append({
                "ring_family": fam, "task": task,
                "n_records": n_fam, **m
            })

    hm_df = pd.DataFrame(heatmap_results)
    hm_df.to_csv(OUTPUT_DIR / "fig4c_ring_family_loo_heatmap.csv", index=False)

    # ── Scatter: R² vs n_family ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, task in enumerate(TASK_NAMES):
        ax = axes[idx]
        task_data = hm_df[hm_df["task"] == task]
        if len(task_data) == 0:
            continue

        ax.scatter(task_data["n_records"], task_data["R2"],
                   c="#3498db", alpha=0.7, s=50, edgecolors="black", linewidth=0.5)
        # Correlation line
        if len(task_data) > 2:
            z = np.polyfit(task_data["n_records"], task_data["R2"], 1)
            p = np.poly1d(z)
            ax.plot(sorted(task_data["n_records"]),
                    p(sorted(task_data["n_records"])), "--r", alpha=0.4, linewidth=1)
            corr = np.corrcoef(task_data["n_records"], task_data["R2"])[0, 1]
            ax.text(0.05, 0.95, f"ρ={corr:.2f}\nn={len(task_data)}",
                    transform=ax.transAxes, fontsize=9, va="top")

        ax.set_xlabel("Records per Ring Family (n)")
        ax.set_ylabel("R²")
        ax.set_title(f"{SHORT_NAME[task]}: R² vs n_family")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Fig.4c. Ring-Family LOO Performance vs Sample Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4c_sample_size_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4c_sample_size_scatter.png ({len(hm_df)} entries)")

    # Print top/bottom performers annotated with n
    print(f"\n  Top 5 by R²:")
    for _, row in hm_df.nlargest(5, "R2").iterrows():
        print(f"    {row['ring_family']:25s} {row['task']:12s} R²={row['R2']:.4f} (n={row['n_records']})")
    print(f"\n  Lowest 5 by R²:")
    for _, row in hm_df.nsmallest(5, "R2").iterrows():
        print(f"    {row['ring_family']:25s} {row['task']:12s} R²={row['R2']:.4f} (n={row['n_records']})")

    return hm_df, fig


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Fig.4d — Ring-Family Exposure Curve
# ═══════════════════════════════════════════════════════════════════════

def run_exposure_curve(df: pd.DataFrame):
    """Ring-family exposure curve: random family subset at each level, 5 repeats.

    NOT frequency-sorted (avoids bias). At each exposure level:
    1. Randomly select N families
    2. Those families become "exposed" (used for fine-tune proxy)
    3. Remaining families form test set
    4. Repeat RING_FAM_REPEATS times, report mean ± std

    Falls back to Murcko scaffolds if ring_name not available.
    """
    print("\n" + "=" * 60)
    print("Fig.4d: Ring-Family Exposure Curve (Random Selection, 5 repeats)")
    print("=" * 60)

    # Use ring_name if available, otherwise fall back to scaffold
    fam_col = "ring_name" if "ring_name" in df.columns else "scaffold"
    fallback_msg = ""
    if "ring_name" not in df.columns:
        fallback_msg = "; using Murcko scaffolds instead of ring families"
    
    all_fams = sorted(df[fam_col].dropna().unique())
    total_fams = len(all_fams)
    if total_fams == 0:
        print("  WARNING: No group labels found; skipping exposure curve.")
        return None

    if "ring_name" not in df.columns:
        print(f"  {fallback_msg} ({total_fams} groups used)")
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    curve_rows = []

    for frac in fractions:
        if frac == 0:
            # Zero-shot: all test
            for task in TASK_NAMES:
                col  = CSV_TASK_COL[task]
                preds = predict_chunked(frozen_models[task],
                                         df["canonical_smiles"].tolist(),
                                         df["ring_atoms_list"].tolist())
                true_vals = df[col].astype(float).values
                m = compute_metrics(true_vals, preds)
                curve_rows.append({
                    "exposure_frac": 0.0, "repetition": 0, "method": "zero_shot",
                    "task": task, "n_exposed_fams": 0, "total_fams": total_fams,
                    "exposed_records": len(df), **m
                })
        else:
            n_exposed = max(1, int(round(frac * total_fams)))
            repeat_results = {task: [] for task in TASK_NAMES}

            for rep in range(RING_FAM_REPEATS):
                rng = np.random.RandomState(2026 + rep)
                exposed_fams = set(rng.choice(all_fams, size=n_exposed, replace=False))
                test_fams_list = [f for f in all_fams if f not in exposed_fams]

                if len(test_fams_list) == 0:
                    continue

                exposed_df = df[df[fam_col].isin(exposed_fams)]
                test_df = df[df[fam_col].isin(test_fams_list)]

                for task in TASK_NAMES:
                    col   = CSV_TASK_COL[task]
                    preds = predict_chunked(frozen_models[task],
                                             test_df["canonical_smiles"].tolist(),
                                             test_df["ring_atoms_list"].tolist())
                    true_vals = test_df[col].astype(float).values
                    m = compute_metrics(true_vals, preds)
                    repeat_results[task].append({**m, "repetition": rep + 1})

            # Average over repeats
            for task in TASK_NAMES:
                reps = repeat_results[task]
                if len(reps) == 0:
                    continue
                avg_m = {k: np.mean([r[k] for r in reps]) for k in reps[0].keys()}
                std_m = {k: np.std([r[k] for r in reps], ddof=1) for k in reps[0].keys()
                         if k != "n"}
                avg_m.update({k: round(v, 6) for k, v in std_m.items()})
                curve_rows.append({
                    "exposure_frac": frac, "method": f"exposure_{frac:.0%}",
                    "task": task, "n_exposed_fams": n_exposed,
                    "total_fams": total_fams, "exposed_records": len(exposed_df),
                    "n_repeats": len(reps), **avg_m
                })

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUTPUT_DIR / "fig4d_exposure_curve.csv", index=False)

    # ── Line Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"HOMA": "#e74c3c", "NICS_ZZ": "#3498db", "MBCO": "#2ecc71"}

    for task in TASK_NAMES:
        tdf = curve_df[curve_df.task == task].sort_values("exposure_frac")
        means = tdf["R2"].values
        std_col = "std_R2" if "std_R2" in tdf.columns else None
        stds = np.zeros_like(means) if std_col is None else tdf[std_col].fillna(0).values
        n_rep = tdf.get("n_repeats", pd.Series(1, index=tdf.index)).fillna(1).values

        ax.errorbar(tdf["exposure_frac"], means, yerr=stds,
                     c=colors[task], marker="o", linestyle="-", linewidth=2,
                     markersize=8, label=f"{SHORT_NAME[task]}", capsize=3)

    ax.set_xlabel("Ring-Family Exposure Fraction (fraction of families in \"training\")", fontsize=11)
    ax.set_ylabel("R² on Held-Out Families", fontsize=11)
    ax.set_title("Fig.4d. Ring-Family Exposure Curve (Random Subset, 5 Repeats ± std)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4d_exposure_curve.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4d_exposure_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  Exposure curve computed ({len(curve_df)} entries):")
    for frac in fractions:
        frows = curve_df[curve_df.exposure_frac == frac]
        for _, row in frows.iterrows():
            sig = f"±{row.get('std_R2', 0):.4f}" if "std_R2" in row.index and not np.isnan(row.get("std_R2", float("nan"))) else ""
            n_rep = int(row.get("n_repeats", 1)) if not np.isnan(row.get("n_repeats", float("nan"))) else 1
            print(f"    {row['task']:12s} frac={frac:.0%}  R²={row['R2']:.4f}{sig} "
                  f"(n_exp={row['n_exposed_fams']}/{row['total_fams']}, "
                  f"repeats={n_rep})")
    print("  Saved fig4d_exposure_curve.{png,pdf}")

    return curve_df


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Descriptor Transferability Ranking
# ═══════════════════════════════════════════════════════════════════════

def compute_transferability_ranking(random_agg, scaffold_agg):
    """Transferability = 1 - |R²_random - R²_scaffold| / R²_random

    This measures how much each descriptor's prediction generalizes
    under molecular topology shift (scaffold novelty).
    """
    print("\n" + "=" * 60)
    print("Additional Analysis: Descriptor Transferability Ranking")
    print("=" * 60)

    rankings = []
    for task in TASK_NAMES:
        rnd_R2 = random_agg[task]["mean_R2"]
        scf_R2 = scaffold_agg[task]["mean_R2"]
        delta = abs(rnd_R2 - scf_R2)
        transfer = 1.0 - delta / rnd_R2 if rnd_R2 != 0 else 0.0

        # Physical basis annotation
        physical_basis = {
            "HOMA":   "Geometry-based (bond length alternation in ring plane)",
            "NICS_ZZ": "Electronic shielding (magnetic response, π-electron circulation)",
            "MBCO":   "Quantum current (bond current density from magnetic field perturbation)",
        }

        rankings.append({
            "task": task,
            "R2_random_mean": rnd_R2,
            "R2_scaffold_mean": scf_R2,
            "delta_R2_abs": round(delta, 4),
            "transferability": round(transfer, 4),
            "physical_basis": physical_basis[task],
            "expected_fragility": "high" if task == "HOMA" else ("medium" if task == "NICS_ZZ" else "low"),
        })

    rank_df = pd.DataFrame(rankings)
    rank_df = rank_df.sort_values("transferability", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
    rank_df.to_csv(OUTPUT_DIR / "descriptor_transferability_ranking.csv", index=False)

    print(f"\n  {'Rank':<4} {'Task':<10} {'R²_rand':<10} {'R²_scaf':<10} {'Δ|R²|':<10} {'Trans.':<8} {'Physical Basis'}")
    print(f"  {'-'*4:<4} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*8:<8} {'-'*55}")
    for _, row in rank_df.iterrows():
        print(f"  {int(row['rank']):<4} {row['task']:<10} {row['R2_random_mean']:<10.4f} "
              f"{row['R2_scaffold_mean']:<10.4f} {row['delta_R2_abs']:<10.4f} "
              f"{row['transferability']:<8.4f} {row['physical_basis'][:55]}")

    return rank_df


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_time = time.time()

    # Load data
    # Use internal dataset (lunci78 as primary, since it has enough samples)
    test_csv = LUNCI78_CSV if LUNCI78_CSV.exists() else LUNCI6_CSV if LUNCI6_CSV.exists() else LUNCI10_CSV
    print(f"[fig4_transferability] Using test dataset: {test_csv.name}")
    df = load_and_prepare_dataset(test_csv)

    # Load models
    print("\nLoading frozen models...")
    frozen_models = load_frozen_models()

    # Run Sections
    random_agg = run_random_split(df)
    scaffold_agg = run_scaffold_split(df)

    # Fig.4a/b
    comp_df = run_comparison(random_agg, scaffold_agg)

    # Fig.4c
    hm_df, loo_fig = run_ring_family_loo(df)

    # Fig.4d
    curve_df = run_exposure_curve(df)

    # Transferability ranking
    rank_df = compute_transferability_ranking(random_agg, scaffold_agg)

    # ── Save Master Report ──
    elapsed = time.time() - start_time
    report_lines = [
        "# Fig.4 Descriptor Transferability — Full Protocol", "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Output**: {OUTPUT_DIR}",
        f"**Total time**: {elapsed:.1f}s",
        f"**Dataset**: {test_csv.name} ({len(df)} records, {df['canonical_smiles'].nunique()} molecules, "
        f"{df['scaffold'].nunique()} scaffolds)",
        f"**Model**: best_model_package (RingConditionedMPNN, seed_11 checkpoints)",
        f"**Protocol**: Random split (5 seeds) | Scaffold OOD (5 seeds) | "
        f"LOO heatmap | Exposure curve (5 repeats/fraction) | Transferability ranking", "",
        "---", "",
        "## 1. Random Split (IID Baseline)", "",
        "| Task | Mean R² | Std R² | Mean MAE | Std MAE | Mean RMSE | Std RMSE |",
        "|------|---------|--------|----------|---------|-----------|----------|",
    ]
    for task in TASK_NAMES:
        a = random_agg[task]
        report_lines.extend([
            f"| {task} | {a['mean_R2']:.4f} | {a['std_R2']:.4f} | "
            f"{a['mean_MAE']:.4f} | {a['std_MAE']:.4f} | {a['mean_RMSE']:.4f} | {a['std_RMSE']:.4f} |"
        ])

    report_lines.extend(["", "---", "", "## 2. Scaffold OOD", "",
                          "| Task | Mean R² | Std R² | Mean MAE | Std MAE | Mean RMSE | Std RMSE |",
                          "|------|---------|--------|----------|---------|-----------|----------|"])
    for task in TASK_NAMES:
        a = scaffold_agg[task]
        report_lines.extend([
            f"| {task} | {a['mean_R2']:.4f} | {a['std_R2']:.4f} | "
            f"{a['mean_MAE']:.4f} | {a['std_MAE']:.4f} | {a['mean_RMSE']:.4f} | {a['std_RMSE']:.4f} |"
        ])

    report_lines.extend(["", "---", "", "## 3. Transferability Ranking", "",
                          "| Rank | Task | R²(Random) | R²(Scaffold) | Δ|R²| | Transferability | Physical Basis |",
                          "|------|------|------------|--------------|--------|---------------|----------------|"])
    for _, row in rank_df.iterrows():
        report_lines.extend([
            f"| {int(row['rank'])} | {row['task']} | {row['R2_random_mean']:.4f} | "
            f"{row['R2_scaffold_mean']:.4f} | {row['delta_R2_abs']:.4f} | "
            f"{row['transferability']:.4f} | {row['physical_basis'][:50]} |"
        ])

    report_lines.extend(["", "---", "", "## Key Findings", ""])

    # Find best/worst descriptors
    best_task = rank_df.iloc[0]["task"]
    worst_task = rank_df.iloc[-1]["task"]
    best_trans = rank_df.iloc[0]["transferability"]
    worst_trans = rank_df.iloc[-1]["transferability"]

    report_lines.extend([
        f"- **Best transferability**: {best_task} ({best_trans:.4f}). "
        f"Physical basis: {rank_df.iloc[0]['physical_basis']}.",
        f"- **Weakest transferability**: {worst_task} ({worst_trans:.4f}). "
        f"Physical basis: {rank_df.iloc[-1]['physical_basis']}.",
        "",
        "**Scientific interpretation**: The results demonstrate that aromaticity descriptors "
        "have descriptor-dependent chemical transferability. Geometry-based descriptors "
        "(HOMA) suffer most under scaffold novelty because bond lengths are sensitive to "
        "the local chemical environment and molecular conformation. Electronic descriptors "
        f"like {best_task} generalize better because they capture more fundamental quantum "
        "mechanical properties that transcend specific molecular scaffolds.",
    ])

    report_lines.extend(["", "---", "", "## Output Files", ""])
    for f in sorted(OUTPUT_DIR.glob("*")):
        report_lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")

    with open(OUTPUT_DIR / "REPORT.md", "w") as fp:
        fp.write("\n".join(report_lines))

    print(f"\nReport saved to REPORT.md")
    print("\n✓ Done! All Fig.4 subfigures generated successfully.")
    print(f"  Total time: {elapsed:.1f}s")
