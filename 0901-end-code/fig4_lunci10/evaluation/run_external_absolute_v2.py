"""Phase 4 strict zero-shot external prediction on lunci10_clean_external.

STRICT PROTOCOL (per user):
  - NO training, NO fine-tuning, NO early-stop
  - NO fit scaler on lunci10 (only load pre-trained scalers)
  - NO test-time bias correction
  - NO seed/checkpoint/model selection based on lunci10 performance
  - Main external analysis uses ONLY lunci10_clean_manifest.csv
  - exact_seen subset is diagnostic only
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

# path bootstrap
PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"
ORIG_MODELS_ROOT = str(PROJ_ROOT / "unified_models")
for p in (str(PROJ_ROOT), str(CODE_END), ORIG_MODELS_ROOT):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from common.graph_data import load_adj_format  # noqa: E402
from unified_models.mpnn.model import MPNNModel  # noqa: E402
from unified_models.gat.model import GATModel  # noqa: E402
from unified_models.gnn.model import GNNModel  # noqa: E402
from unified_models.gin.model import GINModel  # noqa: E402

LAYER2_GNN_DIR = CODE_END / "results/layer2_gnn"
LAYER3_RING_DIR = CODE_END / "results/layer3_ring_fixed"

AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"
SEEN_MANIFEST = AUDIT_OUT / "lunci10_exact_seen_manifest.csv"

OUT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/01_external_absolute"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CLEAN_CSV = OUT_DIR / "lunci10_absolute_predictions.csv"
PRED_SEEN_CSV = OUT_DIR / "lunci10_exact_seen_diagnostic_predictions.csv"
SUMMARY_CSV = OUT_DIR / "fig4a_generalization_summary.csv"
RUN_LOG = OUT_DIR / "phase4_run.log"

NVL = 60
MAX_ATOMS = 75
RING_FLAG = 10

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
TRUTH_COL = {"HOMA": "HOMA", "NICS_1zz": "NICS_ZZ", "MBCO": "MBCO"}
SEEDS = [42, 123, 456, 789, 2024]

# Backbones to evaluate (per user protocol: at least Base_GNN + frozen final RC_GNN)
MODELS = {
    "Base_GNN": {
        "layer": "layer2_gnn",
        "cls": GNNModel,
        # Only seed 42 has layer2 files
        "available_seeds": [42],
    },
    "RC_MPNN": {
        "layer": "layer3_ring_fixed",
        "cls": MPNNModel,
        "available_seeds": SEEDS,
    },
    "RC_GAT": {
        "layer": "layer3_ring_fixed",
        "cls": GATModel,
        "available_seeds": SEEDS,
    },
}


def _find_ckpt(layer_dir: Path, task: str, backbone: str, seed: int, encoding: str = "label") -> Optional[Path]:
    # layer3 pattern
    for enc in [encoding, "combined", "mask", "pool", "none"]:
        p = layer_dir / f"seed_{seed}" / task / f"{backbone}_{enc}" / "best_model.pth"
        if p.is_file():
            return p
    # layer2 pattern
    p2 = layer_dir / f"seed_{seed}" / task / backbone / "best_model.pth"
    if p2.is_file():
        return p2
    return None


def _build_model(model_name: str) -> torch.nn.Module:
    """Build a model with same defaults as Fig.3 training (NVL=60, hidden=128, n_conv=3, n_hidden=2, dropout=0.2)."""
    if model_name == "MPNN":
        return MPNNModel(node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2, n_outputs=1, p_dropout=0.2, mode="label")
    if model_name == "GAT":
        # GATModel signature: node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs, p_dropout, n_heads, mode
        return GATModel(node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2, n_outputs=1, p_dropout=0.2,
                        n_heads=4, mode="label")
    if model_name == "GNN":
        return GNNModel(node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2, n_outputs=1, p_dropout=0.2, mode="label")
    if model_name == "GIN":
        return GINModel(node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2, n_outputs=1, p_dropout=0.2,
                        ring_flag_value=RING_FLAG, mode="label")
    raise ValueError(f"unknown model: {model_name}")


def predict_one_model(task: str, model_name: str, seed: int, data: Dict[str, Any], device: str = "cuda") -> np.ndarray:
    """Load frozen checkpoint, run inference, return predictions. STRICT no-training."""
    spec = MODELS[model_name]
    layer_dir = LAYER2_GNN_DIR if spec["layer"] == "layer2_gnn" else LAYER3_RING_DIR
    ckpt_path = _find_ckpt(layer_dir, task, model_name.replace("Base_", "").replace("RC_", ""), seed)
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint not found for {model_name} {task} seed={seed}")

    model = _build_model(model_name.replace("Base_", "").replace("RC_", ""))
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    node_mats = data["node_mats"].to(device)
    adj_mats = data["adj_mats"].to(device)

    preds = []
    with torch.no_grad():
        batch = 64
        for i in range(0, node_mats.size(0), batch):
            nm = node_mats[i:i + batch]
            am = adj_mats[i:i + batch]
            # MPNN/GNN/GIN/GAT label-mode forward signature: (node_mat, adj_mat, mask_mat=None, ring_indices=None)
            out = model(nm, am)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[finite]
    y_p = y_pred[finite]
    err = y_p - y_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if len(y_t) >= 2:
        try:
            pearson_r, _ = stats.pearsonr(y_t, y_p)
        except Exception:
            pearson_r = float("nan")
        try:
            spearman_r, _ = stats.spearmanr(y_t, y_p)
        except Exception:
            spearman_r = float("nan")
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": pearson_r, "Spearman_rho": spearman_r, "n": int(len(y_t))}


def molecule_cluster_bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, molecule_ids: np.ndarray,
    n_boot: int = 500, seed: int = 42, ci: float = 0.95
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    molecules = np.unique(molecule_ids)
    n_mol = len(molecules)
    mae_samples, rmse_samples = [], []
    for _ in range(n_boot):
        sampled = rng.choice(molecules, size=n_mol, replace=True)
        idx_list = []
        for m in sampled:
            idx = np.where(molecule_ids == m)[0]
            idx_list.extend(idx.tolist())
        idx_arr = np.array(idx_list)
        m = compute_metrics(y_true[idx_arr], y_pred[idx_arr])
        mae_samples.append(m["MAE"])
        rmse_samples.append(m["RMSE"])
    alpha = (1 - ci) / 2
    lo = int(alpha * n_boot)
    hi = int((1 - alpha) * n_boot)
    return {
        "MAE": {
            "mean": float(np.mean(mae_samples)),
            "ci_low": float(np.sort(mae_samples)[lo]),
            "ci_high": float(np.sort(mae_samples)[hi]),
        },
        "RMSE": {
            "mean": float(np.mean(rmse_samples)),
            "ci_low": float(np.sort(rmse_samples)[lo]),
            "ci_high": float(np.sort(rmse_samples)[hi]),
        },
    }


def predict_subset(manifest_path: Path, out_pred_csv: Path, device: str = "cuda") -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    rows_pred: List[Dict[str, Any]] = []
    summary_per_model_seed: List[Dict[str, Any]] = []

    for task in TASKS:
        truth_col = TRUTH_COL[task]
        if truth_col not in df.columns:
            print(f"  [warn] missing {truth_col}; skip {task}", flush=True)
            continue
        # prepare graph data ONCE per task (same preprocessing for all models/seeds)
        # 关键: 预先剔除 truth 列 NaN 行, 保证 df_task 与 load_adj_format 行对齐
        df_task = df[df[truth_col].notna()].reset_index(drop=True)
        # Build a tmp CSV with sample_id + smiles + atom_on_ring + target (truth_col)
        tmp = df_task[["sample_id", "canonical_smiles", "target_ring_atoms", truth_col]].copy()
        tmp = tmp.rename(columns={"canonical_smiles": "smiles", truth_col: task})
        # atom_on_ring: 1-indexed list from target_ring_atoms
        # handle bracket+space format like "[1, 6, 5, 4, 3, 2]" from corrected CSV
        tmp["atom_on_ring"] = tmp["target_ring_atoms"].fillna("").astype(str).apply(
            lambda s: [int(x) for x in str(s).replace("[", "").replace("]", "").replace(" ", "").split(",") if x.strip()]
        )
        tmp = tmp.drop(columns=["target_ring_atoms"])
        tmp_path = AUDIT_OUT / f"_tmp_{task}_{manifest_path.stem}.csv"
        tmp.to_csv(tmp_path, index=False)
        print(f"[{task}] preprocess {len(tmp)} rows with ring_flag={RING_FLAG}", flush=True)
        data = load_adj_format(str(tmp_path), task, NVL, MAX_ATOMS, RING_FLAG, device=device)

        y_true = data["outputs"].cpu().numpy()
        smiles_arr = data["smiles"]
        # molecule id for bootstrap: canonical SMILES, aligned with cleaned rows (df_task)
        mol_ids = df_task["canonical_smiles"].astype(str).to_numpy()

        for model_name in MODELS:
            spec = MODELS[model_name]
            for seed in spec["available_seeds"]:
                try:
                    y_pred = predict_one_model(task, model_name, seed, data, device=device)
                except FileNotFoundError as e:
                    print(f"  [skip] {model_name} {task} seed={seed}: {e}", flush=True)
                    continue
                # per-row prediction records (df_task is row-aligned with y_true/y_pred)
                for i, sid in enumerate(df_task["sample_id"].astype(str).tolist()):
                    if i >= len(y_pred):
                        continue
                    rows_pred.append({
                        "sample_id": sid,
                        "canonical_smiles": df_task.iloc[i]["canonical_smiles"],
                        "molecule_id": df_task.iloc[i]["canonical_smiles"],
                        "ring_name": df_task.iloc[i].get("ring_name", ""),
                        "ring_pos": df_task.iloc[i].get("ring_pos", ""),
                        "target_ring_id": df_task.iloc[i].get("target_ring_id", ""),
                        "sub_name": df_task.iloc[i].get("sub_name", ""),
                        "sub_type": df_task.iloc[i].get("sub_type", ""),
                        "fused": df_task.iloc[i].get("fused", ""),
                        "task": task,
                        "model": model_name,
                        "seed": seed,
                        "true_value": float(y_true[i]) if np.isfinite(y_true[i]) else float("nan"),
                        "pred_value": float(y_pred[i]) if np.isfinite(y_pred[i]) else float("nan"),
                        "signed_error": float(y_pred[i] - y_true[i]) if (np.isfinite(y_pred[i]) and np.isfinite(y_true[i])) else float("nan"),
                        "abs_error": float(abs(y_pred[i] - y_true[i])) if (np.isfinite(y_pred[i]) and np.isfinite(y_true[i])) else float("nan"),
                        "exact_seen": (manifest_path == SEEN_MANIFEST),
                    })

                # per-(model, task, seed) metrics
                m = compute_metrics(y_true, y_pred)
                # bootstrap CI
                boot = molecule_cluster_bootstrap_ci(y_true, y_pred, mol_ids, n_boot=500, seed=seed + 2024)
                summary_per_model_seed.append({
                    "task": task,
                    "model": model_name,
                    "seed": seed,
                    "n_unique_molecules": int(df_task["canonical_smiles"].nunique()),
                    "n_ring_records": int(len(df_task)),
                    "n_valid_labels": m["n"],
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "R2": m["R2"],
                    "Pearson_r": m["Pearson_r"],
                    "Spearman_rho": m["Spearman_rho"],
                    "MAE_boot_mean": boot["MAE"]["mean"],
                    "MAE_ci_low": boot["MAE"]["ci_low"],
                    "MAE_ci_high": boot["MAE"]["ci_high"],
                    "RMSE_boot_mean": boot["RMSE"]["mean"],
                    "RMSE_ci_low": boot["RMSE"]["ci_low"],
                    "RMSE_ci_high": boot["RMSE"]["ci_high"],
                })
                print(f"  [{task}/{model_name}/seed={seed}] MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} n={m['n']}", flush=True)

        # cleanup tmp
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # write per-row predictions
    pred_df = pd.DataFrame(rows_pred)
    pred_df.to_csv(out_pred_csv, index=False)
    print(f"  wrote {out_pred_csv} ({len(pred_df)} rows)", flush=True)
    return pd.DataFrame(summary_per_model_seed)


def aggregate_mean_sd(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    g = per_seed_df.groupby(["task", "model"], as_index=False).agg(
        n_unique_molecules=("n_unique_molecules", "first"),
        n_ring_records=("n_ring_records", "first"),
        n_valid_labels_mean=("n_valid_labels", "mean"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", lambda x: float(np.std(x, ddof=1))),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", lambda x: float(np.std(x, ddof=1))),
        R2_mean=("R2", "mean"),
        R2_std=("R2", lambda x: float(np.std(x, ddof=1))),
        Pearson_r_mean=("Pearson_r", "mean"),
        Pearson_r_std=("Pearson_r", lambda x: float(np.std(x, ddof=1))),
        Spearman_rho_mean=("Spearman_rho", "mean"),
        Spearman_rho_std=("Spearman_rho", lambda x: float(np.std(x, ddof=1))),
        MAE_ci_low_mean=("MAE_ci_low", "mean"),
        MAE_ci_high_mean=("MAE_ci_high", "mean"),
        n_seeds=("seed", "count"),
    )
    return g


def main() -> None:
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    # When CUDA_VISIBLE_DEVICES is set, the only visible device is index 0
    device = "cuda:0"
    print(f"[start] CUDA_VISIBLE_DEVICES={gpu_id}, using device={device}", flush=True)

    # ---- main external: lunci10_clean_external ----
    print("=" * 60, flush=True)
    print("MAIN external: lunci10_clean_external (Phase 4 strict zero-shot)", flush=True)
    print("=" * 60, flush=True)
    per_seed_clean = predict_subset(CLEAN_MANIFEST, PRED_CLEAN_CSV, device=device)

    # ---- diagnostic: lunci10_exact_seen ----
    print("=" * 60, flush=True)
    print("DIAGNOSTIC: lunci10_exact_seen (NOT external evidence)", flush=True)
    print("=" * 60, flush=True)
    per_seed_seen = predict_subset(SEEN_MANIFEST, PRED_SEEN_CSV, device=device)

    # ---- aggregate mean ± SD ----
    summary = aggregate_mean_sd(per_seed_clean)
    seen_summary = aggregate_mean_sd(per_seed_seen)
    seen_summary = seen_summary.add_prefix("seen_")
    full = pd.concat([summary, seen_summary], axis=1)
    full.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[wrote] {SUMMARY_CSV}", flush=True)
    print(full.to_string(), flush=True)


if __name__ == "__main__":
    main()