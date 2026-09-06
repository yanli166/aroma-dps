"""Experiment A: Internal Scaffold OOD — training script.

Protocol (per user):
  1. Unified 80/20 Murcko scaffold split (same across HOMA/NICS/MBCO)
  2. 5-fold Group CV on development (by scaffold group)
  3. Per fold: early stopping → record best epoch
  4. E* = median(5 best epochs)
  5. Train full development for E* epochs (no early stopping)
  6. Evaluate on fixed OOD scaffold test

Models: Base_GNN (GNNModel), RC_MPNN (MPNNModel)
Tasks: HOMA, NICS_1zz, MBCO
Seeds: 42, 123, 456, 789, 2024
Total: 3 × 2 × 5 × (5 CV + 1 final) = 180 runs

Outputs:
  02_scaffold_ood/internal_scaffold_ood_predictions.csv
  02_scaffold_ood/internal_scaffold_ood_summary.csv
  02_scaffold_ood/estar_record.csv  (for Experiment B reuse)
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"
DATA1_END = CODE_END / "data1_end"
ORIG_MODELS_ROOT = str(PROJ_ROOT / "unified_models")
SPLIT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/02_scaffold_splits"
OUT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/02_scaffold_ood"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in (str(PROJ_ROOT), str(CODE_END), ORIG_MODELS_ROOT):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from common.graph_data import load_adj_format  # noqa: E402
from unified_models.gnn.model import GNNModel  # noqa: E402
from unified_models.mpnn.model import MPNNModel  # noqa: E402

# --------------------------- Config ---------------------------

TASKS = [
    {"name": "HOMA", "csv": DATA1_END / "collet_homa_0716.csv", "target_col": "homa_value"},
    {"name": "NICS_1zz", "csv": DATA1_END / "collet_nics_0716.csv", "target_col": "NICS_value"},
    {"name": "MBCO", "csv": DATA1_END / "collet_mbco_0716.csv", "target_col": "mbco_value"},
]

MODELS = {
    "Base_GNN": GNNModel,
    "RC_MPNN": MPNNModel,
}

SEEDS = [42, 123, 456, 789, 2024]
NVL = 60
MAX_ATOMS = 75
RING_FLAG = 10
N_FOLDS = 5

PARAMS = {
    "hidden_dim": 128, "n_conv": 3, "n_hidden": 2,
    "lr": 0.001, "weight_decay": 1e-5, "batch_size": 64,
    "n_epochs": 200, "patience": 30,
}


# --------------------------- Helpers ---------------------------

def set_full_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_smiles(smi: str) -> str:
    from rdkit import Chem
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def build_model(model_name: str) -> nn.Module:
    kwargs = dict(
        node_vec_len=NVL, hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv"],
        n_hidden=PARAMS["n_hidden"], n_outputs=1, p_dropout=PARAMS["p_dropout"] if "p_dropout" in PARAMS else 0.2,
        mode="label",
    )
    return MODELS[model_name](**kwargs)


def load_data_with_split(task: Dict, device: str = "cpu") -> Tuple[Dict, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load graph data and align with scaffold split CSV.

    Returns: (data, dev_idx, test_idx, cv_folds)
      dev_idx: indices into data for development set
      test_idx: indices into data for OOD test set
      cv_folds: list of (train_idx, val_idx) for 5-fold Group CV
    """
    split_csv = SPLIT_DIR / f"internal_{task['name']}_scaffold_split.csv"
    split_df = pd.read_csv(split_csv)
    canon_to_split = dict(zip(split_df["canon_smiles"], split_df["split"]))
    canon_to_fold = dict(zip(split_df["canon_smiles"], split_df["cv_fold"]))

    # load graph data (load_adj_format calls clean_dataset_csv internally)
    data = load_adj_format(
        str(task["csv"]), task["target_col"], NVL, MAX_ATOMS,
        ring_flag_value=RING_FLAG, device=device,
    )

    # align via canonical SMILES
    smiles_list = data["smiles"]
    splits = np.array([canon_to_split.get(canonical_smiles(s), "unknown") for s in smiles_list])
    folds = np.array([canon_to_fold.get(canonical_smiles(s), -1) for s in smiles_list])

    dev_idx = np.where(splits == "development")[0]
    test_idx = np.where(splits == "OOD_test")[0]

    cv_folds = []
    for fi in range(N_FOLDS):
        val_idx = np.where(folds == fi)[0]
        val_idx = np.intersect1d(val_idx, dev_idx)
        train_idx = np.setdiff1d(dev_idx, val_idx)
        cv_folds.append((train_idx, val_idx))

    return data, dev_idx, test_idx, cv_folds


# --------------------------- Training ---------------------------

def train_fold(model_cls, data, train_idx, val_idx, device, seed, n_epochs=200, patience=30):
    """Train one CV fold with early stopping. Returns (model, best_epoch)."""
    set_full_seed(seed)
    model = model_cls(
        node_vec_len=NVL, hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv"],
        n_hidden=PARAMS["n_hidden"], n_outputs=1, p_dropout=0.2, mode="label",
    ).to(device)

    node_mats = data["node_mats"]
    adj_mats = data["adj_mats"]
    outputs = data["outputs"]
    batch_size = PARAMS["batch_size"]
    n_train, n_val = len(train_idx), len(val_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    best_val, best_state, best_epoch, pcount = float("inf"), None, 0, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            if len(bi) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze(-1)
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for i in range(0, n_val, batch_size):
                bi = val_idx_t[i:i + batch_size]
                if len(bi) == 0:
                    continue
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze(-1))
                vt.append(outputs[bi])
            vp = torch.cat(vp)
            vt = torch.cat(vt)
            vloss = loss_fn(vp, vt).item()
        scheduler.step(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            pcount = 0
        else:
            pcount += 1
        if pcount >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler
    return model, best_epoch


def train_fixed_epochs(model_cls, data, train_idx, device, seed, n_epochs):
    """Train on full development for fixed E* epochs (no early stopping)."""
    set_full_seed(seed)
    model = model_cls(
        node_vec_len=NVL, hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv"],
        n_hidden=PARAMS["n_hidden"], n_outputs=1, p_dropout=0.2, mode="label",
    ).to(device)

    node_mats = data["node_mats"]
    adj_mats = data["adj_mats"]
    outputs = data["outputs"]
    batch_size = PARAMS["batch_size"]
    n_train = len(train_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            if len(bi) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze(-1)
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # no validation, no early stopping — fixed E* epochs
    del optimizer, scheduler
    return model


@torch.no_grad()
def evaluate(model, data, idx, device, batch_size=64):
    model.eval()
    node_mats = data["node_mats"]
    adj_mats = data["adj_mats"]
    outputs = data["outputs"]
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)

    preds, trues = [], []
    for i in range(0, len(idx_t), batch_size):
        bi = idx_t[i:i + batch_size]
        if len(bi) == 0:
            continue
        p = model(node_mats[bi], adj_mats[bi]).squeeze(-1)
        preds.append(p)
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return pred, true


def compute_metrics(y_true, y_pred):
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[finite]
    y_p = y_pred[finite]
    err = y_p - y_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
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
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": pearson_r, "Spearman_rho": spearman_r, "n": int(len(y_t))}


# --------------------------- Main runner ---------------------------

def run_single(task: Dict, model_name: str, seed: int, device: str = "cuda") -> Dict[str, Any]:
    """Run one (task, model, seed) config: 5-fold CV → E* → final → test."""
    model_cls = MODELS[model_name]
    data, dev_idx, test_idx, cv_folds = load_data_with_split(task, device=device)

    # Step 1: 5-fold Group CV → collect best epochs
    best_epochs = []
    for fi, (tr_idx, va_idx) in enumerate(cv_folds):
        _, be = train_fold(model_cls, data, tr_idx, va_idx, device, seed,
                           n_epochs=PARAMS["n_epochs"], patience=PARAMS["patience"])
        best_epochs.append(be)
        print(f"    fold {fi}: best_epoch={be}", flush=True)

    # Step 2: E* = median(best epochs)
    estar = int(np.median(best_epochs))
    print(f"    E* = median({best_epochs}) = {estar}", flush=True)

    # Step 3: Train full development for E* epochs
    final_model = train_fixed_epochs(model_cls, data, dev_idx, device, seed, n_epochs=estar)

    # Step 4: Evaluate on fixed OOD test
    pred, true = evaluate(final_model, data, test_idx, device)
    metrics = compute_metrics(true, pred)

    print(f"    TEST: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f} "
          f"Pearson={metrics['Pearson_r']:.4f} Spearman={metrics['Spearman_rho']:.4f} n={metrics['n']}", flush=True)

    return {
        "task": task["name"],
        "model": model_name,
        "seed": seed,
        "best_epochs_cv": best_epochs,
        "E_star": estar,
        "n_test": metrics["n"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "Pearson_r": metrics["Pearson_r"],
        "Spearman_rho": metrics["Spearman_rho"],
        "predictions": list(zip(pred.tolist(), true.tolist())),
    }


def main(task_only: str = ""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [t for t in TASKS if not task_only or t["name"] == task_only]
    out_tag = task_only or "all"
    print(f"[start] device={device}")
    print(f"[config] tasks={[t['name'] for t in tasks]}, models={list(MODELS.keys())}, seeds={SEEDS}")
    print(f"[output] {OUT_DIR} (tag={out_tag})")

    all_results = []
    estar_rows = []
    pred_rows = []

    for task in tasks:
        for model_name in MODELS:
            for seed in SEEDS:
                print(f"\n{'='*60}")
                print(f"[run] task={task['name']} model={model_name} seed={seed}")
                print(f"{'='*60}")
                t0 = time.time()
                result = run_single(task, model_name, seed, device=device)
                elapsed = time.time() - t0
                result["train_time_sec"] = elapsed
                print(f"  time: {elapsed:.1f}s", flush=True)

                all_results.append(result)
                estar_rows.append({
                    "task": task["name"], "model": model_name, "seed": seed,
                    "best_epochs_cv": str(result["best_epochs_cv"]),
                    "E_star": result["E_star"],
                })
                for i, (p, t) in enumerate(result["predictions"]):
                    pred_rows.append({
                        "task": task["name"], "model": model_name, "seed": seed,
                        "test_idx": i, "pred_value": p, "true_value": t,
                    })

    # Save predictions
    pred_df = pd.DataFrame(pred_rows)
    pred_path = OUT_DIR / f"internal_scaffold_ood_predictions_{out_tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[wrote] {pred_path} ({len(pred_df)} rows)")

    # Save E* record (for Experiment B)
    estar_df = pd.DataFrame(estar_rows)
    estar_path = OUT_DIR / f"estar_record_{out_tag}.csv"
    estar_df.to_csv(estar_path, index=False)
    print(f"[wrote] {estar_path}")

    # Save summary (mean ± SD per task×model)
    summary_rows = []
    for task in TASKS:
        for model_name in MODELS:
            subset = [r for r in all_results if r["task"] == task["name"] and r["model"] == model_name]
            if not subset:
                continue
            maes = [r["MAE"] for r in subset]
            rmses = [r["RMSE"] for r in subset]
            r2s = [r["R2"] for r in subset]
            pearsons = [r["Pearson_r"] for r in subset]
            spearmans = [r["Spearman_rho"] for r in subset]
            summary_rows.append({
                "task": task["name"],
                "model": model_name,
                "n_seeds": len(subset),
                "n_test_scaffolds": 356,
                "MAE_mean": np.mean(maes), "MAE_std": np.std(maes, ddof=1) if len(maes) > 1 else 0,
                "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses, ddof=1) if len(rmses) > 1 else 0,
                "R2_mean": np.mean(r2s), "R2_std": np.std(r2s, ddof=1) if len(r2s) > 1 else 0,
                "Pearson_mean": np.mean(pearsons), "Pearson_std": np.std(pearsons, ddof=1) if len(pearsons) > 1 else 0,
                "Spearman_mean": np.mean(spearmans), "Spearman_std": np.std(spearmans, ddof=1) if len(spearmans) > 1 else 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / f"internal_scaffold_ood_summary_{out_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[wrote] {summary_path}")

    # Print final summary
    print(f"\n{'='*80}")
    print("Experiment A: Internal Scaffold OOD — Summary")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-only", default="", help="run only a single task (HOMA / NICS_1zz / MBCO)")
    args = parser.parse_args()
    main(task_only=args.task_only)
