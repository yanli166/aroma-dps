"""Experiment B: Internal Scaffold Exposure Curve.

Protocol (per user):
  - Reuses E* from Experiment A (task x model_seed) — NO new epoch selection.
  - For each exposure fraction (20/40/60/80/100%): train on exposed scaffolds only.
  - Fixed OOD test scaffolds (never in training), same test set across fractions.
  - Nested: Train20 ⊂ Train40 ⊂ ... ⊂ Train100.
  - Model: RC_MPNN. Tasks: HOMA, NICS_1zz, MBCO. Seeds: 5.
  - Total: 3 x 5 x 5 = 75 final training runs.

Outputs:
  internal_scaffold_exposure_curve.csv
"""
from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path

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
EXPA_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/02_scaffold_ood"
OUT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/03_scaffold_exposure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in (str(PROJ_ROOT), str(CODE_END), ORIG_MODELS_ROOT):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from common.graph_data import load_adj_format  # noqa: E402
from unified_models.mpnn.model import MPNNModel  # noqa: E402

TASKS = [
    {"name": "HOMA", "csv": DATA1_END / "collet_homa_0716.csv", "target_col": "homa_value"},
    {"name": "NICS_1zz", "csv": DATA1_END / "collet_nics_0716.csv", "target_col": "NICS_value"},
    {"name": "MBCO", "csv": DATA1_END / "collet_mbco_0716.csv", "target_col": "mbco_value"},
]
SEEDS = [42, 123, 456, 789, 2024]
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
NVL, MAX_ATOMS = 60, 75
RING_FLAG = 10
PARAMS = {"hidden_dim": 128, "n_conv": 3, "n_hidden": 2, "lr": 0.001,
          "weight_decay": 1e-5, "batch_size": 64}


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


def build_model() -> nn.Module:
    return MPNNModel(node_vec_len=NVL, hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv"],
                     n_hidden=PARAMS["n_hidden"], n_outputs=1, p_dropout=0.2, mode="label")


def load_task_data(task: dict, device: str) -> dict:
    data = load_adj_format(str(task["csv"]), task["target_col"], NVL, MAX_ATOMS,
                           ring_flag_value=RING_FLAG, device=device)
    return data


def train_fixed_epochs(model, data, train_idx, device, seed, n_epochs):
    set_full_seed(seed)
    node_mats, adj_mats, outputs = data["node_mats"], data["adj_mats"], data["outputs"]
    batch_size = PARAMS["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    loss_fn = nn.MSELoss()
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    n = len(train_idx)
    for _ in range(n_epochs):
        model.train()
        perm = train_idx_t[torch.randperm(n, device=device)]
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            if len(bi) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            preds = model(node_mats[bi], adj_mats[bi]).squeeze(-1)
            loss = loss_fn(preds, outputs[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, data, idx, device, batch_size=64):
    model.eval()
    node_mats, adj_mats, outputs = data["node_mats"], data["adj_mats"], data["outputs"]
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
    preds, trues = [], []
    for i in range(0, len(idx_t), batch_size):
        bi = idx_t[i:i + batch_size]
        if len(bi) == 0:
            continue
        preds.append(model(node_mats[bi], adj_mats[bi]).squeeze(-1))
        trues.append(outputs[bi])
    pred = torch.cat(preds).cpu().numpy()
    true = torch.cat(trues).cpu().numpy()
    return pred, true


def compute_metrics(y_true, y_pred):
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[finite], y_pred[finite]
    err = y_p - y_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if len(y_t) >= 2:
        try:
            rho, _ = stats.spearmanr(y_t, y_p)
        except Exception:
            rho = float("nan")
    else:
        rho = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman_rho": rho, "n": int(len(y_t))}


def main(task_only: str = ""):
    device = "cuda:0"
    print(f"[start] device={device} task_only={task_only or 'ALL'}", flush=True)

    # Load E* records from Experiment A
    estar_df = pd.read_csv(EXPA_DIR / "estar_record.csv")
    estar_map = {}
    for _, r in estar_df.iterrows():
        key = (r["task"], r["model"], int(r["seed"]))
        estar_map[key] = int(r["E_star"])
    print(f"[E*] loaded {len(estar_map)} records", flush=True)

    # Load split CSVs
    tasks = [t for t in TASKS if not task_only or t["name"] == task_only]
    split_dfs = {}
    exp_split_dfs = {}
    for task in tasks:
        name = task["name"]
        split_dfs[name] = pd.read_csv(SPLIT_DIR / f"internal_{name}_scaffold_split.csv")
        exp_split_dfs[name] = pd.read_csv(SPLIT_DIR / f"internal_{name}_exposure_curve_split.csv")

    # Global dev scaffold order (same across tasks, from build_scaffold_splits.py)
    dev_order_df = pd.read_csv(SPLIT_DIR / "global_dev_scaffold_order.csv")
    global_dev_order = dev_order_df["scaffold"].tolist()
    n_global_dev = len(global_dev_order)

    all_rows = []
    for task in tasks:
        name = task["name"]
        split_df = split_dfs[name]
        # canonical smiles -> scaffold, split
        canon_to_scaff = dict(zip(split_df["canon_smiles"], split_df["scaffold"]))
        canon_to_split = dict(zip(split_df["canon_smiles"], split_df["split"]))
        # scaffold -> mols (task-level)
        scaffold_to_mols = {}
        for _, r in split_df.iterrows():
            scaffold_to_mols.setdefault(r["scaffold"], set()).add(r["canon_smiles"])
        # dev scaffolds present in THIS task, keeping global relative order
        dev_scaffolds_set = set(s for s in scaffold_to_mols
                                if canon_to_split.get(next(iter(scaffold_to_mols[s])), "") == "development")
        dev_order = [s for s in global_dev_order if s in dev_scaffolds_set]

        data = load_task_data(task, device)
        smiles_list = data["smiles"]
        idx_canon = [canonical_smiles(s) for s in smiles_list]
        canon_to_idx = {}
        for i, c in enumerate(idx_canon):
            canon_to_idx.setdefault(c, i)

        # test indices
        test_idx = np.array([canon_to_idx[c] for c, spl in canon_to_split.items()
                             if spl == "OOD_test" and c in canon_to_idx])
        test_mols = set(c for c, spl in canon_to_split.items() if spl == "OOD_test")

        for seed in SEEDS:
            key = (name, "RC_MPNN", seed)
            if key not in estar_map:
                # try Base_GNN E* as fallback? No — use RC_MPNN if present, else GNN
                key = (name, "Base_GNN", seed)
            estar = estar_map.get(key)
            if estar is None:
                print(f"  [WARN] no E* for {name} seed={seed}, skip", flush=True)
                continue
            print(f"\n=== {name} seed={seed} E*={estar} ===", flush=True)

            for frac in FRACTIONS:
                n_sel = int(len(dev_order) * frac)
                exposed_scaffs = set(dev_order[:n_sel])
                exposed_mols = set()
                for s in exposed_scaffs:
                    exposed_mols.update(scaffold_to_mols[s])
                train_idx = np.array([canon_to_idx[c] for c in exposed_mols if c in canon_to_idx])

                # skip if empty
                if len(train_idx) < 2:
                    print(f"  [skip] frac={frac} train={len(train_idx)}", flush=True)
                    continue

                t0 = time.time()
                model = build_model().to(device)
                train_fixed_epochs(model, data, train_idx, device, seed, estar)
                pred, true = evaluate(model, data, test_idx, device)
                m = compute_metrics(true, pred)
                elapsed = time.time() - t0

                all_rows.append({
                    "task": name, "split_seed": 42, "model_seed": seed,
                    "exposure_fraction": frac,
                    "n_train_scaffolds": len(exposed_scaffs),
                    "n_train_molecules": len(exposed_mols),
                    "n_test_scaffolds": n_global_dev - len(dev_order),  # global test scaffold count for this task
                    "n_test_molecules": len(test_mols),
                    "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                    "Spearman_rho": m["Spearman_rho"], "n_test": m["n"],
                    "train_time_sec": elapsed,
                })
                print(f"  frac={frac:.0%} train={len(train_idx)}: R2={m['R2']:.4f} MAE={m['MAE']:.4f} ({elapsed:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache()

    out_df = pd.DataFrame(all_rows)
    tag = task_only or "all"
    out_path = OUT_DIR / f"internal_scaffold_exposure_curve_{tag}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[wrote] {out_path} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-only", default="", help="run only a single task (HOMA / NICS_1zz / MBCO)")
    args = parser.parse_args()
    main(task_only=args.task_only)
