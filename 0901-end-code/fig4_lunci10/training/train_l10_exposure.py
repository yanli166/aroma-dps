"""Experiment C: lunci10 Scaffold Exposure / Adaptation Curve.

Protocol (per user):
  - ring_name = scaffold-family unit.
  - 5 split seeds; each fixes ~20% ring families as PERMANENT OOD test (never in fine-tune).
  - Exposure pool: nested 20/40/60/80/100% fractions (E20⊂E40⊂...⊂E100).
  - 0% exposure = frozen RC-MPNN zero-shot prediction (ZERO_SHOT), no fine-tune.
  - >0%: fine-tune from frozen RC_MPNN checkpoint on exposed scaffolds only.
    - exposed data split into train/val (molecule-level) for early stopping.
    - permanent OOD test NEVER in train/val/early-stopping.
  - Molecule-level: if any record of a molecule is in test family, all its records go to test.
  - Models: RC_MPNN. Tasks: HOMA, NICS_1zz, MBCO. split seeds x model seeds = 5x5.
  - Total fine-tune: 3 x 5 x 5 x 5 = 375; zero-shot evals: 3 x 5 x 5 = 75.

Metrics: micro (all test records) + macro-ring-family MAE (mean of per-family MAE).

Outputs (in results/fig4_lunci10_final/04_l10_exposure/):
  l10_adaptation_predictions.csv
  l10_scaffold_exposure_curve.csv
"""
from __future__ import annotations

import os
import sys
import time
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"
ORIG_MODELS_ROOT = str(PROJ_ROOT / "unified_models")
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
SPLIT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/02_scaffold_splits"
LAYER3_DIR = CODE_END / "results/layer3_ring_fixed"
OUT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/04_l10_exposure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in (str(PROJ_ROOT), str(CODE_END), ORIG_MODELS_ROOT):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from common.graph_data import load_adj_format  # noqa: E402
from unified_models.mpnn.model import MPNNModel  # noqa: E402

CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
TRUTH_COL = {"HOMA": "HOMA", "NICS_1zz": "NICS_ZZ", "MBCO": "MBCO"}
SPLIT_SEEDS = [42, 123, 456, 789, 2024]
MODEL_SEEDS = [42, 123, 456, 789, 2024]
FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NONZERO_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_FRAC = 0.2
NVL, MAX_ATOMS = 60, 75
RING_FLAG = 10

FT_PARAMS = {
    "lr": 0.001, "weight_decay": 1e-5, "batch_size": 32,
    "n_epochs": 80, "patience": 12, "val_frac": 0.2,
}


def set_full_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_ring_atoms(s: str):
    if not isinstance(s, str) or not s:
        return []
    return [int(x) for x in s.replace("[", "").replace("]", "").replace(" ", "").split(",") if x.strip()]


def build_model() -> nn.Module:
    return MPNNModel(node_vec_len=NVL, hidden_dim=128, n_conv=3, n_hidden=2,
                     n_outputs=1, p_dropout=0.2, mode="label")


def load_frozen_state(task: str, seed: int, device: str) -> dict:
    ckpt = LAYER3_DIR / f"seed_{seed}" / task / "MPNN_label" / "best_model.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {ckpt}")
    return torch.load(str(ckpt), map_location=device, weights_only=False)


# --------------------------- split reconstruction ---------------------------

def build_ring_family_splits(df: pd.DataFrame) -> dict:
    """Reconstruct per split_seed exposed/test molecule sets (same as build_scaffold_splits)."""
    ring_to_mols: dict = {}
    mol_to_rings: dict = {}
    for _, r in df.iterrows():
        rn, mol = r["ring_name"], r["canonical_smiles"]
        ring_to_mols.setdefault(rn, set()).add(mol)
        mol_to_rings.setdefault(mol, set()).add(rn)
    ring_names = sorted(ring_to_mols.keys())
    n_rings = len(ring_names)

    splits = {}
    for ss in SPLIT_SEEDS:
        rng = np.random.RandomState(ss)
        ring_order = list(ring_names)
        rng.shuffle(ring_order)
        n_test_fam = int(TEST_FRAC * n_rings)
        test_families = set(ring_order[:n_test_fam])
        pool_order = list(ring_order[n_test_fam:])

        test_mols = set()
        for fam in test_families:
            test_mols.update(ring_to_mols[fam])
        for mol, rings in mol_to_rings.items():
            if rings & test_families:
                test_mols.add(mol)

        exposure = {}
        for frac in NONZERO_FRACTIONS:
            n_sel = int(len(pool_order) * frac)
            exposed_fams = set(pool_order[:n_sel])
            exposed_mols = set()
            for fam in exposed_fams:
                exposed_mols.update(ring_to_mols[fam])
            exposed_mols = exposed_mols - test_mols
            exposure[frac] = {"families": exposed_fams, "mols": exposed_mols}

        splits[ss] = {
            "test_families": test_families,
            "test_mols": test_mols,
            "exposure": exposure,
        }
    return splits


# --------------------------- data loading ---------------------------

def load_task_data(task: str, df: pd.DataFrame, device: str):
    """Build graph data for one task; return aligned arrays with metadata."""
    truth_col = TRUTH_COL[task]
    tmp = df[["sample_id", "canonical_smiles", "target_ring_atoms", "ring_name", truth_col]].copy()
    tmp = tmp.rename(columns={"canonical_smiles": "smiles", truth_col: task})
    tmp = tmp.dropna(subset=[task]).reset_index(drop=True)
    tmp["atom_on_ring"] = tmp["target_ring_atoms"].apply(parse_ring_atoms)
    tmp = tmp.drop(columns=["target_ring_atoms"])
    tmp_path = AUDIT_OUT / f"_tmp_expc_{task}.csv"
    tmp.to_csv(tmp_path, index=False)
    print(f"[{task}] preprocess {len(tmp)} rows ring_flag={RING_FLAG}", flush=True)
    data = load_adj_format(str(tmp_path), task, NVL, MAX_ATOMS, RING_FLAG, device=device)

    # align data index -> metadata via canonical smiles
    idx_canon = list(data["smiles"])
    # map each data row to its tmp row (preserve order)
    row_map = []
    canon_to_rows = {}
    for i, c in enumerate(tmp["smiles"]):
        canon_to_rows.setdefault(c, []).append(i)
    for c in idx_canon:
        rows = canon_to_rows.get(c, [])
        row_map.append(rows[0] if rows else -1)

    meta = tmp.iloc[row_map].reset_index(drop=True) if all(i >= 0 for i in row_map) else tmp.iloc[[i for i in row_map if i >= 0]].reset_index(drop=True)
    # if any -1, we need to filter data too; handle below
    valid = [i for i in row_map if i >= 0]
    if len(valid) != len(row_map):
        # filter data arrays
        keep = np.array([i for i in range(len(row_map)) if row_map[i] >= 0])
        data = {k: (v[keep] if hasattr(v, "__getitem__") and len(v) == len(row_map) else v) for k, v in data.items()}
        row_map = [row_map[i] for i in keep]
        meta = tmp.iloc[row_map].reset_index(drop=True)
    return data, meta


# --------------------------- fine-tune ---------------------------

def fine_tune(model, data, train_idx, val_idx, device, seed):
    """Fine-tune from loaded frozen state with early stopping on val."""
    set_full_seed(seed)
    node_mats, adj_mats, outputs = data["node_mats"], data["adj_mats"], data["outputs"]
    batch_size = FT_PARAMS["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=FT_PARAMS["lr"],
                                 weight_decay=FT_PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=FT_PARAMS["patience"] // 3, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    n_tr = len(train_idx)

    best_val, best_state, pcount = float("inf"), None, 0
    for epoch in range(1, FT_PARAMS["n_epochs"] + 1):
        model.train()
        perm = train_t[torch.randperm(n_tr, device=device)]
        for i in range(0, n_tr, batch_size):
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
            for i in range(0, len(val_idx), batch_size):
                bi = val_t[i:i + batch_size]
                if len(bi) == 0:
                    continue
                vp.append(model(node_mats[bi], adj_mats[bi]).squeeze(-1))
                vt.append(outputs[bi])
            vloss = loss_fn(torch.cat(vp), torch.cat(vt)).item()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pcount = 0
        else:
            pcount += 1
        if pcount >= FT_PARAMS["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, data, idx, device, batch_size=64):
    model.eval()
    node_mats, adj_mats = data["node_mats"], data["adj_mats"]
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
    preds = []
    for i in range(0, len(idx_t), batch_size):
        bi = idx_t[i:i + batch_size]
        if len(bi) == 0:
            continue
        out = model(node_mats[bi], adj_mats[bi])
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


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
            pr, _ = stats.pearsonr(y_t, y_p)
        except Exception:
            pr = float("nan")
        try:
            sr, _ = stats.spearmanr(y_t, y_p)
        except Exception:
            sr = float("nan")
    else:
        pr, sr = float("nan"), float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": pr, "Spearman_rho": sr, "n": int(len(y_t))}


def macro_ring_family_mae(y_true, y_pred, family_ids):
    """Per-ring-family MAE averaged (avoid large-family dominance)."""
    fam_maes = {}
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    for fam in np.unique(family_ids[finite]):
        m = finite & (family_ids == fam)
        if m.sum() == 0:
            continue
        fam_maes[fam] = float(np.mean(np.abs(y_pred[m] - y_true[m])))
    if not fam_maes:
        return float("nan")
    return float(np.mean(list(fam_maes.values())))


# --------------------------- main ---------------------------

def main(task_only: str = ""):
    device = "cuda:0"
    print(f"[start] device={device} task_only={task_only or 'ALL'}", flush=True)

    df = pd.read_csv(CLEAN_MANIFEST)
    print(f"[manifest] {len(df)} records, {df['canonical_smiles'].nunique()} molecules", flush=True)
    splits = build_ring_family_splits(df)

    pred_rows = []
    curve_rows = []

    tasks = [t for t in TASKS if not task_only or t == task_only]
    for task in tasks:
        data, meta = load_task_data(task, df, device)
        # data indices by canonical smiles
        meta = meta.reset_index(drop=True)
        # exposed/test index masks from meta
        mol_list = meta["smiles"].tolist()
        fam_list = meta["ring_name"].tolist()
        y_true = data["outputs"].cpu().numpy()
        n = len(mol_list)

        for ss in SPLIT_SEEDS:
            sp = splits[ss]
            test_fams = sp["test_families"]
            test_mols = sp["test_mols"]
            # test indices (all records of test molecules)
            test_idx = np.array([i for i in range(n) if mol_list[i] in test_mols])
            test_fam_arr = np.array([fam_list[i] for i in test_idx])
            test_y = y_true[test_idx]

            # print split info
            print(f"\n=== {task} split_seed={ss}: test_families={len(test_fams)}, "
                  f"test_records={len(test_idx)} ===", flush=True)

            for frac in FRACTIONS:
                for ms in MODEL_SEEDS:
                    key = (task, ss, ms, frac)
                    if frac == 0.0:
                        # ZERO_SHOT: use frozen RC_MPNN directly
                        set_full_seed(ms)
                        model = build_model().to(device)
                        model.load_state_dict(load_frozen_state(task, ms, device))
                        model.eval()
                        pred = predict(model, data, test_idx, device)
                        phase = "ZERO_SHOT"
                    else:
                        exp = sp["exposure"][frac]
                        exposed_mols = exp["mols"]
                        exp_idx_all = np.array([i for i in range(n) if mol_list[i] in exposed_mols])
                        # molecule-level train/val split
                        exp_mols = sorted(set(mol_list[i] for i in exp_idx_all))
                        rng = np.random.RandomState(ms)
                        rng.shuffle(exp_mols)
                        n_val = max(1, int(len(exp_mols) * FT_PARAMS["val_frac"]))
                        val_mols = set(exp_mols[:n_val])
                        train_mols = set(exp_mols[n_val:])
                        train_idx = np.array([i for i in exp_idx_all if mol_list[i] in train_mols])
                        val_idx = np.array([i for i in exp_idx_all if mol_list[i] in val_mols])
                        if len(train_idx) < 2 or len(val_idx) < 1:
                            print(f"  [skip] frac={frac} ms={ms}: train={len(train_idx)} val={len(val_idx)}", flush=True)
                            continue

                        set_full_seed(ms)
                        model = build_model().to(device)
                        model.load_state_dict(load_frozen_state(task, ms, device))
                        t0 = time.time()
                        model = fine_tune(model, data, train_idx, val_idx, device, ms)
                        pred = predict(model, data, test_idx, device)
                        phase = "L10_ADAPTED"

                    m = compute_metrics(test_y, pred)
                    macro_mae = macro_ring_family_mae(test_y, pred, test_fam_arr)

                    # record curve row
                    curve_rows.append({
                        "task": task, "split_seed": ss, "model_seed": ms,
                        "exposure_fraction": frac, "phase": phase,
                        "n_exposed_families": len(exp["families"]) if frac > 0 else 0,
                        "n_exposed_molecules": len(exp["mols"]) if frac > 0 else 0,
                        "n_test_families": len(test_fams),
                        "n_test_molecules": len(test_mols),
                        "n_test_records": len(test_idx),
                        "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                        "Pearson_r": m["Pearson_r"], "Spearman_rho": m["Spearman_rho"],
                        "macro_ring_family_MAE": macro_mae,
                        "n": m["n"],
                    })

                    # record predictions
                    for j, idx in enumerate(test_idx):
                        pred_rows.append({
                            "task": task, "split_seed": ss, "model_seed": ms,
                            "exposure_fraction": frac, "phase": phase,
                            "sample_id": meta.iloc[idx]["sample_id"],
                            "ring_name": meta.iloc[idx]["ring_name"],
                            "true_value": float(test_y[j]),
                            "pred_value": float(pred[j]),
                        })

                    print(f"  frac={frac:.0%} ms={ms} [{phase}]: R2={m['R2']:.4f} MAE={m['MAE']:.4f} "
                          f"macroMAE={macro_mae:.4f} n={m['n']}", flush=True)
                    del model
                    torch.cuda.empty_cache()

    # save outputs
    tag = task_only or "all"
    pred_df = pd.DataFrame(pred_rows)
    pred_path = OUT_DIR / f"l10_adaptation_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[wrote] {pred_path} ({len(pred_df)} rows)", flush=True)

    curve_df = pd.DataFrame(curve_rows)
    curve_path = OUT_DIR / f"l10_scaffold_exposure_curve_{tag}.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"[wrote] {curve_path} ({len(curve_df)} rows)", flush=True)

    # print compact summary: mean across model seeds per (task, split_seed, fraction)
    g = curve_df.groupby(["task", "split_seed", "exposure_fraction", "phase"])[["MAE", "R2", "macro_ring_family_MAE"]].agg(["mean", "std"]).reset_index()
    print("\n=== Summary (mean±std across model seeds) ===")
    print(g.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-only", default="", help="run only a single task (HOMA / NICS_1zz / MBCO)")
    args = parser.parse_args()
    main(task_only=args.task_only)
