#!/usr/bin/env python3
"""Fig.4 Full Retraining Pipeline — Strictly follows fig4.txt protocol.

Three experiments, ALL require actual model retraining (no frozen checkpoints):
1. Random IID vs Scaffold OOD
2. Leave-One-Ring-Family-Out
3. Lunci10 domain exposure curve

Execution order (per fig4.txt §36):
A→B→C→D→E(smoke)→F+G(Random baseline)→H(Scaffold OOD)→I(LORFO)→J+K(Exposure)→L(Figures+Report)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
AROMA_PROJ = Path("/home/ubuntu/aroma-dps-code")
OUT_DIR    = AROMA_PROJ / "0901-end-code" / "results" / "fig4_generalization_retrain_v4"
for d in ["00_audit","01_random_iid","02_scaffold_ood","03_ring_family_loo",
          "04_lunci10_adaptation","05_figures","06_tables","checkpoints","logs"]:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

DATA_DIR = AROMA_PROJ / "code_end" / "data1_end"
BEST_PKG = AROMA_PROJ / "best_model_package"
RMAP     = OUT_DIR / "00_audit" / "ring_family_mapping.csv"

sys.path.insert(0, str(BEST_PKG))
from graph_utils import build_graph, NODE_VEC_LEN, MAX_ATOMS
from model_arch import RingConditionedMPNN

# ── Constants ────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "HOMA":     {"task_col": "homa_value",  "ckpt": "homa_best.pt",        "use_proj": True,  "rf_val": 10},
    "NICS_ZZ":  {"task_col": "NICS_value",  "ckpt": "nics_1zz_best.pt",    "use_proj": False, "rf_val": 1},
    "MBCO":     {"task_col": "mbco_value",  "ckpt": "mbco_best.pt",        "use_proj": False, "rf_val": 1},
}
EVAL_SEEDS       = list(range(100, 105))
REPEAT_SEEDS     = [2026, 2027, 2028, 2029, 2030]
QUICK_MODE       = "--quick" in sys.argv   # 1 seed, 3 fams, 1 repeat for fast verification
if QUICK_MODE:
    EVAL_SEEDS   = [100]
    REPEAT_SEEDS = [2026]
CHUNK_SIZE       = 20
MAX_ATOMS_CFG    = 85             # Increased to handle up to 77-atom molecules
TRAIN_PARAMS     = dict(hidden_dim=128, n_conv=3, n_hidden=2, p_dropout=0.2,
                        lr=0.001, weight_decay=1e-5, batch_size=64,
                        n_epochs=200, patience=30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[fig4_generalization_retrain] Device={device}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Loading & Preparation
# ═══════════════════════════════════════════════════════════════════════

def safe_canonical(smi: str) -> str:
    from rdkit import Chem
    if not smi or not isinstance(smi, str): return ""
    mol = Chem.MolFromSmiles(smi.strip())
    return Chem.MolToSmiles(mol) if mol else ""

def murcko_scaffold(smi: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    try:
        s = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(s) if s else ""
    except Exception:
        return ""

def parse_ring_atoms(raw) -> List[int]:
    """Parse '[, 1, 10, 9]' → [0,9,8,7,6,5] or [1,10,...] depending on format."""
    if not isinstance(raw, str) or not raw: return []
    s = raw.replace("[", "").replace("]", '').replace("'", '').strip()
    if '"' in s: s = s.replace('"', '')
    parts = []
    for p in s.split(","):
        p = p.strip()
        if p:
            try: parts.append(int(p))
            except ValueError: pass
    if not parts: return []
    return parts


def load_base_dataset(task: str) -> pd.DataFrame:
    """Load base training data from collet_{task_lower}.csv"""
    tlower = task.lower().replace("_zz", "")
    path = DATA_DIR / f"collet_{tlower}_0716.csv"
    df = pd.read_csv(path)
    df["canonical_smiles"] = df["smiles"].apply(safe_canonical)
    df.dropna(subset=["canonical_smiles"], inplace=True)
    df["canonical_smiles"] = df["canonical_smiles"].apply(lambda x: x or "")
    df = df[df["canonical_smiles"].str.len() > 0]
    df["ring_atoms_list"] = df["atom_on_ring"].apply(parse_ring_atoms)
    df[f"{task}_value"] = pd.to_numeric(df[TASK_CONFIG[task]["task_col"]], errors="coerce")
    df.dropna(subset=[f"{task}_value"], inplace=True)
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    df = df.reset_index(drop=True)   # Ensure positional index == row order (matches list indexing)
    print(f"  Loaded base {task}: {len(df)} records, {df['canonical_smiles'].nunique()} molecules")
    return df


def load_lunci10_dataset() -> pd.DataFrame:
    """Load and clean lunci10 unified dataset."""
    csv_path = AROMA_PROJ / "lunci10" / "lunci10_unified.csv"
    df = pd.read_csv(csv_path)
    df["canonical_smiles"] = df["smiles"].apply(safe_canonical)
    df.dropna(subset=["canonical_smiles"], inplace=True)
    df["canonical_smiles"] = df["canonical_smiles"].apply(lambda x: x or "")
    df = df[df["canonical_smiles"].str.len() > 0]
    df["ring_atoms_list"] = df["ring_atoms"].apply(parse_ring_atoms)
    for t in ["HOMA", "MBCO", "NICS_ZZ"]:
        df[f"{t}_value"] = pd.to_numeric(df[t], errors="coerce")
    df = df.dropna(subset=["HOMA_value", "MBCO_value", "NICS_ZZ_value"])
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    # Ring family from existing ring_name column
    if "ring_name" in df.columns:
        df["ring_family"] = df["ring_name"].fillna("unknown")
    else:
        df["ring_family"] = "unknown"
    df = df.reset_index(drop=True)   # Ensure positional index == row order (matches list indexing)
    print(f"  Loaded lunci10: {len(df)} records, {df['canonical_smiles'].nunique()} molecules, "
          f"{df['ring_family'].nunique()} families, {df['scaffold'].nunique()} scaffolds")
    return df


def remove_exact_overlap(base_df: pd.DataFrame, locked_test_ids: set) -> pd.DataFrame:
    """Remove molecules from base that exactly overlap with locked test."""
    overlap_mask = base_df["canonical_smiles"].isin(locked_test_ids)
    n_removed = overlap_mask.sum()
    if n_removed > 0:
        print(f"  Removing {n_removed} overlapping molecules from base dataset")
        return base_df[~overlap_mask].reset_index(drop=True)
    return base_df


# ═══════════════════════════════════════════════════════════════════════
# 2. Model Training Engine (matches Fig.3 exactly)
# ═══════════════════════════════════════════════════════════════════════

def init_model(task: str) -> RingConditionedMPNN:
    cfg = TASK_CONFIG[task]
    return RingConditionedMPNN(
        node_vec_len=NODE_VEC_LEN, hidden_dim=TRAIN_PARAMS["hidden_dim"],
        n_conv=TRAIN_PARAMS["n_conv"], n_hidden=TRAIN_PARAMS["n_hidden"],
        p_dropout=TRAIN_PARAMS["p_dropout"],
        use_projection=cfg["use_proj"], ring_flag_value=cfg["rf_val"],
    ).to(device)


def prepare_batches(indices, canonical_smis, ring_atom_lists, values, task: str):
    """Build graphs and pack into tensors for given indices."""
    if not isinstance(indices, list): indices = list(indices)
    if not isinstance(canonical_smis, list): canonical_smis = list(canonical_smis)
    if not isinstance(ring_atom_lists, list): ring_atom_lists = list(ring_atom_lists)
    if not isinstance(values, list): values = list(values)
    n = len(canonical_smis)
    max_idx = max(indices) if indices else 0
    if max_idx >= n:
        raise ValueError(f"Index {max_idx} out of range (len={n}). indices[:5]={indices[:5]}")
    smis = [canonical_smis[idx] for idx in indices]
    rings = [ring_atom_lists[idx] for idx in indices]
    vals = np.array([values[idx] for idx in indices], dtype=np.float32)

    graphs = [build_graph(s, r, NODE_VEC_LEN, MAX_ATOMS_CFG,
                          ring_flag_value=TASK_CONFIG[task]["rf_val"])
              for s, r in zip(smis, rings)]

    node_mats = np.stack([g["node_mat"] for g in graphs]).astype(np.float32)
    adj_mats  = np.stack([g["adj_mat"] for g in graphs]).astype(np.float32)
    ring_idx  = np.stack([g["ring_indices"] for g in graphs]).astype(np.int64)
    targets   = torch.tensor(vals).unsqueeze(1).to(device)

    return (torch.tensor(node_mats, device=device),
            torch.tensor(adj_mats, device=device),
            torch.tensor(ring_idx, device=device), targets)


def train_one_task(train_indices, val_indices, canonical_smis, ring_atom_lists,
                   values, task: str, seed: int, ckpt_path: Optional[str] = None) -> Dict:
    """Train model for ONE task with one random seed. Returns metrics dict."""
    # Ensure all inputs are plain Python lists (not Series/DataFrame Index) for reliable indexing
    if not isinstance(canonical_smis, list): canonical_smis = list(canonical_smis)
    if not isinstance(ring_atom_lists, list): ring_atom_lists = list(ring_atom_lists)
    if not isinstance(values, list): values = list(values)
    rng = np.random.RandomState(seed)

    model = init_model(task)
    if ckpt_path:
        sd_raw = torch.load(ckpt_path, map_location=device)
        sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw
        model.load_state_dict(sd, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS["lr"],
                                weight_decay=TRAIN_PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=TRAIN_PARAMS["patience"] // 3, min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_epochs = TRAIN_PARAMS["n_epochs"]
    batch_size = TRAIN_PARAMS["batch_size"]

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        idx_arr = np.array(train_indices)
        perm = idx_arr[rng.permutation(len(train_indices))]
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            bi = perm[i:i + batch_size]
            node_m, adj_m, ring_i, targets = \
                prepare_batches(bi, canonical_smis, ring_atom_lists, values, task)

            pred = model(node_m, adj_m, ring_i).reshape(-1)
            loss = nn.MSELoss()(pred, targets.squeeze())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(bi)
            n_batches += 1

        # ── Validate ──
        model.eval()
        n_train_val = max(len(val_indices), 1)
        val_mae_sum = 0.0
        val_batch_count = 0

        for start in range(0, len(val_indices), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(val_indices))
            vi = val_indices[start:end]
            node_m, adj_m, ring_i, targets = \
                prepare_batches(vi, canonical_smis, ring_atom_lists, values, task)
            with torch.no_grad():
                preds = model(node_m, adj_m, ring_i).cpu().numpy().flatten()
            true_v = targets.cpu().numpy().flatten()
            val_mae_sum += np.abs(preds - true_v).sum()
            val_batch_count += len(vi)

        val_mae = val_mae_sum / max(n_train_val, 1)
        val_loss = epoch_loss / max(len(train_indices), 1)
        scheduler.step(val_mae)

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TRAIN_PARAMS["patience"]:
            break

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:3d}: train_loss={val_loss:.6f} val_mae={val_mae:.4f} "
                  f"(patience={patience_counter}/{TRAIN_PARAMS['patience']})")

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate(model, test_indices, canonical_smis, ring_atom_lists, values, task: str) -> Dict:
    """Evaluate trained model on test set. Returns metrics dict."""
    results = []
    true_all = []
    n = len(test_indices)

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        ti = test_indices[start:end]
        node_m, adj_m, ring_i, targets = \
            prepare_batches(ti, canonical_smis, ring_atom_lists, values, task)
        model.eval()
        with torch.no_grad():
            preds = model(node_m, adj_m, ring_i).cpu().numpy().flatten()
        true_v = targets.cpu().numpy().flatten()
        results.append(preds)
        true_all.extend(true_v.tolist())

    pred_all = np.concatenate(results)
    true_arr = np.array(true_all)

    mask = ~np.isnan(true_arr) & ~np.isnan(pred_all)
    yt, yp = true_arr[mask], pred_all[mask]

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    return {
        "n": int(mask.sum()),
        "R2": round(float(r2_score(yt, yp)), 6) if len(yt) >= 2 else float("nan"),
        "MAE": round(float(mean_absolute_error(yt, yp)), 6),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Smoke Test
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Step E: Quick verification that retraining works."""
    print("\n" + "=" * 60)
    print("Step E: Smoke Test — verify retraining works end-to-end")
    print("=" * 60)

    for task in TASK_CONFIG:
        print(f"\n  [{task}] ...")
        df = load_base_dataset(task)
        # Tiny subset for speed
        rng = np.random.RandomState(42)
        n_smoke = min(200, len(df))
        idx = rng.choice(len(df), n_smoke, replace=False)
        tr = idx[:int(0.8*n_smoke)]; va = idx[int(0.8*n_smoke):]
        print(f"    Train={len(tr)}, Val={len(va)}")

        model = train_one_task(tr, va, df["canonical_smiles"].tolist(),
                               df["ring_atoms_list"].tolist(),
                               df[f"{task}_value"].tolist(), task, seed=9999)

        m = evaluate(model, va, df["canonical_smiles"].tolist(),
                     df["ring_atoms_list"].tolist(),
                     df[f"{task}_value"].tolist(), task)
        print(f"    Val R²={m['R2']:.4f} MAE={m['MAE']:.4f}")
        assert not np.isnan(m["R2"]), f"Smoke test FAILED for {task}"
        assert m["R2"] > 0.5, f"Smoke test suspiciously low R² for {task}: {m['R2']}"
        print(f"    ✓ Smoke test passed for {task}")


# ═══════════════════════════════════════════════════════════════════════
# 4. Experiment 1: Random IID
# ═══════════════════════════════════════════════════════════════════════

def run_ex1_random_iid(base_dfs: Dict[str, pd.DataFrame]):
    """Step F+G: Random IID baseline across 5 seeds. Fully retrains each time."""
    print("\n" + "=" * 60)
    print("Experiment 1: Random IID Baseline")
    print("=" * 60)

    per_seed_rows = []
    for task in TASK_CONFIG:
        df = base_dfs[task]
        smiles_list = df["canonical_smiles"].tolist()
        ring_list   = df["ring_atoms_list"].tolist()
        values      = df[f"{task}_value"].tolist()

        print(f"\n  [{task}] Base: {len(df)} records, {df['canonical_smiles'].nunique()} molecules")

        for seed in EVAL_SEEDS:
            rng = np.random.RandomState(seed)
            unique_smi = df["canonical_smiles"].unique()
            n_total = len(unique_smi)
            idx_perm = rng.permutation(n_total)
            n_test = int(0.2 * n_total)

            test_smis = set(unique_smi[idx_perm[:n_test]])
            train_val_smis = set(unique_smi[idx_perm[n_test:]])

            te_mask = df["canonical_smiles"].isin(test_smis)
            tv_mask = df["canonical_smiles"].isin(train_val_smis)

            te_idx = df.index[te_mask].tolist()
            tv_idx = df.index[tv_mask].tolist()

            # Split train/val from train_val
            tv_unique = df.loc[tv_mask, "canonical_smiles"].unique()
            n_tv = len(tv_unique)
            tv_perm = rng.permutation(n_tv)
            n_val = int(0.125 * n_tv)  # 10% of 80%=10% total
            va_smis = set(tv_unique[tv_perm[:n_val]])
            tr_smis = set(tv_unique[tv_perm[n_val:]])

            va_mask = df["canonical_smiles"].isin(va_smis)
            tr_mask = df["canonical_smiles"].isin(tr_smis)
            va_idx = df.index[va_mask].tolist()
            tr_idx = df.index[tr_mask].tolist()

            print(f"\n    Seed {seed}: train={len(tr_idx)} rec ({df.loc[tr_idx,'canonical_smiles'].nunique()} mol) | "
                  f"val={len(va_idx)} rec ({df.loc[va_idx,'canonical_smiles'].nunique()} mol) | "
                  f"test={len(te_idx)} rec ({df.loc[te_idx,'canonical_smiles'].nunique()} mol)")

            model = train_one_task(tr_idx, va_idx, smiles_list, ring_list, values, task, seed)
            m = evaluate(model, te_idx, smiles_list, ring_list, values, task)
            per_seed_rows.append({
                "experiment": "random_iid", "task": task, "seed": seed,
                "n_train_rec": len(tr_idx), "n_val_rec": len(va_idx), "n_test_rec": len(te_idx),
                "n_train_mol": df.loc[tr_idx, "canonical_smiles"].nunique(),
                "n_val_mol": df.loc[va_idx, "canonical_smiles"].nunique(),
                "n_test_mol": df.loc[te_idx, "canonical_smiles"].nunique(),
                **m
            })
            print(f"    → Test R²={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")

    result_df = pd.DataFrame(per_seed_rows)
    result_df.to_csv(OUT_DIR / "06_tables" / "random_iid_per_seed.csv", index=False)

    # Summary
    summary = result_df.groupby(["experiment", "task"]).agg(
        mean_R2=("R2", "mean"), std_R2=("R2", "std"),
        mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
    ).round(4).reset_index()
    summary.to_csv(OUT_DIR / "06_tables" / "random_iid_summary.csv", index=False)

    for _, row in summary.iterrows():
        print(f"\n  [{row['task']}] R²={row['mean_R2']:.4f}±{row['std_R2']:.4f}  "
              f"MAE={row['mean_MAE']:.4f}±{row['std_MAE']:.4f}  RMSE={row['mean_RMSE']:.4f}±{row['std_RMSE']:.4f}")

    return result_df, summary


# ═══════════════════════════════════════════════════════════════════════
# 5. Experiment 1b: Scaffold OOD
# ═══════════════════════════════════════════════════════════════════════

def run_ex2_scaffold_ood(base_dfs: Dict[str, pd.DataFrame]):
    """Step H: Scaffold OOD — Murcko scaffold split, full retraining each."""
    print("\n" + "=" * 60)
    print("Experiment 2: Scaffold OOD (Murcko)")
    print("=" * 60)

    per_seed_rows = []
    for task in TASK_CONFIG:
        df = base_dfs[task]
        smiles_list = df["canonical_smiles"].tolist()
        ring_list   = df["ring_atoms_list"].tolist()
        values      = df[f"{task}_value"].tolist()

        print(f"\n  [{task}] Base: {len(df)} records, {df['scaffold'].nunique()} scaffolds")

        for seed in EVAL_SEEDS:
            scaff_groups = df.groupby("scaffold").size().reset_index(name="count")
            paired = sorted(zip(scaff_groups["count"], scaff_groups["scaffold"].tolist()), reverse=True)
            total = len(paired)
            n_test_scaff = max(5, int(0.2 * total))

            rng = np.random.RandomState(seed)
            idx_perm = rng.permutation(total)
            test_scafs = set(paired[i][1] for i in idx_perm[:n_test_scaff])
            train_val_scafs = set(paired[i][1] for i in idx_perm[n_test_scaff:])

            te_mask = df["scaffold"].isin(test_scafs)
            tv_mask = df["scaffold"].isin(train_val_scafs)

            te_idx = df.index[te_mask].tolist()
            tv_idx = df.index[tv_mask].tolist()

            # Split train/val from train_val
            tv_unique_smi = df.loc[tv_mask, "canonical_smiles"].unique()
            rng2 = np.random.RandomState(seed + 100)
            tv_perm = rng2.permutation(len(tv_unique_smi))
            n_val = max(20, int(0.125 * len(tv_unique_smi)))
            va_smi_set = set(tv_unique_smi[tv_perm[:n_val]])
            tr_smi_set = set(tv_unique_smi[tv_perm[n_val:]])

            va_mask = df["canonical_smiles"].isin(va_smi_set)
            tr_mask = df["canonical_smiles"].isin(tr_smi_set)
            va_idx = df.index[va_mask].tolist()
            tr_idx = df.index[tr_mask].tolist()

            # Invariant check
            test_smis_in_tr = set(df.loc[tr_mask, "canonical_smiles"]) & set(df.loc[te_mask, "canonical_smiles"])
            test_scafs_in_tr = set(df.loc[tr_mask, "scaffold"]) & set(test_scafs)
            assert len(test_smis_in_tr) == 0, f"Molecule leakage! {len(test_smis_in_tr)} overlap"
            assert len(test_scafs_in_tr) == 0, f"Scaffold leakage!"

            n_test_mol = df.loc[te_idx, "canonical_smiles"].nunique()
            n_test_sf = df.loc[te_idx, "scaffold"].nunique()
            print(f"\n    Seed {seed}: train={len(tr_idx)} rec | val={len(va_idx)} | test={len(te_idx)} rec "
                  f"({n_test_mol} mol, {n_test_sf} scaffolds)")
            print(f"    Scaffold split invariant OK (no overlap)")

            model = train_one_task(tr_idx, va_idx, smiles_list, ring_list, values, task, seed)
            m = evaluate(model, te_idx, smiles_list, ring_list, values, task)
            per_seed_rows.append({
                "experiment": "scaffold_ood", "task": task, "seed": seed,
                "n_train_rec": len(tr_idx), "n_val_rec": len(va_idx), "n_test_rec": len(te_idx),
                "n_test_molecules": n_test_mol, "n_test_scaffolds": n_test_sf,
                **m
            })
            print(f"    → Test R²={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")

    result_df = pd.DataFrame(per_seed_rows)
    result_df.to_csv(OUT_DIR / "06_tables" / "scaffold_ood_per_seed.csv", index=False)

    summary = result_df.groupby(["experiment", "task"]).agg(
        mean_R2=("R2", "mean"), std_R2=("R2", "std"),
        mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
    ).round(4).reset_index()
    summary.to_csv(OUT_DIR / "06_tables" / "scaffold_ood_summary.csv", index=False)

    for _, row in summary.iterrows():
        print(f"\n  [{row['task']}] R²={row['mean_R2']:.4f}±{row['std_R2']:.4f}  "
              f"MAE={row['mean_MAE']:.4f}±{row['std_MAE']:.4f}")

    return result_df, summary


# ═══════════════════════════════════════════════════════════════════════
# 6. Experiment 2: Leave-One-Ring-Family-Out
# ═══════════════════════════════════════════════════════════════════════

def run_ex3_ring_family_loo(base_dfs, lunci10_df):
    """Step I: LORFO — held-out ring family completely absent from training."""
    print("\n" + "=" * 60)
    print("Experiment 3: Leave-One-Ring-Family-Out")
    print("=" * 60)

    # Use lunci10 as the source of ring families (rich metadata)
    fam_stats = lunci10_df["ring_family"].value_counts()
    include_fams = fam_stats[fam_stats >= 10].index.tolist()  # min 10 samples
    if QUICK_MODE:
        include_fams = fam_stats.head(5).index.tolist()   # top-5 by size for quick verification

    print(f"  Total families: {fam_stats.nunique()}, keeping ≥10 samples: {len(include_fams)}")
    print(f"  Families kept: {sorted(include_fams)[:20]}...")

    loo_rows = []
    for target_fam in sorted(include_fams):
        fam_data = lunci10_df[lunci10_df["ring_family"] == target_fam]
        n_test_mol = fam_data["canonical_smiles"].nunique()
        n_test_rec = len(fam_data)

        if n_test_rec < 10:
            continue

        # All other families (excluding test family AND excluding its molecules)
        other_mask = ~lunci10_df["ring_family"].isin([target_fam])
        other_data = lunci10_df[other_mask]

        print(f"\n  Family '{target_fam}': test={n_test_rec} rec/{n_test_mol} mol, "
              f"base={len(other_data)} rec")

        for task in TASK_CONFIG:
            # Use OTHER data only for training (not base collet_)
            # But we want more training data — also add base collet_ minus any overlap
            base_task = base_dfs[task]

            # Combine other lunci10 records (excluding target family) with base data
            # Actually, per protocol: training should contain everything EXCEPT target family members
            # Let's use base collet_ + other lunci10 records
            combined_train = pd.concat([base_task, other_data], ignore_index=True)
            combined_train = combined_train.drop_duplicates(subset=["canonical_smiles", "ring_atoms_input"],
                                                            keep="first")

            train_smis = combined_train["canonical_smiles"].unique()
            test_smis  = fam_data["canonical_smiles"].unique()
            overlap = set(train_smis) & set(test_smis)
            if overlap:
                print(f"    WARNING: {len(overlap)} molecule overlap removed")
                # Remove overlapping molecules from combined_train
                combined_train = combined_train[~combined_train["canonical_smiles"].isin(test_smis)]

            # Now split combined_train into train/val
            cm = combined_train["canonical_smiles"]
            all_smi = cm.unique()
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(all_smi))
            n_val = max(20, int(0.125 * len(all_smi)))
            va_smi_set = set(all_smi[perm[:n_val]])
            tr_smi_set = set(all_smi[perm[n_val:]])

            # Build positional indices into combined_train
            ct_smiles = combined_train["canonical_smiles"]
            tr_idx = [i for i in range(len(ct_smiles)) if ct_smiles.iloc[i] in tr_smi_set]
            va_idx = [i for i in range(len(ct_smiles)) if ct_smiles.iloc[i] in va_smi_set]

            # Test data appended after combined_train in the concatenated list
            smiles_all = combined_train["canonical_smiles"].tolist() + fam_data["canonical_smiles"].tolist()
            rings_all  = combined_train["ring_atoms_list"].tolist() + fam_data["ring_atoms_list"].tolist()
            vals_all   = combined_train[f"{task}_value"].tolist() + fam_data[f"{task}_value"].tolist()
            te_start = len(combined_train)
            te_idx = list(range(te_start, te_start + n_test_rec))

            model = train_one_task(tr_idx, va_idx, smiles_all, rings_all, vals_all, task, seed=42)
            m = evaluate(model, te_idx, smiles_all, rings_all, vals_all, task)
            loo_rows.append({
                "ring_family": target_fam, "task": task,
                "n_test_records": n_test_rec, "n_test_molecules": n_test_mol,
                **m
            })
            print(f"    {task}: R²={m['R2']:.4f} MAE={m['MAE']:.4f}")

    # Save raw
    pd.DataFrame(loo_rows).to_csv(OUT_DIR / "06_tables" / "ring_family_loo_r2.csv", index=False)
    print(f"\nSaved ring_family_loo_r2.csv ({len(loo_rows)} entries)")
    return pd.DataFrame(loo_rows)


# ═══════════════════════════════════════════════════════════════════════
# 7. Experiment 3: Lunci10 Domain Exposure
# ═══════════════════════════════════════════════════════════════════════

def run_ex4_lunci10_exposure(base_dfs, lunci10_df):
    """Step J+K: Lunci10 domain adaptation curve with nested sampling + real retraining."""
    print("\n" + "=" * 60)
    print("Experiment 4: Lunci10 Domain Adaptation Curve")
    print("=" * 60)

    # Step J: Prepare locked external test
    overlap_count = 58  # From audit output: exact SMILES overlap between base and lunci10

    # Molecule-level split of lunci10
    rng = np.random.RandomState(42)
    unique_smis = lunci10_df["canonical_smiles"].unique()
    perm = rng.permutation(len(unique_smis))
    n_locked = max(50, int(0.2 * len(unique_smis)))

    locked_smis = set(unique_smis[perm[:n_locked]])
    adapt_smis  = set(unique_smis[perm[n_locked:]])

    locked_df = lunci10_df[lunci10_df["canonical_smiles"].isin(locked_smis)].copy()
    adapt_pool_df = lunci10_df[lunci10_df["canonical_smiles"].isin(adapt_smis)].copy()

    # Save locked test IDs
    locked_df[["canonical_smiles"]].to_csv(
        OUT_DIR / "04_lunci10_adaptation" / "locked_test_ids.csv", index=False)

    print(f"  Locked test: {len(locked_df)} records, {locked_df['canonical_smiles'].nunique()} molecules")
    print(f"  Adaptation pool: {len(adapt_pool_df)} records, {adapt_pool_df['canonical_smiles'].nunique()} molecules")

    frac_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    exposure_rows = []

    for frac in frac_levels:
        if frac == 0.0:
            n_adapt = 0
        else:
            n_adapt_mol = max(1, int(round(frac * len(adapt_smis))))
            # Nested sampling: pick first N molecules from permutation
            # Use same permutation but different cutoffs per repeat
            pass  # Handled in repeat loop below

        print(f"\n  Fraction {frac:.0%}:")
        for rep_idx, seed in enumerate(REPEAT_SEEDS):
            rng_exp = np.random.RandomState(seed)

            if frac == 0.0:
                adapt_subset_smis = set()
            else:
                pool_smis_list = list(adapt_smis)
                exp_perm = rng_exp.permutation(len(pool_smis_list))
                n_pick = max(1, int(round(frac * len(pool_smis_list))))
                adapt_subset_smis = set(exp_perm[:n_pick])
                # Map back to SMILES
                adapt_subset_smis = {pool_smis_list[i] for i in exp_perm[:n_pick]}

            # Combined training: base datasets + adapted lunci10 subset
            for task in TASK_CONFIG:
                base_task = base_dfs[task]
                adapt_df = adapt_pool_df[adapt_pool_df["canonical_smiles"].isin(adapt_subset_smis)]

                # Remove any exact overlaps
                valid_adapt = adapt_df[~adapt_df["canonical_smiles"].isin(locked_smis)]
                # Also remove from base if overlap exists (58 molecules)
                valid_base = base_task[~base_task["canonical_smiles"].isin(locked_smis)]

                # For exposure curve: combine base + adapted subset
                # Note: base and adapted may have overlapping molecules — dedup by (smiles, ring_atoms)
                combined = pd.concat([valid_base, valid_adapt], ignore_index=True)
                combined = combined.drop_duplicates(subset=["canonical_smiles", "ring_atoms_input"], keep="first")

                # Train/val split from combined
                cm = combined["canonical_smiles"]
                all_smi = cm.unique()
                tr_perm = rng_exp.permutation(len(all_smi))
                n_val = max(20, int(0.125 * len(all_smi)))
                va_smi = set(all_smi[tr_perm[:n_val]])
                tr_smi = set(all_smi[tr_perm[n_val:]])

                tr_idx = combined.index[combined["canonical_smiles"].isin(tr_smi)].tolist()
                va_idx = combined.index[combined["canonical_smiles"].isin(va_smi)].tolist()

                smiles_all = combined["canonical_smiles"].tolist()
                rings_all  = combined["ring_atoms_list"].tolist()
                vals_all   = combined[f"{task}_value"].tolist()

                print(f"    Rep {rep_idx+1}: train={len(tr_idx)} rec | val={len(va_idx)} | "
                      f"adapt_mols={len(adapt_subset_smis)} | base_mols={valid_base['canonical_smiles'].nunique()}")

                model = train_one_task(tr_idx, va_idx, smiles_all, rings_all, vals_all, task, seed=seed)
                m = evaluate(model, list(range(len(locked_df))),
                             locked_df["canonical_smiles"].tolist(),
                             locked_df["ring_atoms_list"].tolist(),
                             locked_df[f"{task}_value"].tolist(), task)

                exposure_rows.append({
                    "exposure_frac": frac, "repeat": rep_idx + 1, "seed": seed,
                    "n_adapt_molecules": len(adapt_subset_smis),
                    "n_adapt_records": len(valid_adapt),
                    "n_base_molecules": valid_base["canonical_smiles"].nunique(),
                    "n_total_train": len(tr_idx),
                    "task": task,
                    "n_test_records": len(locked_df),
                    "n_test_molecules": locked_df["canonical_smiles"].nunique(),
                    **m
                })
                print(f"      → locked_test R²={m['R2']:.4f} MAE={m['MAE']:.4f}")

    result_df = pd.DataFrame(exposure_rows)
    result_df.to_csv(OUT_DIR / "06_tables" / "lunci10_exposure_raw.csv", index=False)

    # Summary
    summary = result_df.groupby(["exposure_frac", "task"]).agg(
        mean_R2=("R2", "mean"), std_R2=("R2", "std"),
        mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
        n_repeat=("repeat", "count"),
    ).round(4).reset_index()
    summary.to_csv(OUT_DIR / "06_tables" / "lunci10_exposure_summary.csv", index=False)

    for _, row in summary.iterrows():
        sig = f"±{row['std_R2']:.4f}" if not np.isnan(row.get("std_R2", float("nan"))) else ""
        print(f"  frac={row['exposure_frac']:.0%} {row['task']:12s} R²={row['mean_R2']:.4f}{sig} "
              f"MAE={row['mean_MAE']:.4f}±{row.get('std_MAE',0):.4f} (n={int(row['n_repeat'])})")

    return result_df, summary


# ═══════════════════════════════════════════════════════════════════════
# 8. Figures
# ═══════════════════════════════════════════════════════════════════════

def generate_figures(random_sum, scaffold_sum, lorfo_df, exposure_sum):
    """Step L: Generate publication-quality figures."""
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SHORT = {"HOMA": "HOMA", "NICS_ZZ": "NICS ZZ", "MBCO": "MBCO"}

    # ── Fig.4a: Random vs Scaffold bar plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    w = 0.35

    rnd_means = [random_sum[random_sum["task"]==t]["mean_R2"].values[0] for t in TASK_CONFIG]
    rnd_stds  = [random_sum[random_sum["task"]==t]["std_R2"].values[0] for t in TASK_CONFIG]
    scf_means = [scaffold_sum[scaffold_sum["task"]==t]["mean_R2"].values[0] for t in TASK_CONFIG]
    scf_stds  = [scaffold_sum[scaffold_sum["task"]==t]["std_R2"].values[0] for t in TASK_CONFIG]

    ax.bar(x-w/2, rnd_means, w, yerr=rnd_stds, label="Random IID", color="#3498db", capsize=3, alpha=0.85)
    ax.bar(x+w/2, scf_means, w, yerr=scf_stds, label="Scaffold OOD", color="#e74c3c", capsize=3, alpha=0.85)

    for i in range(3):
        delta = rnd_means[i] - scf_means[i]
        pct = abs(delta)/rnd_means[i]*100 if rnd_means[i]!=0 else 0
        ax.annotate(f"↓{pct:.0f}%", xy=(x[i]+w/2, scf_means[i]),
                    xytext=(0,-15), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TASK_CONFIG])
    ax.set_ylabel("R² on Held-Out Test")
    ax.set_title("Fig.4a. Random IID vs Scaffold OOD Generalization\n(Retrained from scratch, mean ± std over 5 seeds)")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(min(0, min(min(rnd_means), min(scf_means))-0.1), 1.1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_figures" / "fig4a_random_vs_scaffold.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "05_figures" / "fig4a_random_vs_scaffold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4a_random_vs_scaffold.{png,pdf}")

    # ── Fig.4b: LORFO Heatmap ──
    if lorfo_df is not None and len(lorfo_df) > 0:
        pivot = lorfo_df.pivot_table(index="ring_family", columns="task", values="R2", aggfunc="mean")
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        tasks_with_n = {}
        for idx, task in enumerate(TASK_CONFIG):
            ax = axes[idx]
            task_data = lorfo_df[lorfo_df["task"] == task]
            if len(task_data) == 0:
                continue
            # Add sample size annotation
            for _, row in task_data.iterrows():
                ax.scatter(row["n_test_records"], row["R2"], s=row["n_test_records"],
                           alpha=0.6, c="#3498db", edgecolors="black", linewidth=0.3)
            if len(task_data) > 2:
                z = np.polyfit(task_data["n_test_records"], task_data["R2"], 1)
                p = np.poly1d(z)
                ax.plot(sorted(task_data["n_test_records"]),
                        p(sorted(task_data["n_test_records"])), "--r", alpha=0.4)
                corr = np.corrcoef(task_data["n_test_records"], task_data["R2"])[0, 1]
                ax.text(0.05, 0.95, f"ρ={corr:.2f}\nn={len(task_data)} families",
                        transform=ax.transAxes, fontsize=9, va="top")
            ax.set_xlabel("n_test samples")
            ax.set_ylabel("R²")
            ax.set_title(f"Fig.4b. LORFO: {SHORT[task]} performance\nvs sample size")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle("Leave-One-Ring-Family-Out: Predictive Performance vs Sample Size",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_figures" / "fig4b_lorfo_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig4b_lorfo_scatter.png")

    # ── Fig.4c: Lunci10 Exposure Curve ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"HOMA": "#e74c3c", "NICS_ZZ": "#3498db", "MBCO": "#2ecc71"}

    for task in TASK_CONFIG:
        tdf = exposure_sum[exposure_sum["task"]==task].sort_values("exposure_frac")
        means = tdf["mean_R2"].values
        stds  = np.array([tdf["std_R2"].iloc[i] if not np.isnan(tdf["std_R2"].iloc[i]) else 0
                         for i in range(len(tdf))])
        ax.errorbar(tdf["exposure_frac"], means, yerr=stds,
                     c=colors[task], marker="o", linestyle="-", linewidth=2,
                     markersize=8, label=f"{SHORT[task]}", capsize=3)

    ax.set_xlabel("Lunci10 Adaptation Data (%)", fontsize=11)
    ax.set_ylabel("R² on Locked External Test", fontsize=11)
    ax.set_title("Fig.4c. Lunci10 Domain Adaptation Curve\n(Retrained at each exposure level, mean ± std over 5 repeats)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_figures" / "fig4c_exposure_curve.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "05_figures" / "fig4c_exposure_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4c_exposure_curve.{png,pdf}")

    return True


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_time = time.time()

    # Load data
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    base_dfs = {task: load_base_dataset(task) for task in TASK_CONFIG}
    lunci10_df = load_lunci10_dataset()

    # Step E: Smoke test (optional — set SMOKE_ONLY=True to only run smoke)
    SMOKE_ONLY = False
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        SMOKE_ONLY = True

    if SMOKE_ONLY:
        smoke_test()
    else:
        # Step E: Quick smoke — skip via env FIG4_SKIP_SMOKE=1 once already validated
        if os.environ.get("FIG4_SKIP_SMOKE") != "1":
            print("\nRunning quick smoke test...")
            smoke_test()

        # Steps F+G: Random IID
        random_per_seed, random_summary = run_ex1_random_iid(base_dfs)

        # Step H: Scaffold OOD
        scaffold_per_seed, scaffold_summary = run_ex2_scaffold_ood(base_dfs)

        # Step I: LORFO
        lorfo_df = run_ex3_ring_family_loo(base_dfs, lunci10_df)

        # Steps J+K: Lunci10 exposure
        exposure_per_seed, exposure_summary = run_ex4_lunci10_exposure(base_dfs, lunci10_df)

        # Step L: Figures
        generate_figures(random_summary, scaffold_summary, lorfo_df, exposure_summary)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Done! Total runtime: {elapsed:.1f}s")
    print(f"Output directory: {OUT_DIR}")
