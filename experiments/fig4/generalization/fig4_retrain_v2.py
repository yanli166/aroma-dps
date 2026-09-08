#!/usr/bin/env python3
"""Fig.4 Full Retraining Pipeline v2 — 图缓存 + 张量切片训练引擎。

严格遵循 fig4.txt:
  1. Random IID (molecule-level, 5 seeds)  vs Scaffold OOD (Murcko, 5 seeds)
  2. Leave-One-Ring-Family-Out (基于 target-ring family taxonomy)
  3. Lunci10 domain exposure curve (nested 0-100%, 5 repeats)

关键架构(性能修复):
  - build_graph 只在数据预处理阶段调用一次 (含 RDKit 3D bounds, ~0.1s/mol)
  - 图缓存以 npz 落盘, 之后按 name+rf 复用
  - 训练/评估全程只用 numpy 切片 -> torch tensor, 不再触碰 RDKit
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
AROMA_PROJ = Path("/home/ubuntu/aroma-dps-code")
BASE_OUT_DIR = AROMA_PROJ / "0901-end-code" / "results" / "fig4_generalization_retrain_v4"
OUT_DIR = BASE_OUT_DIR
DIR_LIST = ["00_audit","01_random_iid","02_scaffold_ood","03_ring_family_loo",
            "04_lunci10_adaptation","05_figures","06_tables","checkpoints","logs","cache"]

DATA_DIR = AROMA_PROJ / "code_end" / "data1_end"
BEST_PKG = AROMA_PROJ / "best_model_package"
CACHE_DIR = None  # resolved after parse_args (depends on --noe)

sys.path.insert(0, str(BEST_PKG))
from graph_utils import build_graph
from model_arch import RingConditionedMPNN

NODE_VEC_LEN = 60
MAX_ATOMS_CFG = 85

# ── Tasks ────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "HOMA":    {"task_col": "homa_value", "rf_val": 10, "col_idx": 60-15},
    "NICS_ZZ": {"task_col": "NICS_value", "rf_val": 1,  "col_idx": 60-15},
    "MBCO":    {"task_col": "mbco_value", "rf_val": 1,  "col_idx": 60-15},
}
TRAIN_PARAMS = dict(hidden_dim=128, n_conv=3, n_hidden=2, p_dropout=0.2,
                    lr=0.001, weight_decay=1e-5, batch_size=64,
                    n_epochs=200, patience=30)
CHUNK = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed / top-5 fams / 1 repeat")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--noe", action="store_true",
                    help="drop the contaminated 'e*' New_ID block from base datasets")
    ap.add_argument("--only-exp", default="",
                    help="comma list of experiments to run: random,scaffold,lorfo,exposure "
                         "(default: run all)")
    return ap.parse_args()


ARGS = parse_args()
QUICK = ARGS.quick
EVAL_SEEDS = [100] if QUICK else [100, 101, 102, 103, 104]
REPEAT_SEEDS = [2026] if QUICK else [2026, 2027, 2028, 2029, 2030]

# ---- resolve output dirs (noe variant keeps results fully separated) ----
if ARGS.noe:
    OUT_DIR = AROMA_PROJ / "0901-end-code" / "results" / "fig4_generalization_retrain_v4_noe"
CACHE_DIR = OUT_DIR / "cache"
for _d in DIR_LIST:
    (OUT_DIR / _d).mkdir(parents=True, exist_ok=True)
if ARGS.noe:
    print(f"[fig4_retrain_v2] --noe mode: base 'e*' block dropped; OUT_DIR={OUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# 1. SMILES / ring / scaffold helpers
# ═══════════════════════════════════════════════════════════════════════
def safe_canonical(smi):
    from rdkit import Chem
    if not smi or not isinstance(smi, str): return ""
    mol = Chem.MolFromSmiles(smi.strip())
    return Chem.MolToSmiles(mol) if mol else ""


def murcko_scaffold(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc else ""
    except Exception:
        return ""


def parse_ring_atoms(raw) -> List[int]:
    if not isinstance(raw, str) or not raw: return []
    s = raw.replace("[", "").replace("]", "").replace("'", "").replace('"', '').strip()
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            try: out.append(int(p))
            except ValueError: pass
    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. Data loaders
# ═══════════════════════════════════════════════════════════════════════
def load_base_dataset(task: str) -> pd.DataFrame:
    tlower = task.lower().replace("_zz", "")
    path = DATA_DIR / f"collet_{tlower}_0716.csv"
    df = pd.read_csv(path)
    df["canonical_smiles"] = df["smiles"].apply(safe_canonical)
    df["canonical_smiles"] = df["canonical_smiles"].replace("", np.nan)
    df.dropna(subset=["canonical_smiles"], inplace=True)
    df["ring_atoms_list"] = df["atom_on_ring"].apply(parse_ring_atoms)
    df[f"{task}_value"] = pd.to_numeric(df[TASK_CONFIG[task]["task_col"]], errors="coerce")
    df.dropna(subset=[f"{task}_value"], inplace=True)
    # drop rows with empty target ring (unbuildable)
    df = df[df["ring_atoms_list"].apply(len) >= 3]
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    # --noe: drop the contaminated 'e*' New_ID block (wrong labels vs lunci10 truth)
    if ARGS.noe:
        n_before = len(df)
        df = df[~df["New_ID"].astype(str).str.lower().str.startswith("e")]
        print(f"  [load_base] {task}: --noe dropped {n_before - len(df)} 'e*' rows "
              f"({n_before} -> {len(df)})")
    df = df.reset_index(drop=True)
    return df


def load_lunci10_dataset() -> pd.DataFrame:
    path = AROMA_PROJ / "lunci10" / "lunci10_unified.csv"
    df = pd.read_csv(path)
    df["canonical_smiles"] = df["smiles"].apply(safe_canonical)
    df["canonical_smiles"] = df["canonical_smiles"].replace("", np.nan)
    df.dropna(subset=["canonical_smiles"], inplace=True)
    df["ring_atoms_list"] = df["ring_atoms"].apply(parse_ring_atoms)
    for t in ["HOMA", "NICS_ZZ", "MBCO"]:
        df[f"{t}_value"] = pd.to_numeric(df[t], errors="coerce")
    df.dropna(subset=["HOMA_value", "NICS_ZZ_value", "MBCO_value"], inplace=True)
    df = df[df["ring_atoms_list"].apply(len) >= 3]
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    df["ring_family"] = df["ring_name"].fillna("unknown") if "ring_name" in df.columns else "unknown"
    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Graph cache (build once -> npz)
# ═══════════════════════════════════════════════════════════════════════
def _build_one(args):
    smi, ring = args
    g = build_graph(smi, ring, NODE_VEC_LEN, MAX_ATOMS_CFG, ring_flag_value=1)
    return g["node_mat"], g["adj_mat"], g["ring_indices"]


def build_and_cache(df: pd.DataFrame, name: str) -> Dict[str, np.ndarray]:
    """Build graphs for all rows in df once, save npz. Return dict of arrays.
    rf=1 base feature; rf=10 variant derivable by scaling column col_idx."""
    npz_path = CACHE_DIR / f"{name}.npz"
    meta_path = CACHE_DIR / f"{name}_meta.json"
    if npz_path.exists() and not ARGS.skip_build:
        print(f"  [cache] load {npz_path.name}")
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}

    smis = df["canonical_smiles"].tolist()
    rings = df["ring_atoms_list"].tolist()
    pairs = list(zip(smis, rings))

    print(f"  [build] {name}: building {len(pairs)} graphs ...")
    t0 = time.time()
    import multiprocessing as mp
    n_proc = min(8, mp.cpu_count())
    try:
        with mp.Pool(n_proc) as pool:
            results = pool.map(_build_one, pairs, chunksize=64)
    except Exception:
        results = [_build_one(p) for p in pairs]

    node = np.stack([r[0] for r in results]).astype(np.float32)
    adj = np.stack([r[1] for r in results]).astype(np.float32)
    ring = np.stack([r[2] for r in results]).astype(np.int64)
    dt = time.time() - t0
    print(f"  [build] {name}: done {len(pairs)} graphs in {dt:.1f}s")

    np.savez_compressed(npz_path, node_mats=node, adj_mats=adj, ring_indices=ring)
    return {"node_mats": node, "adj_mats": adj, "ring_indices": ring}


def derive_rf10(node_mats_rf1: np.ndarray, col_idx: int = 45) -> np.ndarray:
    """HOMA 用 rf=10: 复制并放大标记列 (60-15=45) *10。"""
    out = node_mats_rf1.copy()
    out[..., col_idx] = out[..., col_idx] * 10.0
    return out


# ═══════════════════════════════════════════════════════════════════════
# 4. Training engine (tensor slicing only, no RDKit in loop)
# ═══════════════════════════════════════════════════════════════════════
def _init_model(task: str) -> RingConditionedMPNN:
    cfg = TASK_CONFIG[task]
    return RingConditionedMPNN(
        node_vec_len=NODE_VEC_LEN, hidden_dim=TRAIN_PARAMS["hidden_dim"],
        n_conv=TRAIN_PARAMS["n_conv"], n_hidden=TRAIN_PARAMS["n_hidden"],
        p_dropout=TRAIN_PARAMS["p_dropout"],
        use_projection=(task == "HOMA"), ring_flag_value=cfg["rf_val"],
    ).to(device)


def train_model(node_mats, adj_mats, ring_idx, values,
                tr_idx, va_idx, task: str, seed: int,
                n_epochs: Optional[int] = None,
                patience: Optional[int] = None):
    """node_mats/adj_mats/ring_idx: np arrays (N, ...). values: np array (N,)."""
    model = _init_model(task)
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS["lr"],
                           weight_decay=TRAIN_PARAMS["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5,
        patience=(patience or TRAIN_PARAMS["patience"]) // 3, min_lr=1e-6)

    # move arrays to GPU once
    node_t = torch.from_numpy(node_mats).to(device)
    adj_t = torch.from_numpy(adj_mats).to(device)
    ring_t = torch.from_numpy(ring_idx).to(device)
    val_t = torch.from_numpy(values).to(device).reshape(-1)

    tr_arr = np.array(tr_idx)
    va_arr = np.array(va_idx)
    bs = TRAIN_PARAMS["batch_size"]
    n_ep = n_epochs or TRAIN_PARAMS["n_epochs"]
    pat = patience or TRAIN_PARAMS["patience"]

    best_mae = float("inf")
    best_state = None
    wait = 0
    loss_fn = nn.MSELoss()

    for ep in range(1, n_ep + 1):
        model.train()
        perm = tr_arr[rng.permutation(len(tr_arr))]
        ep_loss = 0.0
        for i in range(0, len(perm), bs):
            bi = perm[i:i+bs]
            if len(bi) < 2:      # BatchNorm1d requires >1 sample in training
                break
            nb, ab, rb = node_t[bi], adj_t[bi], ring_t[bi]
            yb = val_t[bi]
            pred = model(nb, ab, rb).reshape(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(bi)

        # validate (chunked on GPU in slices)
        model.eval()
        vmae_sum, vcnt = 0.0, 0
        for s in range(0, len(va_arr), CHUNK):
            vi = va_arr[s:s+CHUNK]
            nb, ab, rb = node_t[vi], adj_t[vi], ring_t[vi]
            yb = val_t[vi]
            with torch.no_grad():
                p = model(nb, ab, rb).reshape(-1)
            vmae_sum += (p - yb).abs().sum().item()
            vcnt += len(vi)
        vmae = vmae_sum / max(vcnt, 1)
        sched.step(vmae)

        if vmae < best_mae:
            best_mae = vmae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep % 40 == 0:
            print(f"    ep{ep:3d} tr={ep_loss/max(len(tr_arr),1):.4f} valMAE={vmae:.4f} wait={wait}")
        if wait >= pat:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_mae


def predict_model(model, node_mats, adj_mats, ring_idx, idx_list) -> Tuple[np.ndarray, np.ndarray]:
    node_t = torch.from_numpy(node_mats).to(device)
    adj_t = torch.from_numpy(adj_mats).to(device)
    ring_t = torch.from_numpy(ring_idx).to(device)
    arr = np.array(idx_list)
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(arr), CHUNK):
            bi = arr[s:s+CHUNK]
            p = model(node_t[bi], adj_t[bi], ring_t[bi]).cpu().numpy().reshape(-1)
            preds.append(p)
    return np.concatenate(preds)


def metrics(y_true, y_pred) -> Dict:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[m], y_pred[m]
    from sklearn.metrics import r2_score, mean_absolute_error
    n = int(m.sum())
    if n < 2:
        return {"n": n, "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    return {
        "n": n,
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Cache stores: 每个数据集一张缓存 (rf=1 底版, HOMA 用 rf10 派生)
# ═══════════════════════════════════════════════════════════════════════
def value_array(df: pd.DataFrame, task: str) -> np.ndarray:
    return df[f"{task}_value"].to_numpy(dtype=np.float32)


def split_mol_train_val(test_mol_ids, rng_seed, df, va_ratio=0.125):
    """Split non-test molecules into train/val by molecule (canonical smiles)."""
    others = [m for m in df["canonical_smiles"].unique() if m not in test_mol_ids]
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(len(others))
    n_va = max(20, int(va_ratio * len(others)))
    va_ids = set(np.array(others)[perm[:n_va]].tolist())
    tr_ids = set(np.array(others)[perm[n_va:]].tolist())
    return tr_ids, va_ids


def concat_arrays(parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    """parts: [(node, adj, ring, vals)] -> concatenated arrays."""
    nodes = np.concatenate([p[0] for p in parts], axis=0)
    adjs = np.concatenate([p[1] for p in parts], axis=0)
    rings = np.concatenate([p[2] for p in parts], axis=0)
    vals = np.concatenate([p[3] for p in parts], axis=0)
    return nodes, adjs, rings, vals


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1a: Random IID (molecule-level)
# ═══════════════════════════════════════════════════════════════════════
def exp1_random_iid(dfs, caches):
    print("\n" + "=" * 66)
    print("EXP1a  Random IID  (molecule-level, 5 seeds)")
    print("=" * 66)
    rows = []
    for task in TASK_CONFIG:
        df = dfs[task]
        cache = caches[task]   # {node, adj, ring}
        values = value_array(df, task)
        node, adj, ring = cache["node_mats"], cache["adj_mats"], cache["ring_indices"]

        # HOMA needs rf=10 node variant
        if TASK_CONFIG[task]["rf_val"] == 10:
            node = derive_rf10(node)

        mol_ids = df["canonical_smiles"].unique()
        for seed in EVAL_SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(mol_ids))
            n_test = int(0.2 * len(mol_ids))
            test_ids = set(mol_ids[perm[:n_test]].tolist())
            tr_ids, va_ids = split_mol_train_val(test_ids, seed + 1000, df)

            te_idx = df.index[df["canonical_smiles"].isin(test_ids)].tolist()
            tr_idx = df.index[df["canonical_smiles"].isin(tr_ids)].tolist()
            va_idx = df.index[df["canonical_smiles"].isin(va_ids)].tolist()
            print(f"  [{task}] seed{seed}: tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)}")

            model, bmae = train_model(node, adj, ring, values, tr_idx, va_idx, task, seed)
            yp = predict_model(model, node, adj, ring, te_idx)
            m = metrics(values[te_idx], yp)
            rows.append({"split": "random", "task": task, "seed": seed,
                         "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
                         "best_val_mae": bmae, **m})
            print(f"      -> R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "06_tables" / "random_iid_per_seed.csv", index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1b: Scaffold OOD (Murcko)
# ═══════════════════════════════════════════════════════════════════════
def exp2_scaffold_ood(dfs, caches):
    print("\n" + "=" * 66)
    print("EXP1b  Scaffold OOD  (Murcko, 5 seeds)")
    print("=" * 66)
    rows = []
    for task in TASK_CONFIG:
        df = dfs[task]
        cache = caches[task]
        values = value_array(df, task)
        node, adj, ring = cache["node_mats"], cache["adj_mats"], cache["ring_indices"]
        if TASK_CONFIG[task]["rf_val"] == 10:
            node = derive_rf10(node)

        scaff_groups = df.groupby("scaffold").size().sort_values(ascending=False)
        scaffs = scaff_groups.index.tolist()
        for seed in EVAL_SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(scaffs))
            n_test_sc = max(5, int(0.2 * len(scaffs)))
            test_scaffs = set(np.array(scaffs)[perm[:n_test_sc]].tolist())

            te_mask = df["scaffold"].isin(test_scaffs)
            te_idx = df.index[te_mask].tolist()
            test_mol_ids = set(df.loc[te_mask, "canonical_smiles"])
            tr_ids, va_ids = split_mol_train_val(test_mol_ids, seed + 1000, df)
            tr_idx = df.index[df["canonical_smiles"].isin(tr_ids)].tolist()
            va_idx = df.index[df["canonical_smiles"].isin(va_ids)].tolist()

            # invariants
            assert not (set(df.loc[tr_idx, "scaffold"]) & test_scaffs), "scaffold leakage!"
            print(f"  [{task}] seed{seed}: tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)} "
                  f"(te_scaffolds={len(test_scaffs)})")
            model, bmae = train_model(node, adj, ring, values, tr_idx, va_idx, task, seed)
            yp = predict_model(model, node, adj, ring, te_idx)
            m = metrics(values[te_idx], yp)
            rows.append({"split": "scaffold_ood", "task": task, "seed": seed,
                         "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
                         "n_test_scaffolds": len(test_scaffs), "best_val_mae": bmae, **m})
            print(f"      -> R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "06_tables" / "scaffold_ood_per_seed.csv", index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2: Leave-One-Ring-Family-Out (target ring family)
# 训练 = base[task](去掉分子泄漏) + lunci10 其他 family; 测试 = lunci10 该 family
# ═══════════════════════════════════════════════════════════════════════
def exp3_lorfo(base_dfs, base_caches, l10_df, l10_cache):
    print("\n" + "=" * 66)
    print("EXP2  Leave-One-Ring-Family-Out")
    print("=" * 66)
    fam_counts = l10_df["ring_family"].value_counts()
    fams = fam_counts.index.tolist()
    if QUICK:
        fams = fams[:5]
    print(f"  ring families: {len(fams)} -> {sorted(fams)}")

    l10_mol_ids = set(l10_df["canonical_smiles"])
    rows = []
    for fam in sorted(fams):
        fam_df = l10_df[l10_df["ring_family"] == fam]
        te_idx_l10 = fam_df.index.tolist()
        n_te_mol = fam_df["canonical_smiles"].nunique()
        print(f"\n  Family '{fam}': n_rec={len(fam_df)} n_mol={n_te_mol}")
        if len(fam_df) < 5:
            print("    too small, skip"); continue
        # others = all l10 minus this family's molecules
        other_l10 = l10_df[~l10_df["canonical_smiles"].isin(set(fam_df["canonical_smiles"]))]

        for task in TASK_CONFIG:
            base_df = base_dfs[task]
            base_cache = base_caches[task]
            base_vals = value_array(base_df, task)
            # base minus molecules appearing in ANY l10 (avoid test leakage + contamination)
            keep = ~base_df["canonical_smiles"].isin(l10_mol_ids)
            b_idx = base_df.index[keep].tolist()
            b_node = base_cache["node_mats"][b_idx]
            b_adj = base_cache["adj_mats"][b_idx]
            b_ring = base_cache["ring_indices"][b_idx]
            b_val = base_vals[b_idx]

            o_idx = other_l10.index.tolist()
            o_node = l10_cache["node_mats"][o_idx]
            o_adj = l10_cache["adj_mats"][o_idx]
            o_ring = l10_cache["ring_indices"][o_idx]
            o_val = other_l10[f"{task}_value"].to_numpy(dtype=np.float32)

            # concat base+other_l10 for training
            node = np.concatenate([b_node, o_node], axis=0)
            adj = np.concatenate([b_adj, o_adj], axis=0)
            ring = np.concatenate([b_ring, o_ring], axis=0)
            vals = np.concatenate([b_val, o_val], axis=0)
            if TASK_CONFIG[task]["rf_val"] == 10:
                node = derive_rf10(node)

            # molecule-level tr/va split over training pool
            pool_mols = np.concatenate([base_df.loc[b_idx, "canonical_smiles"].to_numpy(),
                                        other_l10.loc[o_idx, "canonical_smiles"].to_numpy()])
            rng = np.random.RandomState(42)
            um = np.unique(pool_mols)
            perm = rng.permutation(len(um))
            n_va = max(20, int(0.125 * len(um)))
            va_ids = set(um[perm[:n_va]].tolist())
            tr_idx = np.where(np.isin(pool_mols, um[perm[n_va:]]))[0].tolist()
            va_idx = np.where(np.isin(pool_mols, list(va_ids)))[0].tolist()

            print(f"    [{task}] tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx_l10)}")
            model, bmae = train_model(node, adj, ring, vals, tr_idx, va_idx, task, 42)
            # test: family records in l10
            te_node = l10_cache["node_mats"][te_idx_l10]
            te_adj = l10_cache["adj_mats"][te_idx_l10]
            te_ring = l10_cache["ring_indices"][te_idx_l10]
            te_val = fam_df[f"{task}_value"].to_numpy(dtype=np.float32)
            if TASK_CONFIG[task]["rf_val"] == 10:
                te_node = derive_rf10(te_node)
            yp = predict_model(model, te_node, te_adj, te_ring, list(range(len(te_idx_l10))))
            m = metrics(te_val, yp)
            rows.append({"ring_family": fam, "task": task,
                         "n_test_rec": len(te_idx_l10), "n_test_mol": n_te_mol,
                         "best_val_mae": bmae, **m})
            print(f"      -> R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "06_tables" / "ring_family_loo_r2.csv", index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3: Lunci10 exposure curve
# ═══════════════════════════════════════════════════════════════════════
def exp4_exposure(base_dfs, base_caches, l10_df, l10_cache):
    print("\n" + "=" * 66)
    print("EXP3  Lunci10 exposure curve (nested, locked test)")
    print("=" * 66)

    # locked test: 20% l10 molecules held out forever
    rng_lock = np.random.RandomState(42)
    l10_mols = l10_df["canonical_smiles"].unique()
    perm = rng_lock.permutation(len(l10_mols))
    n_lock = int(0.2 * len(l10_mols))
    locked_mols = set(l10_mols[perm[:n_lock]].tolist())
    pool_mols = l10_mols[perm[n_lock:]]  # ordered pool -> nested exposure

    locked_df = l10_df[l10_df["canonical_smiles"].isin(locked_mols)]
    print(f"  locked test: {len(locked_df)} rec / {len(locked_mols)} mol | "
          f"pool: {len(pool_mols)} mol")
    locked_df[["canonical_smiles"]].to_csv(OUT_DIR / "04_lunci10_adaptation" / "locked_test_ids.csv", index=False)

    lock_idx = locked_df.index.tolist()
    l10_all_mols = set(l10_mols)

    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for frac in levels:
        if frac > 0:
            n_pick = int(round(frac * len(pool_mols)))
        for rep, rseed in enumerate(REPEAT_SEEDS):
            if frac == 0.0:
                exposed = set()
            else:
                rg = np.random.RandomState(rseed)
                order = rg.permutation(len(pool_mols))
                exposed = set(pool_mols[order[:n_pick]].tolist())
            print(f"\n  frac={frac:.0%} rep={rep+1} n_exposed_mol={len(exposed)}")
            for task in TASK_CONFIG:
                base_df = base_dfs[task]
                base_cache = base_caches[task]
                base_vals = value_array(base_df, task)
                # base: drop any molecule overlapping l10 (58) for cleanliness
                keep = ~base_df["canonical_smiles"].isin(l10_all_mols)
                b_idx = base_df.index[keep].tolist()
                b_node = base_cache["node_mats"][b_idx]
                b_adj = base_cache["adj_mats"][b_idx]
                b_ring = base_cache["ring_indices"][b_idx]
                b_val = base_vals[b_idx]

                parts_node, parts_adj, parts_ring, parts_val = [], [], [], []
                if len(b_idx):
                    parts_node.append(b_node); parts_adj.append(b_adj)
                    parts_ring.append(b_ring); parts_val.append(b_val)
                if frac > 0 and len(exposed):
                    exp_df = l10_df[l10_df["canonical_smiles"].isin(exposed)]
                    e_idx = exp_df.index.tolist()
                    parts_node.append(l10_cache["node_mats"][e_idx])
                    parts_adj.append(l10_cache["adj_mats"][e_idx])
                    parts_ring.append(l10_cache["ring_indices"][e_idx])
                    parts_val.append(exp_df[f"{task}_value"].to_numpy(dtype=np.float32))

                node, adj, ring, vals = concat_arrays(
                    list(zip(parts_node, parts_adj, parts_ring, parts_val)))
                if TASK_CONFIG[task]["rf_val"] == 10:
                    node = derive_rf10(node)

                # tr/va split over train pool by molecule
                mol_list = base_df.loc[b_idx, "canonical_smiles"].tolist()
                if frac > 0 and len(exposed):
                    exp_df = l10_df[l10_df["canonical_smiles"].isin(exposed)]
                    mol_list += exp_df["canonical_smiles"].tolist()
                mol_arr = np.array(mol_list)
                um = np.unique(mol_arr)
                rg2 = np.random.RandomState(rseed + 500)
                pm2 = rg2.permutation(len(um))
                n_va = max(20, int(0.125 * len(um)))
                va_ids = set(um[pm2[:n_va]].tolist())
                tr_idx = np.where(np.isin(mol_arr, um[pm2[n_va:]]))[0].tolist()
                va_idx = np.where(np.isin(mol_arr, list(va_ids)))[0].tolist()

                model, bmae = train_model(node, adj, ring, vals, tr_idx, va_idx, task, rseed)
                # locked test predict
                te_node = l10_cache["node_mats"][lock_idx]
                te_adj = l10_cache["adj_mats"][lock_idx]
                te_ring = l10_cache["ring_indices"][lock_idx]
                te_val = locked_df[f"{task}_value"].to_numpy(dtype=np.float32)
                if TASK_CONFIG[task]["rf_val"] == 10:
                    te_node = derive_rf10(te_node)
                yp = predict_model(model, te_node, te_adj, te_ring, list(range(len(lock_idx))))
                m = metrics(te_val, yp)
                rows.append({"exposure_frac": frac, "repeat": rep + 1, "seed": rseed,
                             "n_exposed_mol": len(exposed), "task": task,
                             "n_test_rec": len(lock_idx), "n_test_mol": len(locked_mols),
                             "best_val_mae": bmae, **m})
                print(f"    [{task}] R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "06_tables" / "lunci10_exposure_raw.csv", index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 8. Figures + summaries
# ═══════════════════════════════════════════════════════════════════════
def summarize(per_seed_df, exp_name):
    if per_seed_df is None or len(per_seed_df) == 0:
        return None
    g = per_seed_df.groupby("task").agg(
        mean_R2=("R2", "mean"), std_R2=("R2", "std"),
        mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
    ).round(4)
    g.to_csv(OUT_DIR / "06_tables" / f"{exp_name}_summary.csv")
    return g


def plot_figures(rnd_df, scaf_df, lorfo_df, expo_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SHORT = {"HOMA": "HOMA", "NICS_ZZ": "NICS ZZ", "MBCO": "MBCO"}

    # fig4a: random vs scaffold bar
    if rnd_df is not None and scaf_df is not None and len(rnd_df) and len(scaf_df):
        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(3); w = 0.35
        r_means = [rnd_df.groupby("task")["R2"].mean()[t] for t in TASK_CONFIG]
        r_stds = [rnd_df.groupby("task")["R2"].std()[t] for t in TASK_CONFIG]
        s_means = [scaf_df.groupby("task")["R2"].mean()[t] for t in TASK_CONFIG]
        s_stds = [scaf_df.groupby("task")["R2"].std()[t] for t in TASK_CONFIG]
        ax.bar(x - w/2, r_means, w, yerr=r_stds, label="Random IID", color="#3498db", capsize=4, alpha=0.9)
        ax.bar(x + w/2, s_means, w, yerr=s_stds, label="Scaffold OOD", color="#e74c3c", capsize=4, alpha=0.9)
        for i in range(3):
            drop = r_means[i] - s_means[i]
            ax.text(x[i] + w/2, s_means[i] - 0.03, f"{drop:+.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TASK_CONFIG])
        ax.set_ylabel("R² (held-out test)")
        ax.set_title("Random IID vs Scaffold OOD (retrained from scratch)")
        ax.legend(); ax.grid(alpha=0.25); ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_figures" / "fig4a_random_vs_scaffold.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT_DIR / "05_figures" / "fig4a_random_vs_scaffold.pdf", bbox_inches="tight")
        plt.close(fig)

    # fig4c: lorfo heatmap (family x task) + sample-size scatter
    if lorfo_df is not None and len(lorfo_df):
        piv = lorfo_df.pivot_table(index="ring_family", columns="task", values="R2")
        fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(piv))))
        im = ax.imshow(piv.values.astype(float), cmap="RdYlBu_r", vmin=-2, vmax=1, aspect="auto")
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        for i in range(len(piv)):
            for j in range(len(piv.columns)):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if -2 <= v <= 0.4 else "black")
        plt.colorbar(im, label="R²")
        ax.set_title("Leave-One-Ring-Family-Out R²")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_figures" / "fig4b_lorfo_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # R2 vs n scatter
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, task in zip(axes, TASK_CONFIG):
            td = lorfo_df[lorfo_df["task"] == task].dropna(subset=["R2"])
            ax.scatter(td["n_test_rec"], td["R2"], c="#3498db", s=45, alpha=0.8)
            if len(td) > 2:
                z = np.polyfit(td["n_test_rec"], td["R2"], 1)
                p = np.poly1d(z)
                xs = np.linspace(td["n_test_rec"].min(), td["n_test_rec"].max(), 50)
                ax.plot(xs, p(xs), "--", color="gray")
                from scipy.stats import spearmanr
                rho, _ = spearmanr(td["n_test_rec"], td["R2"])
                ax.set_title(f"{SHORT[task]}  ρ={rho:.2f}")
            ax.set_xlabel("n_test_records"); ax.set_ylabel("R²")
            ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.2)
        fig.suptitle("LORFO R² vs family sample size")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_figures" / "fig4b_lorfo_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # fig4d: exposure curve
    if expo_df is not None and len(expo_df):
        colors = {"HOMA": "#e74c3c", "NICS_ZZ": "#3498db", "MBCO": "#2ecc71"}
        fig, ax = plt.subplots(figsize=(9, 6))
        for task in TASK_CONFIG:
            td = expo_df[expo_df["task"] == task].groupby("exposure_frac")["R2"].agg(["mean", "std"]).reset_index()
            ax.errorbar(td["exposure_frac"], td["mean"], yerr=td["std"], marker="o",
                        capsize=3, color=colors[task], linewidth=2, label=SHORT[task])
        ax.set_xlabel("Lunci10 adaptation fraction")
        ax.set_ylabel("R² on locked external test")
        ax.set_title("Lunci10 exposure curve (retrained, nested)")
        ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(-0.03, 1.03)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_figures" / "fig4c_exposure_curve.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT_DIR / "05_figures" / "fig4c_exposure_curve.pdf", bbox_inches="tight")
        plt.close(fig)
    print("  Figures saved in", OUT_DIR / "05_figures")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t_all = time.time()
    print(f"[fig4_retrain_v2] device={device} quick={QUICK}")
    print("Loading base datasets...")
    base_dfs = {t: load_base_dataset(t) for t in TASK_CONFIG}
    l10_df = load_lunci10_dataset()

    print("\nBuilding graph caches ...")
    base_caches = {}
    for t in TASK_CONFIG:
        base_caches[t] = build_and_cache(base_dfs[t], f"base_{t}")
    l10_cache = build_and_cache(l10_df, "lunci10")
    print(f"  base cache keys: {list(base_caches.keys())}, l10 rows: {l10_cache['node_mats'].shape[0]}")
    for t in TASK_CONFIG:
        if len(base_dfs[t]) != base_caches[t]["node_mats"].shape[0]:
            print(f"  !! cache size mismatch base_{t}: df={len(base_dfs[t])} cache={base_caches[t]['node_mats'].shape[0]}")
    if len(l10_df) != l10_cache["node_mats"].shape[0]:
        print(f"  !! cache size mismatch lunci10: df={len(l10_df)} cache={l10_cache['node_mats'].shape[0]}")

    if ARGS.smoke:
        return

    want = {s.strip() for s in ARGS.only_exp.split(",") if s.strip()}
    rnd = scf = lor = exp = None
    if (not want) or "random" in want:
        rnd = exp1_random_iid(base_dfs, base_caches)
    if (not want) or "scaffold" in want:
        scf = exp2_scaffold_ood(base_dfs, base_caches)
    if (not want) or "lorfo" in want:
        lor = exp3_lorfo(base_dfs, base_caches, l10_df, l10_cache)
    if (not want) or "exposure" in want:
        exp = exp4_exposure(base_dfs, base_caches, l10_df, l10_cache)

    print("\nSummaries:")
    for name, df in [("random", rnd), ("scaffold_ood", scf), ("lorfo", lor), ("exposure", exp)]:
        if df is None or len(df) == 0:
            continue
        print(f"\n=== {name} ===")
        if "exposure_frac" in df.columns:
            grp = df.groupby(["task", "exposure_frac"])["R2"].agg(["mean", "std", "count"]).round(4)
            print(grp.to_string())
        else:
            g = df.groupby("task")["R2"].agg(["mean", "std", "count"]).round(4)
            print(g.to_string())

    summarize(rnd, "random_iid")
    summarize(scf, "scaffold_ood")
    plot_figures(rnd, scf, lor, exp)
    print(f"\n[DONE] total {time.time()-t_all:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
