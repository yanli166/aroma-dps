"""Task 8 (按协议修正): molecule-cluster bootstrap.

要求:
  - 因为 2153 ring-level rows 来自 1389 molecules, 统计 CI / bootstrap 时至少使用
    molecule-cluster bootstrap, 不能默认每个 ring-level row 完全独立
  - 主点估计仍可报告 ring-level MAE, 因为任务本身是 ring-resolved prediction
  - uncertainty 必须考虑 molecule clustering
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = Path("_PROJ_ROOT")
FIG4_ROOT = PROJ_ROOT / "0901-end-code/fig4_lunci10"
sys.path.insert(0, str(FIG4_ROOT))


def ring_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(y_true))}


def molecule_cluster_bootstrap(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    molecule_col: str = "canonical_smiles",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap at molecule level: sample molecules with replacement, then collect
    ALL ring-level rows of each selected molecule. Returns CI for ring-level MAE / RMSE.

    This correctly accounts for within-molecule correlation between ring-level rows.
    """
    rng = np.random.default_rng(seed)
    # precompute per-molecule row positions (positional 0..N-1) to avoid label-index mismatch
    df = df.reset_index(drop=True)
    mol_to_pos: Dict[Any, np.ndarray] = {
        m: np.where(df[molecule_col].to_numpy() == m)[0] for m in df[molecule_col].unique()
    }
    n_mol = len(mol_to_pos)
    mae_samples: List[float] = []
    rmse_samples: List[float] = []
    r2_samples: List[float] = []

    molecules = list(mol_to_pos.keys())
    for _ in range(n_boot):
        sampled = rng.choice(molecules, size=n_mol, replace=True)
        # collect ring-level rows (positional indices)
        rows_idx: List[int] = []
        for m in sampled:
            rows_idx.extend(mol_to_pos[m].tolist())
        boot = df.iloc[rows_idx]
        m_dict = ring_level_metrics(
            boot[y_true_col].to_numpy(dtype=float),
            boot[y_pred_col].to_numpy(dtype=float),
        )
        mae_samples.append(m_dict["MAE"])
        rmse_samples.append(m_dict["RMSE"])
        r2_samples.append(m_dict["R2"])

    alpha = (1 - ci) / 2
    lo = int(alpha * n_boot)
    hi = int((1 - alpha) * n_boot)
    return {
        "MAE": {
            "mean": float(np.mean(mae_samples)),
            "std": float(np.std(mae_samples, ddof=1)),
            "ci_low": float(np.sort(mae_samples)[lo]),
            "ci_high": float(np.sort(mae_samples)[hi]),
            "n_boot": n_boot,
        },
        "RMSE": {
            "mean": float(np.mean(rmse_samples)),
            "std": float(np.std(rmse_samples, ddof=1)),
            "ci_low": float(np.sort(rmse_samples)[lo]),
            "ci_high": float(np.sort(rmse_samples)[hi]),
            "n_boot": n_boot,
        },
        "R2": {
            "mean": float(np.nanmean(r2_samples)),
            "std": float(np.nanstd(r2_samples, ddof=1)),
            "ci_low": float(np.sort(r2_samples)[lo]),
            "ci_high": float(np.sort(r2_samples)[hi]),
            "n_boot": n_boot,
        },
    }


def point_estimate_with_boot_ci(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    molecule_col: str = "canonical_smiles",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Return both point estimate and molecule-cluster bootstrap CI."""
    y_t = df[y_true_col].to_numpy(dtype=float)
    y_p = df[y_pred_col].to_numpy(dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    sub = df.loc[valid].copy()
    pt = ring_level_metrics(sub[y_true_col].to_numpy(dtype=float),
                            sub[y_pred_col].to_numpy(dtype=float))
    boot = molecule_cluster_bootstrap(
        sub, y_true_col, y_pred_col, molecule_col, n_boot=n_boot, seed=seed
    )
    return {"point_estimate": pt, "molecule_cluster_bootstrap_ci": boot}


if __name__ == "__main__":
    # demo: load clean manifest, compute placeholder predictions
    clean_path = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit/lunci10_clean_manifest.csv"
    df = pd.read_csv(clean_path)
    # placeholder: y_pred = y_true (sanity)
    df["y_pred_demo"] = df["HOMA"]
    out = point_estimate_with_boot_ci(df, "HOMA", "y_pred_demo")
    print(json.dumps(out, indent=2))