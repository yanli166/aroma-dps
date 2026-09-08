"""Phase 7 (pairwise delta prediction): 基于 frozen 绝对模型计算 ΔAhat_ij = Ahat_i - Ahat_j.

约束:
  - 仅调用 frozen checkpoints (来自 load_frozen_models.load_model) — 不做训练/fine-tune
  - scaler 仅来自训练阶段 — 禁止 fit on lunci10
  - sign convention: ΔAhat = Ahat_i - Ahat_j (与 build_pairs.py 保持一致)
  - 报告指标: MAE, RMSE, R², Pearson r, Spearman ρ, sign accuracy
  - 三个 aggregation level:
      * micro: 全局 flat 评估
      * macro-context: 按 (ring_name, ring_pos, target_ring_id, sub_type) 分组求均值
      * macro-ring: 按 ring_name 分组求均值

主要产物:
  - lunci10_pair_predictions.csv
  - fig4c_pairwise_summary.csv
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/evaluation/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
CODE_END = os.path.join(PROJ_ROOT, "archive/deprecated/code_end")
ORIG_MODELS_ROOT = os.path.join(PROJ_ROOT, "unified_models")
for p in (PROJ_ROOT, CODE_END, ORIG_MODELS_ROOT):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(PROJ_ROOT) / "0901-end-code/fig4_lunci10"
PAIR_DIR = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/03_pairwise"
PAIR_MANIFEST = PAIR_DIR / "lunci10_pair_manifest.csv"
AUDIT_OUT = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"
MANIFEST_PATH = AUDIT_OUT / "lunci10_manifest.csv"

OUT_DIR = PAIR_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = OUT_DIR / "lunci10_pair_predictions.csv"
SUMMARY_CSV = OUT_DIR / "fig4c_pairwise_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO", "NICS_iso"]
PRIMARY_METRIC = "MAE"

NVL, MAX_ATOMS = 60, 75
RING_FLAG_VALUE = 10


# --------------------------------------------------------------------------
# graph 构建 (与 run_external_absolute 共享相同入口)
# --------------------------------------------------------------------------
def build_graph_tensors_from_manifest(
    manifest: pd.DataFrame,
    nvl: int = NVL,
    max_atoms: int = MAX_ATOMS,
    ring_flag_value: int = RING_FLAG_VALUE,
) -> Dict[str, torch.Tensor]:
    from common.graph_data import mol_to_graph_tensors  # type: ignore
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    n = len(manifest)
    node_mats = np.zeros((n, max_atoms, nvl), dtype=np.float32)
    adj_mats = np.zeros((n, max_atoms, max_atoms), dtype=np.float32)
    mask_mats = np.zeros((n, max_atoms), dtype=np.float32)
    ring_indices = np.full((n, max_atoms), -1, dtype=np.int64)
    for i, row in enumerate(manifest.itertuples(index=False)):
        smi = getattr(row, "canonical_smiles", "") or ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            node_mat, adj_mat, mask, ring_idx = mol_to_graph_tensors(
                mol, nvl=nvl, max_atoms=max_atoms, ring_flag_value=ring_flag_value,
            )
            node_mats[i] = node_mat
            adj_mats[i] = adj_mat
            mask_mats[i] = mask
            ring_indices[i] = ring_idx
        except Exception as e:
            warnings.warn(f"graph build failed for idx={i}: {e}")
    return {
        "node_mats": torch.from_numpy(node_mats),
        "adj_mats": torch.from_numpy(adj_mats),
        "mask_mats": torch.from_numpy(mask_mats),
        "ring_indices": torch.from_numpy(ring_indices),
    }


@torch.no_grad()
def predict_frozen(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    model = model.to(device).eval()
    node_mats = data["node_mats"].to(device)
    adj_mats = data["adj_mats"].to(device)
    n = node_mats.size(0)
    preds: List[np.ndarray] = []
    for i in range(0, n, batch_size):
        nm = node_mats[i:i + batch_size]
        am = adj_mats[i:i + batch_size]
        out = model(nm, am)
        if isinstance(out, torch.Tensor):
            out = out.squeeze(-1)
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)


def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> Optional[float]:
    sub = np.stack([x, y], axis=1)
    sub = sub[np.isfinite(sub).all(axis=1)]
    if sub.shape[0] < 3:
        return None
    try:
        v = fn(sub[:, 0], sub[:, 1])
        return float(v.statistic) if hasattr(v, "statistic") else float(v[0])
    except Exception:
        return None


def compute_pair_metrics(d_true: np.ndarray, d_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE/RMSE/R²/Pearson r/Spearman ρ/sign-accuracy for one slice."""
    sub = np.stack([d_true, d_pred], axis=1)
    sub = sub[np.isfinite(sub).all(axis=1)]
    if sub.shape[0] == 0:
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                "pearson_r": float("nan"), "spearman_rho": float("nan"),
                "sign_accuracy": float("nan")}
    y_t, y_p = sub[:, 0], sub[:, 1]
    err = y_p - y_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    pr = _safe_corr(y_t, y_p, lambda a, b: pearsonr(a, b))
    sp = _safe_corr(y_t, y_p, lambda a, b: spearmanr(a, b))
    sign_acc = float(np.mean(np.sign(y_t) == np.sign(y_p)))
    return {
        "n": int(sub.shape[0]),
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "pearson_r": pr if pr is not None else float("nan"),
        "spearman_rho": sp if sp is not None else float("nan"),
        "sign_accuracy": sign_acc,
    }


def _aggregate_macro(values: List[Dict[str, float]], metric_keys: List[str]) -> Dict[str, float]:
    """Simple macro-aggregation: mean over groups where the metric is finite."""
    out: Dict[str, float] = {}
    for k in metric_keys:
        vals = [v[k] for v in values if v.get(k) is not None and np.isfinite(v.get(k, float("nan")))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_groups"] = int(len(values))
    out["n_total"] = int(sum(v.get("n", 0) for v in values))
    return out


def main(
    model_types: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase 7 main entry.

    Args:
        model_types: 待评测模型, 默认 ['RC_MPNN', 'Base_MPNN', 'Base_GAT', 'Base_GNN']
        seed: frozen model seed (默认 42; 严禁用 lunci10 选 seed)
    """
    from fig4_lunci10.evaluation.load_frozen_models import load_model  # type: ignore

    if model_types is None:
        model_types = ["RC_MPNN", "Base_MPNN", "Base_GAT", "Base_GNN"]

    if not PAIR_MANIFEST.is_file():
        raise FileNotFoundError(f"missing pair manifest: {PAIR_MANIFEST}; run build_pairs.py first")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing lunci10 manifest: {MANIFEST_PATH}")

    pair_df = pd.read_csv(PAIR_MANIFEST)
    manifest = pd.read_csv(MANIFEST_PATH)

    # ---- 1. 构造 manifest unique sample 图张量 ----
    # 收集 (sample_id, canonical_smiles) 全集用于推理
    if verbose:
        print(f"[delta] building graph tensors for {len(manifest)} unique molecules …")
    data = build_graph_tensors_from_manifest(manifest)
    # id_map: sample_id -> manifest row index
    id_map = {str(sid): i for i, sid in enumerate(manifest["sample_id"].astype(str).tolist())}

    # ---- 2. 对每个 model: 加载 frozen → 推理 → 写 Δ ----
    pred_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for mt in model_types:
        if verbose:
            print(f"[delta] model={mt} seed={seed} …")
        # 加载 frozen model
        per_task_pred: Dict[str, np.ndarray] = {}
        scaler_ref: Optional[Dict[str, Any]] = None
        meta_ref: Dict[str, Any] = {}
        for task in TASKS:
            try:
                model, scaler, meta = load_model(task=task, model_type=mt,
                                                 seed=seed, encoding="label",
                                                 device="cpu")
            except Exception as e:
                warnings.warn(f"failed to load {mt}/{task}: {e}")
                continue
            scaler_ref = scaler_ref if scaler_ref is not None else scaler
            meta_ref = meta_ref or meta
            preds = predict_frozen(model, data, device="cpu", batch_size=64)
            per_task_pred[task] = preds

        # ---- 3. 写每对 (i, j) 的 Δ ----
        metric_keys = ["MAE", "RMSE", "R2", "pearson_r", "spearman_rho", "sign_accuracy"]
        per_task_micro: Dict[str, Dict[str, float]] = {}
        per_task_ctx: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        per_task_ring: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for task in TASKS:
            preds = per_task_pred.get(task)
            if preds is None:
                continue

            truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
            delta_pred_col = f"delta_{task}_pred"
            delta_true_col = f"delta_{truth_col}_true" if task == "NICS_1zz" else f"delta_{task}_true"
            if delta_true_col not in pair_df.columns:
                continue

            for _, prow in pair_df.iterrows():
                sid_i = str(prow.get("sample_id_i", ""))
                sid_j = str(prow.get("sample_id_j", ""))
                idx_i = id_map.get(sid_i, -1)
                idx_j = id_map.get(sid_j, -1)
                if idx_i < 0 or idx_j < 0:
                    continue
                # sign convention (consistent with build_pairs.py):
                # ΔAhat_ij = Ahat_i - Ahat_j  (i.e. signed diff, orientation deterministic)
                ahat_i = float(preds[idx_i])
                ahat_j = float(preds[idx_j])
                delta_pred = ahat_i - ahat_j
                delta_true = float(prow[delta_true_col]) if pd.notna(prow[delta_true_col]) else float("nan")

                pred_rows.append({
                    "ring_name": prow.get("ring_name", ""),
                    "ring_pos": prow.get("ring_pos", ""),
                    "target_ring_id": prow.get("target_ring_id", ""),
                    "sub_type": prow.get("sub_type", ""),
                    "sub_i": prow.get("sub_i", ""),
                    "sub_j": prow.get("sub_j", ""),
                    "sample_id_i": sid_i,
                    "sample_id_j": sid_j,
                    "task": task,
                    "model": mt,
                    "delta_true": delta_true,
                    "delta_pred": delta_pred,
                })

            # ---- 4. 计算 micro / macro-context / macro-ring ----
            sub_df = pd.DataFrame([r for r in pred_rows if r["model"] == mt and r["task"] == task])
            if sub_df.empty:
                continue
            d_t = sub_df["delta_true"].to_numpy(dtype=float)
            d_p = sub_df["delta_pred"].to_numpy(dtype=float)
            micro = compute_pair_metrics(d_t, d_p)
            per_task_micro[task] = micro

            # macro-context
            ctx_keys = sub_df[["ring_name", "ring_pos", "target_ring_id", "sub_type"]].astype(str).agg("||".join, axis=1)
            for ck, g in sub_df.assign(__ctx=ctx_keys).groupby("__ctx"):
                m = compute_pair_metrics(g["delta_true"].to_numpy(dtype=float),
                                          g["delta_pred"].to_numpy(dtype=float))
                per_task_ctx[task].append(m)

            # macro-ring
            for rn, g in sub_df.groupby("ring_name"):
                m = compute_pair_metrics(g["delta_true"].to_numpy(dtype=float),
                                          g["delta_pred"].to_numpy(dtype=float))
                per_task_ring[task].append(m)

        # 写 summary
        for task in TASKS:
            if task not in per_task_micro:
                continue
            micro = per_task_micro[task]
            macro_ctx = _aggregate_macro(per_task_ctx[task], metric_keys)
            macro_ring = _aggregate_macro(per_task_ring[task], metric_keys)
            summary_rows.append({
                "model": mt,
                "task": task,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": micro["MAE"],
                **{
                    f"micro_{k}": micro[k] for k in metric_keys + ["n"]
                },
                **{f"macro_context_{k}": macro_ctx[k] for k in metric_keys + ["n_groups", "n_total"]},
                **{f"macro_ring_{k}": macro_ring[k] for k in metric_keys + ["n_groups", "n_total"]},
            })
            if verbose:
                print(f"  [{mt}/{task}] micro MAE={micro['MAE']:.4f} "
                      f"R2={micro['R2']:.4f} sign_acc={micro['sign_accuracy']:.3f} "
                      f"r={micro['pearson_r']:.3f} ρ={micro['spearman_rho']:.3f}")

    # ---- 5. 写产物 ----
    pd.DataFrame(pred_rows).to_csv(PRED_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    if verbose:
        print(f"[delta] saved {PRED_CSV}")
        print(f"[delta] saved {SUMMARY_CSV}")

    return {
        "predictions_csv": str(PRED_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "n_pairs": len(pred_rows),
        "n_summary_rows": len(summary_rows),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))
