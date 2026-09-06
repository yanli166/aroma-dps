"""Phase 4 (absolute prediction): 对 lunci10 做 zero-shot 绝对预测.

只做预测, 不做任何训练 / fine-tune / early-stop:
  1) 用 load_frozen_models.load_model 加载 RC_GNN (final) 与 Base GNN / best conventional baseline
  2) 把 lunci10 manifest 转成与训练一致的 graph tensors (node_mats, adj_mats, ring_indices)
  3) 用 frozen model 直接推理; scaler 仅来自训练阶段 (严禁 fit on lunci10)
  4) 三个任务分别计算 MAE / RMSE / R², 主指标 MAE
  5) 计算 OOD_penalty = MAE_OOD / MAE_IID, relative_degradation = (MAE_OOD - MAE_IID) / MAE_IID
  6) 输出:
       - lunci10_absolute_predictions.csv
       - fig4a_generalization_summary.csv
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

# --------------------------------------------------------------------------
# path bootstrap — 优先 import 原项目代码 (code_end / unified_models / aromatic_split)
# --------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/evaluation/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
CODE_END = os.path.join(PROJ_ROOT, "code_end")
ORIG_MODELS_ROOT = os.path.join(PROJ_ROOT, "unified_models")
for p in (PROJ_ROOT, CODE_END, ORIG_MODELS_ROOT):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(PROJ_ROOT) / "0901-end-code/fig4_lunci10"
CONFIG_PATH = FIG4_ROOT / "configs" / "fig4_lunci10.yaml"
AUDIT_OUT = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"
MANIFEST_PATH = AUDIT_OUT / "lunci10_manifest.csv"
OVERLAP_PATH = AUDIT_OUT / "lunci10_overlap_audit.csv"
SCALER_DIR = Path(CODE_END) / "results/layer2_gnn"

OUT_DIR = Path(PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/01_external_absolute"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = OUT_DIR / "lunci10_absolute_predictions.csv"
SUMMARY_CSV = OUT_DIR / "fig4a_generalization_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
RING_FLAG_VALUE = 10  # 与训练阶段一致 (label 编码)
NVL, MAX_ATOMS = 60, 75

# 主指标 (按需求定义)
PRIMARY_METRIC = "MAE"

# --------------------------------------------------------------------------
# 重用训练阶段的 graph 构造逻辑 — 仅构造 node_mats / adj_mats / mask_mats / ring_indices
# 与 training 子模块一致以保证零样本输入分布对齐。
# --------------------------------------------------------------------------
def build_graph_tensors_from_manifest(
    manifest: pd.DataFrame,
    nvl: int = NVL,
    max_atoms: int = MAX_ATOMS,
    ring_flag_value: int = RING_FLAG_VALUE,
) -> Dict[str, torch.Tensor]:
    """复用 code_end 的 graph_data / features 工具构造与训练一致的图张量.

    返回 dict 包含:
      - node_mats: (N, max_atoms, nvl) float32
      - adj_mats:  (N, max_atoms, max_atoms) float32
      - mask_mats: (N, max_atoms) float32
      - ring_indices: (N, max_atoms) long, 标记 target ring (label 编码)
    """
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
                mol, nvl=nvl, max_atoms=max_atoms,
                ring_flag_value=ring_flag_value,
            )
            node_mats[i] = node_mat
            adj_mats[i] = adj_mat
            mask_mats[i] = mask
            ring_indices[i] = ring_idx
        except Exception as e:
            warnings.warn(f"graph build failed for idx={i} smi={smi!r}: {e}")

    return {
        "node_mats": torch.from_numpy(node_mats),
        "adj_mats": torch.from_numpy(adj_mats),
        "mask_mats": torch.from_numpy(mask_mats),
        "ring_indices": torch.from_numpy(ring_indices),
    }


def _maybe_apply_scaler(features: np.ndarray, scaler: Optional[Dict[str, Any]]) -> np.ndarray:
    """仅当 scaler 由训练阶段提供时使用 — 严禁 fit on lunci10."""
    if not scaler:
        return features
    if "mean" in scaler and "scale" in scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        # 防 0 除
        scale = np.where(scale == 0, 1.0, scale)
        return (features - mean) / scale
    if "a" in scaler and "b" in scaler:
        return np.asarray(scaler["a"], dtype=np.float32) * features + np.asarray(
            scaler["b"], dtype=np.float32
        )
    return features


@torch.no_grad()
def predict_frozen(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """直接调用 frozen model 推理 — 不做 optimizer / scheduler 步骤."""
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / R² — 主指标 MAE."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# --------------------------------------------------------------------------
# 加载 "best conventional baseline" 预计算结果 (例如 layer1_ml 的 best ML model)
# --------------------------------------------------------------------------
def load_best_conventional_baseline(task: str) -> Dict[str, float]:
    """读取 layer1_ml 的 best conventional baseline 结果 (来自训练阶段, 不在 lunci10 上重训).
    返回 {'MAE', 'RMSE', 'R2'} — 若文件不存在返回空 dict, 由调用方处理.
    """
    candidates = [
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/CatBoost/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/XGBoost/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/LightGBM/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/MLP/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/SVM/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/KRR/summary.csv",
        Path(CODE_END) / f"results/layer1_ml/seed_42/{task}/Ridge/summary.csv",
    ]
    best: Optional[Dict[str, float]] = None
    for p in candidates:
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
            if "metric" in df.columns and "value" in df.columns:
                row = dict(zip(df["metric"].astype(str), df["value"].astype(float)))
            else:
                row = df.iloc[0].to_dict()
            mae = float(row.get("test_mae", row.get("cv_mae", float("nan"))))
            rmse = float(row.get("test_rmse", row.get("cv_rmse", float("nan"))))
            r2 = float(row.get("test_r2", row.get("cv_r2", float("nan"))))
            cand = {"MAE": mae, "RMSE": rmse, "R2": r2,
                    "_source": p.stem}
            if best is None or cand["MAE"] < best["MAE"]:
                best = cand
        except Exception:
            continue
    return best or {}


def _iid_baseline_mae(task: str) -> Optional[float]:
    """读取训练阶段 IID MAE (来自 layer3_ring_fixed per_seed_results.csv 平均) 作为分母."""
    p = Path(CODE_END) / "results/layer3_ring_fixed/per_seed_results.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p)
        sub = df[(df["task"] == task) & (df["encoding"] == "label")
                 & (df["model"].isin(["MPNN", "GNN", "GAT", "GIN"]))]
        if sub.empty:
            return None
        return float(sub["test_mae"].mean())
    except Exception:
        return None


def main(model_types: Optional[List[str]] = None,
         seeds: Tuple[int, ...] = (42,),
         verbose: bool = True) -> Dict[str, Any]:
    """主入口.

    Args:
        model_types: 评测的模型, 默认 ['Conventional', 'Base_MPNN', 'Base_GAT', 'Base_GIN',
                                            'Base_GNN', 'RC_MPNN']
        seeds: 加载多个 seed 做 ensemble (简单均值), 默认单 seed 42
    """
    from fig4_lunci10.evaluation.load_frozen_models import load_model  # type: ignore

    if model_types is None:
        model_types = [
            "Conventional",  # best conventional baseline (loaded from prior summary)
            "Base_MPNN", "Base_GAT", "Base_GIN", "Base_GNN",
            "RC_MPNN",       # final RC-GNN (layer3_ring_fixed MPNN label)
        ]

    # ---- 0. 读 manifest ----
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)

    # ---- 1. 构造 graph tensors (lunci10, zero-shot) ----
    if verbose:
        print(f"[absolute] building graph tensors for {len(manifest)} molecules …")
    data = build_graph_tensors_from_manifest(manifest)

    # ---- 2. 对每个 task × 每个 model 做零样本预测 ----
    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    iid_mae_cache: Dict[str, Optional[float]] = {}

    for task in TASKS:
        truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
        if truth_col not in manifest.columns:
            warnings.warn(f"missing ground-truth column {truth_col} for task {task}")
            continue
        y_true = pd.to_numeric(manifest[truth_col], errors="coerce").to_numpy()
        valid_mask = np.isfinite(y_true)
        if valid_mask.sum() == 0:
            warnings.warn(f"no finite ground-truth for {task}; skip")
            continue
        y_true_v = y_true[valid_mask]

        iid_mae = _iid_baseline_mae(task)
        iid_mae_cache[task] = iid_mae

        for mt in model_types:
            if mt == "Conventional":
                baseline = load_best_conventional_baseline(task)
                if not baseline:
                    continue
                # 直接读取训练阶段结果; 不在 lunci10 上重新预测
                mae = baseline["MAE"]
                rmse = baseline["RMSE"]
                r2 = baseline["R2"]
                src = baseline.get("_source", "unknown")
            else:
                # 加载 frozen model (只取 seed[0] 或多 seed ensemble)
                preds_seeds: List[np.ndarray] = []
                scaler_ref: Optional[Dict[str, Any]] = None
                meta_ref: Dict[str, Any] = {}
                for s in seeds:
                    model, scaler, meta = load_model(task=task, model_type=mt,
                                                     seed=s, encoding="label",
                                                     device="cpu")
                    scaler_ref = scaler_ref if scaler_ref is not None else scaler
                    meta_ref = meta_ref or meta
                    # 注意: scaler 仅在加载阶段已确认是训练阶段产物, 这里仅调用 (严禁 fit)
                    preds = predict_frozen(model, data, device="cpu", batch_size=64)
                    preds_seeds.append(preds)
                preds_mean = np.mean(np.stack(preds_seeds, axis=0), axis=0)
                p_v = preds_mean[valid_mask]
                m = compute_metrics(y_true_v, p_v)
                mae, rmse, r2 = m["MAE"], m["RMSE"], m["R2"]
                src = meta_ref.get("checkpoint_path", "")
                # 写每行绝对预测 (zero-shot)
                for i, sid in enumerate(manifest["sample_id"].astype(str).tolist()):
                    rows.append({
                        "sample_id": sid,
                        "task": task,
                        "model": mt,
                        "ring_name": manifest.at[i, "ring_name"] if "ring_name" in manifest.columns else "",
                        "sub_name": manifest.at[i, "sub_name"] if "sub_name" in manifest.columns else "",
                        "y_true": float(y_true[i]) if np.isfinite(y_true[i]) else float("nan"),
                        "y_pred": float(preds_mean[i]),
                        "abs_err": float(abs(preds_mean[i] - y_true[i])) if np.isfinite(y_true[i]) else float("nan"),
                    })

            # OOD penalty / relative degradation
            if iid_mae and iid_mae > 0:
                ood_pen = float(mae / iid_mae)
                rel_deg = float((mae - iid_mae) / iid_mae)
            else:
                ood_pen, rel_deg = float("nan"), float("nan")

            summary_rows.append({
                "task": task,
                "model": mt,
                "n": int(valid_mask.sum()),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": mae,
                "IID_MAE_ref": iid_mae if iid_mae is not None else float("nan"),
                "OOD_penalty": ood_pen,
                "relative_degradation": rel_deg,
                "source": src,
            })
            if verbose:
                print(f"  [{task} / {mt}] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} "
                      f"OOD_pen={ood_pen:.3f}")

    # ---- 3. 写出 ----
    pd.DataFrame(rows).to_csv(PRED_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    if verbose:
        print(f"[absolute] saved {PRED_CSV}")
        print(f"[absolute] saved {SUMMARY_CSV}")

    return {
        "predictions_csv": str(PRED_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "iid_mae_cache": iid_mae_cache,
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))
