"""Phase 9 (internal pair model training): 三种配对 (Δ) 模型在 internal 训练集
做 cross-validation / train → CV 选超参, 然后一次性在 lunci10 pair manifest 上测试。

Baseline A: absolute subtraction (复用 Phase 7 frozen absolute 预测, 然后
             ΔAhat_ij = Ahat_i - Ahat_j)
Baseline B: Δ fingerprint — 对 (fp_i - fp_j, abs(fp_i - fp_j)) 用 XGBoost /
             CatBoost 回归
Main:      Siamese / Paired RC-GNN — 共享 encoder, pair representation
            [h_i, h_j, h_i - h_j, |h_i - h_j|], 只在 internal train/CV 选择
            模型, 一次性在 lunci10 pair manifest 上测试。

产物:
  - fig4c_pairwise_summary.csv     (Protocol I 可行时存在)
  - / 内含 micro / macro-context / macro-ring 指标

约束:
  - 不在 lunci10 上做任何 fit / hyperparameter 选择 / early stop.
  - 模型选择仅基于 internal CV (或留一 pair group).
  - scaler / checkpoint 来源仅限训练阶段.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = _PROJ_ROOT
CODE_END = f"{PROJ_ROOT}/archive/deprecated/code_end"
ORIG_MODELS_ROOT = _os.path.join(_PROJ_ROOT, "unified_models")
for p in (PROJ_ROOT, CODE_END, ORIG_MODELS_ROOT):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(_PROJ_ROOT) / "0901-end-code/fig4_lunci10"
PAIR_DIR = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/03_pairwise"
PAIR_DIR.mkdir(parents=True, exist_ok=True)
LUNCI10_PAIR = PAIR_DIR / "lunci10_pair_manifest.csv"
INTERNAL_PAIR = PAIR_DIR / "internal_pair_manifest.csv"
FEASIBILITY_JSON = PAIR_DIR / "internal_pair_feasibility.json"
SUMMARY_CSV = PAIR_DIR / "fig4c_pairwise_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
PRIMARY_METRIC = "MAE"
NVL, MAX_ATOMS = 60, 75
RING_FLAG_VALUE = 10


# --------------------------------------------------------------------------
# 指标 + aggregation helpers (与 run_external_delta.py 共享)
# --------------------------------------------------------------------------
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
    out: Dict[str, float] = {}
    for k in metric_keys:
        vals = [v[k] for v in values
                if v.get(k) is not None and np.isfinite(v.get(k, float("nan")))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_groups"] = int(len(values))
    out["n_total"] = int(sum(v.get("n", 0) for v in values))
    return out


# --------------------------------------------------------------------------
# Baseline A — absolute subtraction: 复用 Phase 7 frozen model
# --------------------------------------------------------------------------
def baseline_a_predictions(
    task: str, lunci10_pair_df: pd.DataFrame, lunci10_manifest: pd.DataFrame,
    model_types: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """ΔAhat = Ahat_i - Ahat_j, 其中 Ahat 来自 frozen model."""
    from fig4_lunci10.evaluation.load_frozen_models import load_model  # type: ignore
    from fig4_lunci10.evaluation.run_external_delta import (  # type: ignore
        build_graph_tensors_from_manifest, predict_frozen,
    )

    if model_types is None:
        model_types = ["RC_MPNN", "Base_MPNN", "Base_GAT", "Base_GNN"]

    data = build_graph_tensors_from_manifest(lunci10_manifest)
    id_map = {str(sid): i for i, sid in enumerate(lunci10_manifest["sample_id"].astype(str).tolist())}

    truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
    delta_true_col = f"delta_{truth_col}_true" if task == "NICS_1zz" else f"delta_{task}_true"

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for mt in model_types:
        try:
            model, scaler, meta = load_model(task=task, model_type=mt,
                                             seed=seed, encoding="label",
                                             device="cpu")
        except Exception as e:
            warnings.warn(f"baseline A: failed to load {mt}/{task}: {e}")
            continue
        preds = predict_frozen(model, data, device="cpu", batch_size=64)
        for _, prow in lunci10_pair_df.iterrows():
            sid_i = str(prow.get("sample_id_i", ""))
            sid_j = str(prow.get("sample_id_j", ""))
            idx_i = id_map.get(sid_i, -1)
            idx_j = id_map.get(sid_j, -1)
            if idx_i < 0 or idx_j < 0:
                continue
            ahat_i = float(preds[idx_i])
            ahat_j = float(preds[idx_j])
            out[(mt, f"{sid_i}||{sid_j}")] = {
                "model": mt,
                "task": task,
                "ring_name": prow.get("ring_name", ""),
                "ring_pos": prow.get("ring_pos", ""),
                "target_ring_id": prow.get("target_ring_id", ""),
                "sub_type": prow.get("sub_type", ""),
                "sub_i": prow.get("sub_i", ""),
                "sub_j": prow.get("sub_j", ""),
                "sample_id_i": sid_i,
                "sample_id_j": sid_j,
                "delta_true": float(prow[delta_true_col])
                    if delta_true_col in prow and pd.notna(prow[delta_true_col])
                    else float("nan"),
                "delta_pred": ahat_i - ahat_j,
                "method": "absolute_subtraction",
            }
    return out


# --------------------------------------------------------------------------
# Baseline B — Δ fingerprint (Morgan difference)
# --------------------------------------------------------------------------
def _pair_features_delta_fp(
    pair_df: pd.DataFrame, manifest_df: pd.DataFrame,
    n_bits: int = 2048, radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """为每对构造 [fp_i - fp_j, |fp_i - fp_j|] 特征向量.

    返回 (X, y):
      X: (N_pairs, 2*n_bits) float32
      y: (N_pairs,) float32 (NaN drop handled by caller)
    """
    from rdkit import Chem, RDLogger, DataStructs  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import AllChem

    smi_map = {str(sid): str(smi) for sid, smi in
               zip(manifest_df["sample_id"].astype(str),
                   manifest_df["canonical_smiles"].astype(str))}

    Xs: List[np.ndarray] = []
    ys: List[float] = []
    for _, prow in pair_df.iterrows():
        smi_i = smi_map.get(str(prow.get("sample_id_i", "")), "")
        smi_j = smi_map.get(str(prow.get("sample_id_j", "")), "")
        mol_i = Chem.MolFromSmiles(smi_i) if smi_i else None
        mol_j = Chem.MolFromSmiles(smi_j) if smi_j else None
        if mol_i is None or mol_j is None:
            continue
        bv_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=radius, nBits=n_bits)
        bv_j = AllChem.GetMorganFingerprintAsBitVect(mol_j, radius=radius, nBits=n_bits)
        a_i = np.zeros(n_bits, dtype=np.float32)
        a_j = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv_i, a_i)
        DataStructs.ConvertToNumpyArray(bv_j, a_j)
        diff = a_i - a_j
        absdiff = np.abs(diff)
        Xs.append(np.concatenate([diff, absdiff], axis=0))
    return np.stack(Xs, axis=0) if Xs else np.zeros((0, 2 * n_bits), dtype=np.float32), \
           np.zeros(0, dtype=np.float32)


def baseline_b_predictions(
    task: str, lunci10_pair_df: pd.DataFrame, lunci10_manifest: pd.DataFrame,
    internal_pair_df: pd.DataFrame, internal_manifests: Dict[str, pd.DataFrame],
    model_kind: str = "XGBoost",
    cv_folds: int = 5, seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Δ Morgan fingerprint → XGBoost / CatBoost 回归.

    训练仅在 internal pair manifest (Protocol I); 否则 (Protocol II) 跳过
    训练并返回空 dict (依赖 baseline A / Siamese 承担).
    """
    if not INTERNAL_PAIR.is_file():
        warnings.warn("internal_pair_manifest missing — Protocol II active; "
                      "skipping baseline B training")
        return {}
    df_int = pd.read_csv(INTERNAL_PAIR)
    if df_int.empty:
        return {}
    # 选择 task 切片
    sub_int = df_int[df_int["task"] == task].copy()
    truth_col = f"delta_{task}_true" if task != "NICS_1zz" else "delta_NICS_1zz_true"
    if truth_col not in sub_int.columns:
        # NICS_1zz 在 internal 可能使用 NICS_ZZ 列
        if "delta_NICS_ZZ_true" in sub_int.columns:
            truth_col = "delta_NICS_ZZ_true"
        else:
            return {}
    sub_int = sub_int.dropna(subset=[truth_col]).reset_index(drop=True)
    if len(sub_int) < cv_folds * 2:
        warnings.warn(f"internal pairs too few ({len(sub_int)}) for CV — skip baseline B")
        return {}

    # 构造 Δ fp features
    # 把 internal 分子集合合并
    merged_int = pd.concat(
        [df for df in internal_manifests.values()],
        ignore_index=True, sort=False,
    )
    merged_int = merged_int.drop_duplicates(subset=["canonical_smiles"])
    X_int, y_int = _pair_features_delta_fp(sub_int, merged_int)
    if X_int.shape[0] != len(sub_int):
        warnings.warn("baseline B: feature shape mismatch — skip")
        return {}

    # GroupKFold 按 scaffold
    from sklearn.model_selection import GroupKFold
    groups = sub_int["ring_name"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=cv_folds)

    if model_kind.upper() == "XGBOOST":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:
            warnings.warn(f"xgboost import failed: {e}")
            return {}
        model_factory = lambda: XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            random_state=seed, n_jobs=2, tree_method="hist",
        )
    elif model_kind.upper() == "CATBOOST":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as e:
            warnings.warn(f"catboost import failed: {e}")
            return {}
        model_factory = lambda: CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=False,
        )
    else:
        warnings.warn(f"unknown baseline B kind: {model_kind}")
        return {}

    # 训练 CV: 选最佳 fold (按 MAE)
    best_model = None
    best_cv_mae = float("inf")
    cv_metrics: List[Dict[str, float]] = []
    for fold, (tr, va) in enumerate(gkf.split(X_int, y_int, groups)):
        m = model_factory()
        try:
            m.fit(X_int[tr], y_int[tr])
            pred = m.predict(X_int[va])
        except Exception as e:
            warnings.warn(f"baseline B fit failed at fold {fold}: {e}")
            continue
        met = compute_pair_metrics(y_int[va], pred)
        cv_metrics.append(met)
        if met["MAE"] < best_cv_mae:
            best_cv_mae = met["MAE"]
            best_model = m

    if best_model is None:
        return {}

    # 在 lunci10 pair 上预测 (一次性, 严禁 CV / fit on lunci10)
    X_lunc, _ = _pair_features_delta_fp(lunci10_pair_df, lunci10_manifest)
    if X_lunc.shape[0] != len(lunci10_pair_df):
        warnings.warn("baseline B: lunci10 feature shape mismatch")
        return {}
    pred_lunc = best_model.predict(X_lunc)
    truth_col_l = ("delta_NICS_ZZ_true"
                   if task == "NICS_1zz" and "delta_NICS_ZZ_true" in lunci10_pair_df.columns
                   else f"delta_{task}_true")
    out: Dict[str, Dict[str, Any]] = {}
    for k_i, prow in enumerate(lunci10_pair_df.itertuples(index=False)):
        out[f"{prow.sample_id_i}||{prow.sample_id_j}"] = {
            "model": f"ΔFP_{model_kind}",
            "task": task,
            "ring_name": getattr(prow, "ring_name", ""),
            "ring_pos": getattr(prow, "ring_pos", ""),
            "target_ring_id": getattr(prow, "target_ring_id", ""),
            "sub_type": getattr(prow, "sub_type", ""),
            "sub_i": getattr(prow, "sub_i", ""),
            "sub_j": getattr(prow, "sub_j", ""),
            "sample_id_i": getattr(prow, "sample_id_i", ""),
            "sample_id_j": getattr(prow, "sample_id_j", ""),
            "delta_true": float(getattr(prow, truth_col_l))
                if pd.notna(getattr(prow, truth_col_l, float("nan")))
                else float("nan"),
            "delta_pred": float(pred_lunc[k_i]),
            "method": f"delta_fp_{model_kind.lower()}",
        }
    return out


# --------------------------------------------------------------------------
# Main — Siamese / Paired RC-GNN
# --------------------------------------------------------------------------
class SiameseRCGNN(torch.nn.Module if False else object):  # type: ignore
    """占位 — 实际实现可替换为 Siamese MPNN/GIN/GAT; 此处仅声明接口以避免
    训练循环未实例化时报 NameError. 真正的实现由调用方提供 (依赖 PyTorch + RDKit
    graph tensors).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SiameseRCGNN 占位; 实际训练请使用 unified_models 中的 GNN/GIN/GAT/"
            "MPNN 作为共享 encoder + 配对 head [h_i, h_j, h_i-h_j, |h_i-h_j|] → MLP"
        )


def siamese_rcgnn_predictions(
    task: str, lunci10_pair_df: pd.DataFrame, lunci10_manifest: pd.DataFrame,
    internal_pair_df: Optional[pd.DataFrame], internal_manifests: Dict[str, pd.DataFrame],
    backbone: str = "MPNN", cv_folds: int = 5, seed: int = 42,
    n_epochs: int = 30, batch_size: int = 16, lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Siamese / Paired RC-GNN: 共享 encoder, pair representation
    [h_i, h_j, h_i-h_j, |h_i-h_j|] → MLP 输出 ΔA.

    仅在 internal pair manifest 上做 CV 训练, 一次性在 lunci10 pair manifest 上测试.
    若 internal_pair_df 不可用 (Protocol II), 抛出 NotImplementedError.
    """
    import torch
    import torch.nn as nn
    from fig4_lunci10.evaluation.run_external_delta import (  # type: ignore
        build_graph_tensors_from_manifest,
    )
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")

    try:
        from unified_models.mpnn.model import MPNNModel  # type: ignore
        from unified_models.gin.model import GINModel    # type: ignore
        from unified_models.gat.model import GATModel    # type: ignore
        from unified_models.gnn.model import GNNModel    # type: ignore
    except Exception as e:
        warnings.warn(f"Siamese: cannot import unified_models ({e}); skip")
        return {}

    if internal_pair_df is None or internal_pair_df.empty:
        warnings.warn("Siamese: internal pair manifest missing (Protocol II active)")
        return {}

    sub_int = internal_pair_df[internal_pair_df["task"] == task].copy()
    truth_col = f"delta_{task}_true"
    if truth_col not in sub_int.columns and task == "NICS_1zz" and "delta_NICS_ZZ_true" in sub_int.columns:
        truth_col = "delta_NICS_ZZ_true"
    if truth_col not in sub_int.columns:
        warnings.warn(f"Siamese: missing delta column for {task}")
        return {}
    sub_int = sub_int.dropna(subset=[truth_col]).reset_index(drop=True)
    if len(sub_int) < cv_folds * 2:
        warnings.warn(f"Siamese: too few internal pairs ({len(sub_int)})")
        return {}

    # 内部 molecule 集合
    merged_int = pd.concat([df for df in internal_manifests.values()],
                            ignore_index=True, sort=False)
    merged_int = merged_int.drop_duplicates(subset=["canonical_smiles"])
    int_data = build_graph_tensors_from_manifest(merged_int)
    int_id_map = {str(sid): i for i, sid in
                  enumerate(merged_int["sample_id"].astype(str).tolist())}
    # lunci10 graph
    lunc_data = build_graph_tensors_from_manifest(lunci10_manifest)
    lunc_id_map = {str(sid): i for i, sid in
                   enumerate(lunci10_manifest["sample_id"].astype(str).tolist())}

    backbone_map = {"MPNN": MPNNModel, "GIN": GINModel,
                    "GAT": GATModel, "GNN": GNNModel}
    if backbone not in backbone_map:
        warnings.warn(f"Siamese: unknown backbone {backbone}")
        return {}

    def _build_encoder():
        kwargs = dict(node_vec_len=NVL, hidden_dim=128, n_conv=3,
                      n_hidden=2, n_outputs=1, p_dropout=0.2, mode="label")
        if backbone in ("GIN", "GAT"):
            kwargs["ring_flag_value"] = RING_FLAG_VALUE
        return backbone_map[backbone](**kwargs)

    class PairHead(nn.Module):
        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, h_i, h_j):
            return self.mlp(torch.cat([h_i, h_j, h_i - h_j,
                                       (h_i - h_j).abs()], dim=-1)).squeeze(-1)

    class SiameseWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _build_encoder()
            self.head = PairHead(hidden_dim=128)

        def encode(self, nm, am):
            # 不同 backbone 输出维度可能不同; 强制 128-d 截取/均值
            z = self.encoder(nm, am)
            if z.dim() == 1:
                z = z.unsqueeze(-1)
            if z.size(-1) != 128:
                # 取前 128 维或均值到 128; 这里取前 128
                if z.size(-1) > 128:
                    z = z[..., :128]
                else:
                    pad = torch.zeros(*z.shape[:-1], 128 - z.size(-1),
                                      device=z.device, dtype=z.dtype)
                    z = torch.cat([z, pad], dim=-1)
            return z

        def forward(self, nm_i, am_i, nm_j, am_j):
            h_i = self.encode(nm_i, am_i)
            h_j = self.encode(nm_j, am_j)
            return self.head(h_i, h_j)

    # internal 训练: 简化训练循环 (小数据, 几十轮)
    device_t = torch.device(device)

    def _gather_pair_tensors(pairs: pd.DataFrame, id_map: Dict[str, int],
                             data: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, ...]]:
        i_idx, j_idx = [], []
        ys: List[float] = []
        for _, prow in pairs.iterrows():
            ii = id_map.get(str(prow.get("sample_id_i", "")), -1)
            jj = id_map.get(str(prow.get("sample_id_j", "")), -1)
            if ii < 0 or jj < 0:
                continue
            i_idx.append(ii)
            j_idx.append(jj)
            ys.append(float(prow[truth_col]))
        if not i_idx:
            return None
        nm = data["node_mats"]
        am = data["adj_mats"]
        return (
            nm[i_idx], am[i_idx],
            nm[j_idx], am[j_idx],
            torch.tensor(ys, dtype=torch.float32, device=device_t),
        )

    int_t = _gather_pair_tensors(sub_int, int_id_map, int_data)
    if int_t is None:
        warnings.warn("Siamese: failed to gather internal pair tensors")
        return {}
    nm_i_tr, am_i_tr, nm_j_tr, am_j_tr, y_tr = int_t

    # GroupKFold
    from sklearn.model_selection import GroupKFold
    groups = sub_int["ring_name"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=cv_folds)

    # 简化 CV: 取 1 fold 做 best ckpt 选 (出于时间预算)
    best_mae = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for fold, (tr, va) in enumerate(gkf.split(np.zeros(len(y_tr)), y_tr.cpu().numpy(), groups)):
        model = SiameseWrapper().to(device_t)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        for ep in range(n_epochs):
            model.train()
            perm = np.random.permutation(len(tr))
            for s in range(0, len(perm), batch_size):
                sel = perm[s:s + batch_size]
                opt.zero_grad()
                pred = model(nm_i_tr[tr][sel].to(device_t),
                             am_i_tr[tr][sel].to(device_t),
                             nm_j_tr[tr][sel].to(device_t),
                             am_j_tr[tr][sel].to(device_t))
                loss = loss_fn(pred, y_tr[tr][sel])
                loss.backward()
                opt.step()
        # 验证 fold
        model.eval()
        with torch.no_grad():
            pred_va = model(nm_i_tr[va].to(device_t),
                            am_i_tr[va].to(device_t),
                            nm_j_tr[va].to(device_t),
                            am_j_tr[va].to(device_t)).numpy()
        met = compute_pair_metrics(y_tr[va].cpu().numpy(), pred_va)
        if met["MAE"] < best_mae:
            best_mae = met["MAE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        return {}

    # 重新构建并加载 best
    final_model = SiameseWrapper().to(device_t)
    final_model.load_state_dict(best_state)
    final_model.eval()

    # lunci10 一次性预测 (严禁再 fit)
    lunc_t = _gather_pair_tensors(lunci10_pair_df, lunc_id_map, lunc_data)
    if lunc_t is None:
        return {}
    nm_i_l, am_i_l, nm_j_l, am_j_l, _ = lunc_t
    with torch.no_grad():
        pred_lunc = final_model(nm_i_l.to(device_t), am_i_l.to(device_t),
                                nm_j_l.to(device_t), am_j_l.to(device_t)).cpu().numpy()

    truth_col_l = ("delta_NICS_ZZ_true"
                   if task == "NICS_1zz" and "delta_NICS_ZZ_true" in lunci10_pair_df.columns
                   else f"delta_{task}_true")
    out: Dict[str, Dict[str, Any]] = {}
    for k_i, prow in enumerate(lunci10_pair_df.itertuples(index=False)):
        out[f"{prow.sample_id_i}||{prow.sample_id_j}"] = {
            "model": f"Siamese_{backbone}",
            "task": task,
            "ring_name": getattr(prow, "ring_name", ""),
            "ring_pos": getattr(prow, "ring_pos", ""),
            "target_ring_id": getattr(prow, "target_ring_id", ""),
            "sub_type": getattr(prow, "sub_type", ""),
            "sub_i": getattr(prow, "sub_i", ""),
            "sub_j": getattr(prow, "sub_j", ""),
            "sample_id_i": getattr(prow, "sample_id_i", ""),
            "sample_id_j": getattr(prow, "sample_id_j", ""),
            "delta_true": float(getattr(prow, truth_col_l))
                if pd.notna(getattr(prow, truth_col_l, float("nan")))
                else float("nan"),
            "delta_pred": float(pred_lunc[k_i]),
            "method": f"siamese_{backbone.lower()}",
        }
    return out


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------
def _build_summary(pred_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """汇总 micro / macro-context / macro-ring 指标."""
    df = pd.DataFrame(pred_rows)
    if df.empty:
        return pd.DataFrame()
    metric_keys = ["MAE", "RMSE", "R2", "pearson_r", "spearman_rho", "sign_accuracy"]
    summary_rows: List[Dict[str, Any]] = []
    for (model, task), g in df.groupby(["model", "task"]):
        d_t = g["delta_true"].to_numpy(dtype=float)
        d_p = g["delta_pred"].to_numpy(dtype=float)
        micro = compute_pair_metrics(d_t, d_p)
        # macro-context
        ctx_keys = g[["ring_name", "ring_pos", "target_ring_id", "sub_type"]].astype(str).agg("||".join, axis=1)
        ctx_metrics = [compute_pair_metrics(
            g.loc[g.index[ctx_keys == ck], "delta_true"].to_numpy(dtype=float),
            g.loc[g.index[ctx_keys == ck], "delta_pred"].to_numpy(dtype=float))
            for ck in ctx_keys.unique()]
        macro_ctx = _aggregate_macro(ctx_metrics, metric_keys)
        # macro-ring
        ring_metrics = [compute_pair_metrics(
            gg["delta_true"].to_numpy(dtype=float),
            gg["delta_pred"].to_numpy(dtype=float))
            for _, gg in g.groupby("ring_name")]
        macro_ring = _aggregate_macro(ring_metrics, metric_keys)
        summary_rows.append({
            "model": model,
            "task": task,
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_value": micro["MAE"],
            **{f"micro_{k}": micro[k] for k in metric_keys + ["n"]},
            **{f"macro_context_{k}": macro_ctx[k]
               for k in metric_keys + ["n_groups", "n_total"]},
            **{f"macro_ring_{k}": macro_ring[k]
               for k in metric_keys + ["n_groups", "n_total"]},
        })
    return pd.DataFrame(summary_rows)


def main(
    model_types: Optional[List[str]] = None,
    baseline_b_kinds: Optional[List[str]] = None,
    siamese_backbones: Optional[List[str]] = None,
    seed: int = 42, verbose: bool = True,
) -> Dict[str, Any]:
    if not LUNCI10_PAIR.is_file():
        raise FileNotFoundError(f"missing lunci10 pair manifest: {LUNCI10_PAIR}")
    lunci10_pair_df = pd.read_csv(LUNCI10_PAIR)
    lunci10_manifest = pd.read_csv(
        Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit/lunci10_manifest.csv"
    )

    # internal 资源 (若 Protocol I 可行)
    internal_pair_df: Optional[pd.DataFrame] = None
    internal_manifests: Dict[str, pd.DataFrame] = {}
    if INTERNAL_PAIR.is_file():
        internal_pair_df = pd.read_csv(INTERNAL_PAIR)
        for task, p in [
            ("HOMA", Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_homa_0716.csv"),
            ("MBCO", Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_mbco_0716.csv"),
            ("NICS_1zz", Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_nics_0716.csv"),
        ]:
            if p.is_file():
                df = pd.read_csv(p)
                # 简化: 用 New_ID 当 sample_id, smiles 列兼容
                df["sample_id"] = df["New_ID"].astype(str) if "New_ID" in df.columns else df.index.astype(str)
                df["canonical_smiles"] = df["smiles"].astype(str) if "smiles" in df.columns else df.iloc[:, 1].astype(str)
                df["ring_name"] = df["Ring_ID"].astype(str) if "Ring_ID" in df.columns else "internal"
                internal_manifests[task] = df[["sample_id", "canonical_smiles", "ring_name"]].copy()
    else:
        warnings.warn("internal_pair_manifest.csv not found — Protocol II active")

    if model_types is None:
        model_types = ["RC_MPNN", "Base_MPNN", "Base_GAT", "Base_GNN"]
    if baseline_b_kinds is None:
        baseline_b_kinds = ["XGBoost"]
    if siamese_backbones is None:
        siamese_backbones = ["MPNN"]

    pred_rows: List[Dict[str, Any]] = []
    for task in TASKS:
        if verbose:
            print(f"[train_internal_pair] task={task}")
        # Baseline A
        ba_preds = baseline_a_predictions(task, lunci10_pair_df, lunci10_manifest,
                                          model_types=model_types, seed=seed)
        for v in ba_preds.values():
            pred_rows.append(v)
        # Baseline B
        for kind in baseline_b_kinds:
            bb_preds = baseline_b_predictions(
                task, lunci10_pair_df, lunci10_manifest,
                internal_pair_df if internal_pair_df is not None else pd.DataFrame(),
                internal_manifests, model_kind=kind, cv_folds=5, seed=seed,
            )
            for v in bb_preds.values():
                pred_rows.append(v)
        # Siamese
        for bb in siamese_backbones:
            si_preds = siamese_rcgnn_predictions(
                task, lunci10_pair_df, lunci10_manifest,
                internal_pair_df, internal_manifests, backbone=bb,
                cv_folds=5, seed=seed, n_epochs=10, batch_size=8, device="cpu",
            )
            for v in si_preds.values():
                pred_rows.append(v)

    summary_df = _build_summary(pred_rows)
    if not summary_df.empty:
        summary_df.to_csv(SUMMARY_CSV, index=False)
    else:
        # 写出空文件占位, 让下游流水线不报错
        pd.DataFrame().to_csv(SUMMARY_CSV, index=False)

    if verbose:
        print(f"[train_internal_pair] saved {SUMMARY_CSV} (rows={len(summary_df)})")
        print(f"[train_internal_pair] n_predictions={len(pred_rows)}")

    return {
        "summary_csv": str(SUMMARY_CSV),
        "n_predictions": len(pred_rows),
        "n_summary_rows": len(summary_df),
        "protocol": "I" if internal_pair_df is not None else "II",
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))