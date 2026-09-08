"""Phase 5 (novelty): 基于 overlap_audit.csv 把 lunci10 分子分到 A/B/C/D 四类, 并报告
近邻 Tanimoto 相似度 / ring family 误差 / substituent 误差.

Category mapping:
  - Category A: seen scaffold + seen substituent (但分子未被见过)
  - Category B: unseen scaffold + seen substituent
  - Category C: seen scaffold + unseen substituent
  - Category D: unseen scaffold + unseen substituent

主要产物:
  - novelty_category_summary.csv      (per task per category: N, MAE)
  - nearest_neighbor_analysis.csv     (per molecule: Tanimoto nearest train neighbor)
  - ring_family_error.csv             (38 ring_name × 误差分布)
  - substituent_error.csv             (30 substituent × 误差分布)
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
for p in (PROJ_ROOT, CODE_END):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(_PROJ_ROOT) / "0901-end-code/fig4_lunci10"
AUDIT_OUT = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/00_audit"
MANIFEST_PATH = AUDIT_OUT / "lunci10_manifest.csv"
OVERLAP_PATH = AUDIT_OUT / "lunci10_overlap_audit.csv"
PRED_CSV = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/01_external_absolute/lunci10_absolute_predictions.csv"

OUT_DIR = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/02_novelty"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CATEGORY_SUMMARY_CSV = OUT_DIR / "novelty_category_summary.csv"
NEAREST_NEIGHBOR_CSV = OUT_DIR / "nearest_neighbor_analysis.csv"
RING_FAMILY_ERR_CSV = OUT_DIR / "ring_family_error.csv"
SUBSTITUENT_ERR_CSV = OUT_DIR / "substituent_error.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
RING_NAMES_EXPECTED = 38
SUBSTITUENTS_EXPECTED = 30


def classify_novelty(row: pd.Series) -> str:
    """依据 overlap_audit flags 输出 A/B/C/D.

    scaffold_seen ∈ {True, False}, substituent_seen ∈ {True, False}
    molecule_seen = exact_molecule_seen (如有), 否则 NaN — A 类定义为「分子未见过」
    """
    sc = bool(row.get("scaffold_seen", False))
    su = bool(row.get("substituent_seen", False))
    em = bool(row.get("exact_molecule_seen", False))
    # Category A: scaffold_seen AND substituent_seen (and not exact_molecule_seen)
    if sc and su and not em:
        return "A"
    if (not sc) and su:
        return "B"
    if sc and (not su):
        return "C"
    return "D"


def load_internal_smiles_fingerprints(
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], np.ndarray]:
    """读取内部训练数据 (HOMA/MBCO/NICS) canonical SMILES, 计算 Morgan fingerprint.
    仅作为近邻查找的 corpus — 不在 lunci10 上做任何 fit / 调参.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    paths = [
        Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_homa_0716.csv",
        Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_mbco_0716.csv",
        Path(_PROJ_ROOT) / "archive/deprecated/code_end/data1_end/collet_nics_0716.csv",
    ]
    smis: List[str] = []
    for p in paths:
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        smi_col = next((c for c in ("canonical_smiles", "smiles", "SMILES", "mol") if c in df.columns), None)
        if smi_col is None:
            continue
        for s in df[smi_col].dropna().astype(str):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                smis.append(Chem.MolToSmiles(m))
    smis = list(dict.fromkeys(smis))  # 去重保序
    fps: List[np.ndarray] = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        fps.append(np.asarray(bv, dtype=np.uint8))
    if not fps:
        return smis, np.zeros((0, 2048), dtype=np.uint8)
    return smis, np.stack(fps, axis=0)


def tanimoto_nearest(
    fp: np.ndarray,
    bank: np.ndarray,
    bank_smis: List[str],
) -> Tuple[float, str]:
    """查询 fp 与 bank 中所有 fp 的 Tanimoto 相似度, 返回 (max_sim, neighbor_smiles)."""
    if bank.size == 0 or fp.sum() == 0:
        return float("nan"), ""
    # fp, bank 为 0/1 bit vectors; Tanimoto = |A∩B| / |A∪B|
    a = fp.astype(np.int32)
    b = bank.astype(np.int32)
    inter = np.bitwise_and(b, a).sum(axis=1).astype(np.float32)
    union = np.bitwise_or(b, a).sum(axis=1).astype(np.float32) + 1e-12
    sims = inter / union
    j = int(np.argmax(sims))
    return float(sims[j]), bank_smis[j]


def _safe_metric(df: pd.DataFrame, value_col: str, pred_col: str) -> Optional[float]:
    """MAE 计算, NaN-safe."""
    sub = df[[value_col, pred_col]].dropna()
    if sub.empty:
        return None
    err = (sub[pred_col] - sub[value_col]).astype(float).abs()
    return float(err.mean())


def main(verbose: bool = True) -> Dict[str, Any]:
    # ---- 0. 读数据 ----
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)

    if not OVERLAP_PATH.is_file():
        raise FileNotFoundError(f"missing overlap audit: {OVERLAP_PATH}; "
                                "run audit_overlap.py first")
    overlap = pd.read_csv(OVERLAP_PATH)

    # 合并: overlap 与 manifest 按 sample_id / canonical_smiles 对齐
    if "sample_id" in overlap.columns and "sample_id" in manifest.columns:
        merged = manifest.merge(overlap, on="sample_id", how="left", suffixes=("", "_ov"))
    else:
        merged = manifest.merge(overlap, on="canonical_smiles", how="left", suffixes=("", "_ov"))

    # 兼容 audit_overlap 输出的列名
    flag_cols = ["exact_molecule_seen", "scaffold_seen", "substituent_seen", "ring_family_seen"]
    for fc in flag_cols:
        if fc not in merged.columns:
            merged[fc] = False

    # ---- 1. 分类 A/B/C/D ----
    merged["novelty_category"] = merged.apply(classify_novelty, axis=1)

    # ---- 2. 近邻 Tanimoto ----
    if verbose:
        print("[novelty] building internal Morgan fingerprint bank …")
    bank_smis, bank_fp = load_internal_smiles_fingerprints()
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    nn_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        smi = str(row.get("canonical_smiles", "") or "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            nn_rows.append({
                "sample_id": row.get("sample_id", ""),
                "canonical_smiles": smi,
                "novelty_category": row["novelty_category"],
                "nn_tanimoto": float("nan"),
                "neighbor_smiles": "",
            })
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp = np.asarray(bv, dtype=np.uint8)
        sim, neigh = tanimoto_nearest(fp, bank_fp, bank_smis)
        nn_rows.append({
            "sample_id": row.get("sample_id", ""),
            "canonical_smiles": smi,
            "novelty_category": row["novelty_category"],
            "nn_tanimoto": sim,
            "neighbor_smiles": neigh,
        })

    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(NEAREST_NEIGHBOR_CSV, index=False)

    # ---- 3. 读绝对预测, 拼成 [merged × pred × truth] ----
    if PRED_CSV.is_file():
        pred = pd.read_csv(PRED_CSV)
        # pred columns: sample_id, task, model, ring_name, sub_name, y_true, y_pred, abs_err
        # 取 final RC-GNN (RC_MPNN) 作为主模型
        pred_main = pred[pred["model"] == "RC_MPNN"].copy()
        if pred_main.empty:
            pred_main = pred.copy()
        # 合并到 merged
        merged_pred = merged.merge(
            pred_main[["sample_id", "task", "y_pred", "abs_err"]],
            on="sample_id", how="left", suffixes=("", "_pred"),
        )
    else:
        warnings.warn(f"{PRED_CSV} missing — category MAE will be NaN")
        merged_pred = merged.copy()
        merged_pred["y_pred"] = float("nan")
        merged_pred["abs_err"] = float("nan")

    # ---- 4. 类别 A/B/C/D × task MAE 汇总 ----
    cat_rows: List[Dict[str, Any]] = []
    for cat in ("A", "B", "C", "D"):
        sub = merged_pred[merged_pred["novelty_category"] == cat]
        n_total = int(len(sub))
        for task in TASKS:
            truth_col = "NICS_ZZ" if task == "NICS_1zz" else task
            sub_t = sub.copy()
            sub_t["__truth__"] = pd.to_numeric(sub_t.get(truth_col), errors="coerce")
            mae = _safe_metric(sub_t, "__truth__", "y_pred")
            cat_rows.append({
                "category": cat,
                "task": task,
                "n_total": n_total,
                "n_with_truth": int(sub_t["__truth__"].notna().sum()),
                "MAE": mae if mae is not None else float("nan"),
                "note": "" if mae is not None else "N too small / no truth",
            })
        if verbose:
            n_t = cat_rows[-len(TASKS)]["n_with_truth"]
            print(f"  Category {cat}: n_total={n_total}, n_with_truth~{n_t}")
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(CATEGORY_SUMMARY_CSV, index=False)

    # ---- 5. ring family × error ----
    # 38 ring_name × 误差分布 (mean, std, median, q25, q75, n)
    if "ring_name" in merged_pred.columns:
        # 过滤空值
        ring_df = merged_pred.dropna(subset=["abs_err"]).copy()
        ring_df = ring_df[ring_df["ring_name"].astype(str).str.len() > 0]
        ring_agg = (
            ring_df.groupby("ring_name")["abs_err"]
            .agg(["count", "mean", "std", "median",
                  lambda s: float(np.quantile(s.dropna(), 0.25)) if s.notna().any() else float("nan"),
                  lambda s: float(np.quantile(s.dropna(), 0.75)) if s.notna().any() else float("nan")])
            .reset_index()
        )
        ring_agg.columns = ["ring_name", "n", "mean_abs_err", "std_abs_err",
                            "median_abs_err", "q25_abs_err", "q75_abs_err"]
        ring_agg = ring_agg.sort_values("mean_abs_err", ascending=False)
        ring_agg.to_csv(RING_FAMILY_ERR_CSV, index=False)
        if verbose:
            print(f"[novelty] ring family: {len(ring_agg)} unique ring_name "
                  f"(expected={RING_NAMES_EXPECTED})")
    else:
        pd.DataFrame(columns=["ring_name", "n", "mean_abs_err"]).to_csv(
            RING_FAMILY_ERR_CSV, index=False
        )

    # ---- 6. substituent × error ----
    if "sub_name" in merged_pred.columns:
        sub_df = merged_pred.dropna(subset=["abs_err"]).copy()
        sub_df = sub_df[sub_df["sub_name"].astype(str).str.len() > 0]
        sub_agg = (
            sub_df.groupby("sub_name")["abs_err"]
            .agg(["count", "mean", "std", "median",
                  lambda s: float(np.quantile(s.dropna(), 0.25)) if s.notna().any() else float("nan"),
                  lambda s: float(np.quantile(s.dropna(), 0.75)) if s.notna().any() else float("nan")])
            .reset_index()
        )
        sub_agg.columns = ["sub_name", "n", "mean_abs_err", "std_abs_err",
                           "median_abs_err", "q25_abs_err", "q75_abs_err"]
        sub_agg = sub_agg.sort_values("mean_abs_err", ascending=False)
        sub_agg.to_csv(SUBSTITUENT_ERR_CSV, index=False)
        if verbose:
            print(f"[novelty] substituent: {len(sub_agg)} unique sub_name "
                  f"(expected={SUBSTITUENTS_EXPECTED})")
    else:
        pd.DataFrame(columns=["sub_name", "n", "mean_abs_err"]).to_csv(
            SUBSTITUENT_ERR_CSV, index=False
        )

    if verbose:
        print(f"[novelty] saved {CATEGORY_SUMMARY_CSV}")
        print(f"[novelty] saved {NEAREST_NEIGHBOR_CSV}")
        print(f"[novelty] saved {RING_FAMILY_ERR_CSV}")
        print(f"[novelty] saved {SUBSTITUENT_ERR_CSV}")

    return {
        "category_summary_csv": str(CATEGORY_SUMMARY_CSV),
        "nearest_neighbor_csv": str(NEAREST_NEIGHBOR_CSV),
        "ring_family_error_csv": str(RING_FAMILY_ERR_CSV),
        "substituent_error_csv": str(SUBSTITUENT_ERR_CSV),
        "n_unique_ring_name": int(merged_pred["ring_name"].astype(str).nunique())
            if "ring_name" in merged_pred.columns else 0,
        "n_unique_sub_name": int(merged_pred["sub_name"].astype(str).nunique())
            if "sub_name" in merged_pred.columns else 0,
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))
