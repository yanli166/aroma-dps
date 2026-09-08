"""Phase 14 (Pretraining transfer control): 对比 random-init Siamese 与
Stage-I-pretrained Siamese encoder, 在不同 internal pair training fraction 下
(100% / 50% / 20% / 10%) 训练并对同一 lunci10 pair manifest 评测.

设计要点:
  - 内层 base encoder 复用 RC-MPNN / Base_MPNN 抽象; Siamese head (pair rep
    [h_i, h_j, h_i - h_j, |h_i - h_j|]) 与 train_internal_pair_models.py
    中的 Siamese 路径完全一致。
  - Stage-I pretraining checkpoint 指 Phase 1 (绝对模型) 在 internal 训练
    上训练的 best.pt; random-init 则跳过 load_state_dict。
  - 由于目前 (2026-09) 只有 lunci10 这一组 OOD 配对可评测, learning curve
    退化为 *within-lunci10 adaptation*: 按 context / scaffold 分层切分
    train/test, 并标记 NOT_strict_external。
  - 产物:
      _PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/si_pretraining/pretraining_transfer.csv
  - 失败 runs 不会删除 (由 run_registry 记录 status=fail)。

约束:
  - 不在 lunci10 上做 hyperparameter 选择 / 早停。
  - 若 internal_pair 可用且足够, 仍以 internal pair fraction 作为补充,
    并把 within-lunci10 adaptation 标为辅助 evidence。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

PROJ_ROOT = Path(_PROJ_ROOT)
CODE_END = PROJ_ROOT / "archive/deprecated/code_end"
for p in (str(PROJ_ROOT), str(CODE_END), str(CODE_END / "layer4_substituent")):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = PROJ_ROOT / "0901-end-code" / "fig4_lunci10"
PAIR_DIR = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final" / "03_pairwise"
FEASIBILITY_JSON = PAIR_DIR / "internal_pair_feasibility.json"
INTERNAL_PAIR = PAIR_DIR / "internal_pair_manifest.csv"
LUNCI10_PAIR = PAIR_DIR / "lunci10_pair_manifest.csv"
LUNCI10_MANIFEST = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final" / "00_audit" / "lunci10_external_manifest.csv"

OUT_DIR = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final" / "si_pretraining"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "pretraining_transfer.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
FRACTIONS = [1.00, 0.50, 0.20, 0.10]
SEEDS = [42, 123, 456]  # 模型随机种子; split 固定
PRETRAIN_VARIANTS = ["random_init", "stage_i_pretrained"]


# --------------------------------------------------------------------------
# Adapter helpers — 仅在缺实现时写最小占位
# --------------------------------------------------------------------------
def _maybe_load_feasibility() -> Dict[str, Any]:
    """读取 build_pairs 阶段写的 feasibility, 决定 learning curve
    是 *internal pair fraction* 还是 *within-lunci10 adaptation*。"""
    if not FEASIBILITY_JSON.exists():
        return {"can_build_internal_pair": False,
                "reason": "feasibility json not produced yet"}
    try:
        return json.loads(FEASIBILITY_JSON.read_text())
    except Exception:
        return {"can_build_internal_pair": False,
                "reason": "feasibility json unreadable"}


def _load_lunci10_pairs() -> pd.DataFrame:
    if not LUNCI10_PAIR.exists():
        return pd.DataFrame()
    df = pd.read_csv(LUNCI10_PAIR)
    return df


def _load_internal_pairs() -> Optional[pd.DataFrame]:
    if not INTERNAL_PAIR.exists():
        return None
    df = pd.read_csv(INTERNAL_PAIR)
    return df if len(df) > 0 else None


def _split_within_lunci10(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """within-lunci10 适配切分: 按 (ring_name, target_ring_id) scaffold 分层
    80/20 split。注意: 这并非 strict external, 仅作 auxiliary evidence。
    """
    from sklearn.model_selection import GroupShuffleSplit

    if "target_ring_id" in df.columns:
        groups = df["target_ring_id"].astype(str).fillna("NA").values
    elif "ring_name" in df.columns:
        groups = df["ring_name"].astype(str).fillna("NA").values
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        cut = int(0.8 * len(df))
        return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


# --------------------------------------------------------------------------
# Per-fraction Siamese training (placeholder construction — 代码层完整)
# --------------------------------------------------------------------------
def _train_siamese_at_fraction(
    task: str,
    fraction: float,
    pretrained: bool,
    seed: int,
    internal_pair_df: Optional[pd.DataFrame],
    lunci10_df: Optional[pd.DataFrame],
    feasibility: Dict[str, Any],
) -> Dict[str, Any]:
    """在 (task, fraction, pretrained, seed) 网格上训练 Siamese Δ-MLP.

    返回 run-level dict (不含真实 metrics, 因为本脚本为代码占位,
    不实际执行训练)。若 *internal_pair_df* 不足以切分到该 fraction,
    或 feasibility 表明无法构建 internal pair, 则自动退到
    within-lunci10 adaptation, 并在 evidence_kind 中注明。
    """
    run: Dict[str, Any] = {
        "task": task,
        "fraction": float(fraction),
        "pretrained": bool(pretrained),
        "model_seed": int(seed),
        "split_seed": 42,
        "evidence_kind": "",
        "internal_pair_size": 0 if internal_pair_df is None else int(len(internal_pair_df)),
        "lunci10_pair_size": 0 if lunci10_df is None else int(len(lunci10_df)),
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "R2": float("nan"),
        "sign_accuracy": float("nan"),
        "n_test": 0,
        "status": "stub",
        "strict_external": False,
        "notes": "",
    }

    can_internal = bool(feasibility.get("can_build_internal_pair", False)) \
        and internal_pair_df is not None and len(internal_pair_df) >= 20

    if can_internal:
        run["evidence_kind"] = "internal_pair_fraction"
        run["strict_external"] = True
        # Real implementation would:
        #   1) Build Siamese encoder; optionally load Stage-I checkpoint.
        #   2) Train on fraction*internal_pair_df with 5-fold CV.
        #   3) Score on lunci10_df once.
        run["notes"] = "real Siamese training deferred to executor"
    else:
        # 仅 lunci10 配对可用 → within-lunci10 adaptation
        run["evidence_kind"] = "within_lunci10_adaptation"
        run["strict_external"] = False
        run["notes"] = ("NOT strict external; only lunci10 pairs available. "
                        "Split by (ring_name, target_ring_id) scaffold.")

    return run


# --------------------------------------------------------------------------
# Batch runner
# --------------------------------------------------------------------------
def run_all(fractions: List[float] = FRACTIONS,
            seeds: List[int] = SEEDS,
            tasks: List[str] = TASKS,
            pretrain_variants: List[str] = PRETRAIN_VARIANTS) -> pd.DataFrame:
    feasibility = _maybe_load_feasibility()
    lunci10_df = _load_lunci10_pairs()
    internal_pair_df = _load_internal_pairs()

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for variant in pretrain_variants:
            pretrained = (variant == "stage_i_pretrained")
            for frac in fractions:
                for seed in seeds:
                    row = _train_siamese_at_fraction(
                        task=task, fraction=frac,
                        pretrained=pretrained, seed=seed,
                        internal_pair_df=internal_pair_df,
                        lunci10_df=(lunci10_df if len(lunci10_df) > 0 else None),
                        feasibility=feasibility,
                    )
                    row["variant"] = variant
                    rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[phase14] feasibility = {_maybe_load_feasibility()}")
    print(f"[phase14] lunci10 pairs: "
          f"{'OK' if LUNCI10_PAIR.exists() else 'MISSING'}")
    print(f"[phase14] internal pairs: "
          f"{'OK' if INTERNAL_PAIR.exists() else 'MISSING'}")
    df = run_all(fractions=args.fractions,
                 seeds=args.seeds,
                 tasks=args.tasks)
    print(f"[phase14] wrote {len(df)} rows to {OUT_CSV}")
    if len(df):
        print(df.groupby(["variant", "fraction"]).size().rename("n_runs"))
