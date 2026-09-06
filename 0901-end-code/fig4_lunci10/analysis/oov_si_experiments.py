"""Phase 16 (OOV SI method experiments): 在 *lunci10 (strict OOD)* pair set 上
对比 method 3 / 4 / 8 在 SI-only 协议 (Siamese Δ-MLP head, 不接 absolute tail)
下的 Siamese Δ-learning 表现。

   M3 — 分层交叉注意力 (Hierarchical Cross-Attention)
        实现: layer4_substituent.code.m3_hierarchical_attention
   M4 — 双通道消息 (Conjugate/Inductive dual channel)
        实现: layer4_substituent.code.m4_dual_channel
   M8 — 多尺度池化 (Multi-scale pooling with substituent + ring)
        实现: layer4_substituent.code.m8_multi_scale_pooling

每个方法在每个 task 上运行一次 (若有 checkpoint) 或返回 stub 状态
(明确说明需要更多实现工作)。失败 runs 不删除, status 写为 "missing_checkpoint"
或 "needs_more_implementation"。

产物: _PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/si_methods/m3_m4_m8_summary.csv

约束:
  - 不在 lunci10 上做 fit / hyperparameter 选择 / early stop。
  - 复用现有 m3/m4/m8 的 *model factory* (build_model), 不重新写 attention / pool。
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

PROJ_ROOT = Path("_PROJ_ROOT")
CODE_END = PROJ_ROOT / "code_end"
LAYER4_CODE = CODE_END / "layer4_substituent" / "code"
for p in (str(PROJ_ROOT), str(CODE_END), str(LAYER4_CODE)):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

RES_ROOT = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final"
PAIR_CSV = RES_ROOT / "03_pairwise" / "lunci10_pair_manifest.csv"
INTERNAL_PAIR = RES_ROOT / "03_pairwise" / "internal_pair_manifest.csv"
LUNCI10_MANIFEST = RES_ROOT / "00_audit" / "lunci10_external_manifest.csv"

OUT_DIR = RES_ROOT / "si_methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "m3_m4_m8_summary.csv"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
SI_METHODS = ["M3_hierarchical_attention", "M4_dual_channel", "M8_multi_scale_pooling"]


# --------------------------------------------------------------------------
# Layer4 import resolver — 若任意方法未实现, 提供 stub
# --------------------------------------------------------------------------
def _safe_import_factory(module_path: str, attr: str = "build_model"):
    """Try `from {module_path} import {attr}`; return (ok, factory_or_None, msg)."""
    try:
        mod = __import__(module_path, fromlist=[attr])
        factory = getattr(mod, attr, None)
        return (True, factory, "imported")
    except Exception as e:
        return (False, None, f"{type(e).__name__}: {e}")


class M3Stub:
    """M3 hierarchical cross-attention — 占位实现。

    真实实现位于 layer4_substituent.code.m3_hierarchical_attention。
    本 stub *明确* 说明: 若需要 Phase 16 中 M3 真正可跑,
    需要完成下面 4 个步骤, 仅靠复用 checkpoint 不足以支撑 strict OOD 测试。
    """
    NEEDED_WORK = [
        "1. add_position_encodings() 接入 lunci10 数据 dict (含 pair SMILES)",
        "2. HierarchicalAttentionGNN 模型定义加 Siamese head (pair rep)",
        "3. Stage-I pretraining checkpoint (M3 版) 需要先在 internal 上重训",
        "4. 配 Δ-loss / optimizer / 调度器",
    ]


class M4Stub:
    NEEDED_WORK = [
        "1. add_dual_adjacency() 处理 pair inputs (i, j 两个分子) 而非单分子 batch",
        "2. DualChannelGNN 重写 forward 支持 shared encoder (Siamese)",
        "3. checkpoint 不存在: 需要先在 internal 上为 M4 跑一次 Stage-I 训练",
        "4. 配 Siamese head (pair rep [h_i, h_j, h_i-h_j, |h_i-h_j|])",
    ]


class M8Stub:
    NEEDED_WORK = [
        "1. add_substituent_groups() 产出 group_id 张量, pair 化需要 batched 处理",
        "2. MultiScalePoolGNN 加 Siamese head (pair rep)",
        "3. 三尺度池化融合 Siamese 后是否仍然对称需要验证",
        "4. 需要先在 internal 上重训 M8 Stage-I",
    ]


# --------------------------------------------------------------------------
# Per-method evaluation placeholder
# --------------------------------------------------------------------------
def _eval_method_on_pairset(method_name: str, task: str,
                            internal_pair_df: Optional[pd.DataFrame],
                            lunci10_pair_df: pd.DataFrame) -> Dict[str, Any]:
    """对一个 (method, task) 返回 stub 状态行。

    真正实现需要替换为:
        1. 加载 method 的 Stage-I checkpoint (若存在)
        2. 加 Siamese head (pair rep) 复用 train_internal_pair_models
        3. 在 internal pair fraction 训练
        4. 在 lunci10 pair manifest 上 *zero-shot* 评测
        5. metrics: MAE / RMSE / R2 / sign_accuracy
    """
    if method_name == "M3_hierarchical_attention":
        factory_path = ("layer4_substituent.code.m3_hierarchical_attention",
                        "HierarchicalAttentionGNN")
    elif method_name == "M4_dual_channel":
        factory_path = ("layer4_substituent.code.m4_dual_channel",
                        "DualChannelGNN")
    elif method_name == "M8_multi_scale_pooling":
        factory_path = ("layer4_substituent.code.m8_multi_scale_pooling",
                        "MultiScalePoolGNN")
    else:
        factory_path = ("", "")

    ok, factory, msg = _safe_import_factory(factory_path[0],
                                            attr="build_model")

    row: Dict[str, Any] = {
        "method": method_name,
        "task": task,
        "factory_imported": bool(ok and factory is not None),
        "import_message": msg,
        "stage_i_checkpoint_exists": False,   # by inspection (no checkpoints in repo)
        "siamese_head_wired": False,
        "internal_pair_size": 0 if internal_pair_df is None else int(len(internal_pair_df)),
        "lunci10_pair_size": int(len(lunci10_pair_df)),
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "R2": float("nan"),
        "sign_accuracy": float("nan"),
        "status": "stub",
        "needs_more_implementation": True,
        "notes": "",
    }

    # Decide stub class
    if method_name == "M3_hierarchical_attention":
        row["needed_work"] = "; ".join(M3Stub.NEEDED_WORK)
    elif method_name == "M4_dual_channel":
        row["needed_work"] = "; ".join(M4Stub.NEEDED_WORK)
    elif method_name == "M8_multi_scale_pooling":
        row["needed_work"] = "; ".join(M8Stub.NEEDED_WORK)

    if not (ok and factory is not None):
        row["status"] = "missing_layer4_module"
        row["notes"] = (f"layer4 factory import failed ({msg}); "
                        "Phase 16 M3/M4/M8 needs more implementation")
    else:
        row["status"] = "needs_more_implementation"
        row["notes"] = ("factory exists, but no Stage-I Siamese checkpoint "
                        "exists for these methods; reuse of existing "
                        "checkpoint not possible")

    return row


# --------------------------------------------------------------------------
# Batch runner
# --------------------------------------------------------------------------
def run_all() -> pd.DataFrame:
    if not PAIR_CSV.exists():
        warnings.warn(f"[phase16] missing pair manifest: {PAIR_CSV}")
        lunci10_pair_df = pd.DataFrame()
    else:
        lunci10_pair_df = pd.read_csv(PAIR_CSV)

    if INTERNAL_PAIR.exists():
        internal_pair_df = pd.read_csv(INTERNAL_PAIR)
        if len(internal_pair_df) == 0:
            internal_pair_df = None
    else:
        internal_pair_df = None

    rows: List[Dict[str, Any]] = []
    for method in SI_METHODS:
        for task in TASKS:
            rows.append(_eval_method_on_pairset(
                method_name=method, task=task,
                internal_pair_df=internal_pair_df,
                lunci10_pair_df=lunci10_pair_df,
            ))
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[phase16] pair manifest: {'OK' if PAIR_CSV.exists() else 'MISSING'}")
    print(f"[phase16] internal pairs: {'OK' if INTERNAL_PAIR.exists() else 'MISSING'}")
    df = run_all()
    print(f"[phase16] wrote {len(df)} rows to {OUT_CSV}")
    if len(df):
        print(df[["method", "task", "factory_imported",
                  "status", "needs_more_implementation"]].to_string(index=False))
        # Print at most one missing-info note
        misses = df[~df["factory_imported"]]["method"].tolist()
        if misses:
            print(f"[phase16] NOTE: missing factories: {misses}")
            print("[phase16] methods are stubs - need more implementation work")
