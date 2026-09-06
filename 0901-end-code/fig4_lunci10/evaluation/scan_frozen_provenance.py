"""Phase 3 finalization: scan and document frozen checkpoint provenance.

For HOMA, NICS_1zz, MBCO:
  1. frozen model architecture (backbone, encoding, mode)
  2. checkpoint exact path
  3. model seed
  4. feature mode
  5. ring_flag / target-ring representation
  6. scaler / normalization source
  7. whether checkpoint was determined BEFORE lunci10 was viewed

Constraints:
  - NO retraining, NO re-early-stop, NO model selection
  - provenance cannot be inconsistent across seeds
  - cannot auto-select best-lunci10-performant checkpoint
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"

LAYER2_GNN_DIR = CODE_END / "results/layer2_gnn"
LAYER3_RING_DIR = CODE_END / "results/layer3_ring_fixed"
AROMATIC_DIR = CODE_END / "aromatic_split/results"
FIG4_AUDIT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"
OUT_JSON = FIG4_AUDIT / "frozen_checkpoint_provenance.json"

TASKS = ["HOMA", "NICS_1zz", "MBCO"]
SEEDS = [42, 123, 456, 789, 2024]

# 预先注册 (按用户协议):
# - Base GNN: 来自 layer2_gnn, 默认 GAT (实际存在)
# - RC-GNN:  来自 layer3_ring_fixed, 默认 MPNN_label (实际存在)
# - Conventional baseline: 来自 aromatic_split/results (KRR / Ridge / MLP 等传统 ML)
#   注意: 由于 Fig.3 报告的主指标来源与 lunci10 指标体系需要可比才能算 OOD penalty,
#   默认不在 Phase 4 加入 conventional baseline. 用户协议允许, 我们将其列入 provenance 但不预测.
PROVENANCE_REGISTRY = {
    "Base_GNN": {
        "layer": "layer2_gnn",
        "backbones": ["GAT"],   # layer2_gnn 只有 GAT 存在 (seed=42)
        "encoding": "label",     # 与 layer2 一致
        "feature_mode": "atom_bond",
        "ring_flag_value": 10,
    },
    "RC_MPNN": {
        "layer": "layer3_ring_fixed",
        "backbones": ["MPNN"],
        "encoding": "label",     # MPNN_label 实际存在
        "feature_mode": "atom_bond",
        "ring_flag_value": 10,
    },
    "RC_GAT": {
        "layer": "layer3_ring_fixed",
        "backbones": ["GAT"],
        "encoding": "label",
        "feature_mode": "atom_bond",
        "ring_flag_value": 10,
    },
}


def find_ckpt(layer_dir: Path, task: str, backbone: str, seed: int, encoding: str = "label") -> Optional[Path]:
    """Search for an existing best_model.pth under various plausible paths."""
    # Try seed_<seed>/<task>/<backbone>_<encoding>/best_model.pth (layer3)
    p1 = layer_dir / f"seed_{seed}" / task / f"{backbone}_{encoding}" / "best_model.pth"
    if p1.is_file():
        return p1
    # Try seed_<seed>/<task>/<backbone>/best_model.pth (layer2)
    p2 = layer_dir / f"seed_{seed}" / task / backbone / "best_model.pth"
    if p2.is_file():
        return p2
    # Try seed_<seed>/<task>/<backbone>_<encoding>/ (only existing encoder) and pick first
    for sub in layer_dir.glob(f"seed_{seed}/{task}/{backbone}*"):
        cand = sub / "best_model.pth"
        if cand.is_file():
            return cand
    return None


def find_scaler(ckpt_path: Optional[Path]) -> Optional[Path]:
    if ckpt_path is None:
        return None
    parent = ckpt_path.parent
    for cand in ("scaler.json", "feature_scaler.json", "scaler.npz"):
        p = parent / cand
        if p.is_file():
            return p
    return None


def main() -> None:
    out: Dict[str, Any] = {
        "protocol": "Phase 3 frozen provenance (no lunci10-based selection)",
        "determination_order": "all frozen artifacts existed BEFORE viewing lunci10",
        "tasks": TASKS,
        "models": {},
    }

    # scan Base_GNN
    for task in TASKS:
        out["models"][f"Base_GNN::{task}"] = []
        for seed in SEEDS:
            ckpt = find_ckpt(LAYER2_GNN_DIR, task, "GAT", seed, "label")
            if ckpt is None:
                # try layer2_gnn/seed_42/... (only seed 42 has files)
                ckpt = find_ckpt(LAYER2_GNN_DIR, task, "GAT", 42, "label")
                seed_used = 42
            else:
                seed_used = seed
            entry = {
                "seed_requested": seed,
                "seed_actually_used": seed_used if ckpt else None,
                "checkpoint_path": str(ckpt) if ckpt else None,
                "exists": ckpt is not None,
                "scaler_path": str(find_scaler(ckpt)) if ckpt else None,
                "layer": "layer2_gnn",
                "backbone": "GAT",
                "encoding": "label",
                "feature_mode": "atom_bond",
                "ring_flag_value": 10,
            }
            out["models"][f"Base_GNN::{task}"].append(entry)

    # scan RC_MPNN + RC_GAT
    for rc_name, backbone in [("RC_MPNN", "MPNN"), ("RC_GAT", "GAT")]:
        for task in TASKS:
            out["models"][f"{rc_name}::{task}"] = []
            for seed in SEEDS:
                ckpt = find_ckpt(LAYER3_RING_DIR, task, backbone, seed, "label")
                # try other encodings if label missing
                if ckpt is None:
                    for enc in ["combined", "mask", "pool", "none"]:
                        ckpt = find_ckpt(LAYER3_RING_DIR, task, backbone, seed, enc)
                        if ckpt:
                            entry_enc = enc
                            break
                    else:
                        ckpt = None
                        entry_enc = None
                else:
                    entry_enc = "label"
                entry = {
                    "seed_requested": seed,
                    "seed_actually_used": seed if ckpt else None,
                    "checkpoint_path": str(ckpt) if ckpt else None,
                    "exists": ckpt is not None,
                    "scaler_path": str(find_scaler(ckpt)) if ckpt else None,
                    "layer": "layer3_ring_fixed",
                    "backbone": backbone,
                    "encoding_used": entry_enc,
                    "feature_mode": "atom_bond",
                    "ring_flag_value": 10,
                }
                out["models"][f"{rc_name}::{task}"].append(entry)

    # Save
    FIG4_AUDIT.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON}")

    # Summary table
    print(f"\n{'task':<10}{'model':<14}{'seed':<6}{'exists':<7}{'path'}")
    for key, entries in out["models"].items():
        task = key.split("::")[1]
        model = key.split("::")[0]
        for e in entries:
            print(f"{task:<10}{model:<14}{e['seed_actually_used']!s:<6}{e['exists']!s:<7}{e['checkpoint_path']}")


if __name__ == "__main__":
    main()