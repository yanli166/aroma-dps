"""Retrain frozen Fig.3 checkpoints with FIXED graphs.py (atom_on_ring 0-based).

Reuses the exact original training protocol:
  - canonical_splits (sample-level random split, seed)
  - 5-fold CV + early stopping + best_state
  - final model on full train (final_tr/final_va) with early stopping
  - saves best_model.pth to the same paths consumed by load_frozen_models.py

Models to retrain (used by fig4):
  - Layer2 GNN (Base_GNN): seed 42 x 3 tasks
  - Layer3 MPNN label (RC_MPNN): 5 seeds x 3 tasks
  - Layer3 GAT label (RC_GAT): 5 seeds x 3 tasks
Total = 33 checkpoints.

Usage:
  python training/retrain_frozen.py --layer3 --gpu 1
  python training/retrain_frozen.py --layer2 --gpu 2
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

import torch

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "archive/deprecated/code_end"
ORIG_MODELS_ROOT = str(PROJ_ROOT / "unified_models")
for p in (str(PROJ_ROOT), str(CODE_END), ORIG_MODELS_ROOT):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

LAYER2_DIR = CODE_END / "results/layer2_gnn"
LAYER3_DIR = CODE_END / "results/layer3_ring_fixed"
SEEDS = [42, 123, 456, 789, 2024]
N_EPOCHS = 200
PATIENCE = 30


def retrain_layer2(gpu: int):
    """Retrain GNN (Base_GNN) for seed 42, all 3 tasks."""
    from baseline_gnn.code.gnn_train_eval import run_model_on_task, DEFAULT_PARAMS
    from common.tasks import TASKS

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    params = dict(DEFAULT_PARAMS)
    params["n_epochs"] = N_EPOCHS
    params["patience"] = PATIENCE

    seed = 42
    seed_dir = LAYER2_DIR / f"seed_{seed}"
    print(f"\n[Layer2 GNN] seed={seed}, output={seed_dir}", flush=True)
    for task in TASKS:
        t0 = time.time()
        res = run_model_on_task("GNN", task, params, device, str(seed_dir),
                                N_EPOCHS, PATIENCE, seed)
        print(f"  [{task['name']}] test R2={res['test_r2']:.4f} MAE={res['test_mae']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[Layer2 GNN] DONE", flush=True)


def retrain_layer3(gpu: int, model_name: str = "MPNN"):
    """Retrain one layer3 backbone (label encoding) for 5 seeds x 3 tasks."""
    from ring_encoding_ablation.code.ring_train_eval import run_experiment, DEFAULT_PARAMS
    from common.tasks import TASKS

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    params = dict(DEFAULT_PARAMS)
    params["n_epochs"] = N_EPOCHS
    params["patience"] = PATIENCE

    for seed in SEEDS:
        seed_dir = LAYER3_DIR / f"seed_{seed}"
        print(f"\n[Layer3 {model_name}_label] seed={seed}, output={seed_dir}", flush=True)
        for task in TASKS:
            t0 = time.time()
            res = run_experiment(model_name, "label", task, params, device, str(seed_dir),
                                 N_EPOCHS, PATIENCE, seed)
            print(f"  [{task['name']}] test R2={res['test_r2']:.4f} MAE={res['test_mae']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[Layer3 {model_name}_label] DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer2", action="store_true", help="retrain layer2 GNN (seed 42)")
    parser.add_argument("--layer3", action="store_true", help="retrain layer3 label models")
    parser.add_argument("--model", type=str, default="MPNN", help="layer3 backbone (MPNN/GAT)")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.layer2:
        retrain_layer2(args.gpu)
    if args.layer3:
        retrain_layer3(args.gpu, args.model)
    print("\nALL DONE", flush=True)
