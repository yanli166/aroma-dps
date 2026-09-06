"""Phase 4 full run on one GPU: split tasks to parallelize across GPUs.

GPU 0 -> HOMA + NICS_1zz
GPU 2 -> MBCO
"""
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'code_end'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'unified_models'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, '0901-end-code', 'fig4_lunci10'))

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import evaluation.run_external_absolute_v2 as v2

# Set tasks from CLI args or env
SUBSET = os.environ.get("PHASE4_TASKS", "HOMA,NICS_1zz,MBCO")
TASKS_SUBSET = [t.strip() for t in SUBSET.split(",") if t.strip()]

# Optional: subset of models via env
SUBSET_MODELS = os.environ.get("PHASE4_MODELS", "Base_GNN,RC_MPNN,RC_GAT")
MODELS_SUBSET = {k: v for k, v in v2.MODELS.items() if k in [m.strip() for m in SUBSET_MODELS.split(",")]}

print(f"[phase4 subset] tasks={TASKS_SUBSET}", flush=True)
print(f"[phase4 subset] models={list(MODELS_SUBSET.keys())}", flush=True)

v2.TASKS = TASKS_SUBSET
v2.MODELS = MODELS_SUBSET


if __name__ == "__main__":
    v2.main()