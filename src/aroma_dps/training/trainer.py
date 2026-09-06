"""
Trainer: unified training loop and early stopping.

Consolidates:
- 0831-end-code/common/train_eval.py (train_custom_model, eval_custom_model, train_pyg_model, eval_pyg_model)
- 0831-end-code/stage*/code/*_fit_final() (4 duplicate implementations)
- 0831-end-code/stage6/code/_train_loop()

The canonical implementation is in 0831-end-code/common/train_eval.py.
This module re-exports it for the publication package.
"""
# Re-export from 0831-end-code
try:
    from common.train_eval import (
        train_custom_model,
        eval_custom_model,
        train_pyg_model,
        eval_pyg_model,
    )
except ImportError:
    pass

# Re-export provenance recorder (from p0_verification, will be moved here)
try:
    from p0_verification.provenance_recorder import ProvenanceRecorder
except ImportError:
    pass


def set_full_seed(seed: int):
    """Set all random seeds for reproducibility.

    Consolidates the 11 duplicate set_full_seed implementations.
    Includes cudnn.deterministic = True for full reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
