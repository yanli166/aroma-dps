"""
aroma_dps: Target-ring-conditioned aromaticity prediction.

Implemented here (authoritative):
- chemistry/      ring_mapping (0-based target ring), ring_family
- data/           splits (get_final_splits, SPLIT_SEED=2026), tasks
- featurization/  graphs, graph_data, features (vendored from 0831-end-code)
- models/         backbones (vendored from unified_models), readout,
                  ring_conditioned, pyg_backbones (DMPNN, needs torch_geometric)
- training/       trainer (vendored from 0831-end-code/common/train_eval.py)
- evaluation/     metrics
- inference/      ring_prediction (released checkpoints + ΔA conventions)

Still pending: inference/reaction_prediction (Fig.6 logic is still in the
pipeline scripts), data/datasets|manifests|scalers, evaluation/ood|uncertainty.

Fig.3/Fig.4 reproduction still runs from 0831-end-code/ and 0901-end-code/;
those trees are frozen snapshots that produced the reported numbers and are not
rewritten by imports from this package. The superseded code_end/ lives under
archive/deprecated/.
"""

__version__ = "0.1.0"

# --- Task naming: canonical MCBO with alias compatibility ---
TASK_ALIASES = {
    "MBCO": "MCBO",
    "nMCBO": "MCBO",
    "mcbo": "MCBO",
    "mbco": "MCBO",
}

CANONICAL_TASKS = ["HOMA", "NICS_1zz", "MCBO"]


def canonicalize_task_name(name: str) -> str:
    """Map legacy task names to canonical names."""
    return TASK_ALIASES.get(name, name)
