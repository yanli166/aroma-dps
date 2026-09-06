"""
Graph construction: single source of truth.

Consolidates logic from:
- unified_models/common/graphs.py::process_and_save_data (upstream source)
- 0831-end-code/common/graph_data.py (load_adj_format, load_pyg_format)

The canonical implementation is process_and_save_data in unified_models.
This module re-exports it for the publication package.
"""
# Re-export from the upstream source (no code duplication)
try:
    from unified_models.common.graphs import process_and_save_data
except ImportError:
    # When unified_models is not on path, provide a clear error
    def process_and_save_data(*args, **kwargs):
        raise ImportError(
            "process_and_save_data is in unified_models.common.graphs. "
            "Ensure unified_models is on PYTHONPATH or installed."
        )
