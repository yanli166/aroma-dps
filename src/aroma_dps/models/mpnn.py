"""MPNN: DMPNN model (PyG-based). Re-export from 0831-end-code."""
try:
    from models.pyg_models import DMPNNModel, build_pyg_model
except ImportError:
    pass
