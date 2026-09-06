"""GAT backbone. Re-export from unified_models."""
try:
    from unified_models.gat.model import MultiHeadGAT, GATModel
except ImportError:
    pass
