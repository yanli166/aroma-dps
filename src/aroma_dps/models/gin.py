"""GIN backbone. Re-export from unified_models."""
try:
    from unified_models.gin.model import GINLayer, GINModel
except ImportError:
    pass
