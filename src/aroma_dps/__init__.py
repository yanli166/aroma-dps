"""
aroma_dps: Target-ring-conditioned aromaticity prediction.

Single source of truth for the publication pipeline.
Historical code in 0831-end-code/, 0901-end-code/, code_end/, unified_models/
remains as-is (archive). New formal runs should import from this package.
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
