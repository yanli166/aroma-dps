"""Phase 4 quick smoke test: only RC_MPNN seed=42 HOMA task."""
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'code_end'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'unified_models'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, '0901-end-code', 'fig4_lunci10'))

# Monkey-patch MODELS to only one config
import evaluation.run_external_absolute_v2 as v2
v2.MODELS = {
    "RC_MPNN": {
        "layer": "layer3_ring_fixed",
        "cls": None,
        "available_seeds": [42],
    }
}
v2.TASKS = ["HOMA"]

if __name__ == "__main__":
    v2.main()