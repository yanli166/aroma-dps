"""Phase 4 final consolidation: rerun all 3 tasks on 1 GPU, then merge predictions."""
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'code_end'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'unified_models'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, '0901-end-code', 'fig4_lunci10'))

import evaluation.run_external_absolute_v2 as v2

v2.TASKS = ["HOMA", "NICS_1zz", "MBCO"]
v2.MODELS = v2.MODELS  # all 3

if __name__ == "__main__":
    v2.main()