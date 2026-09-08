
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
"""
GPU 2 测试: 重复 anchor_step8_pretraining 实验
输出到独立目录 gpu2_test/ 避免覆盖原结果
"""
import os
import sys

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'

# 在导入 anchor_step8 模块前, monkey-patch OUTPUT_DIR
import generalization_test.code.anchor_step8_pretraining as step8
step8.OUTPUT_DIR = os.path.join(PROJ_ROOT, 'results/lunci10_anchor_delta_final/gpu2_test')
os.makedirs(step8.OUTPUT_DIR, exist_ok=True)

# 复用原脚本的所有逻辑, 只是改了 OUTPUT_DIR 和默认 GPU
step8.PAIR_FILE = os.path.join(PROJ_ROOT, 'results/lunci10_anchor_delta_final/anchor_pair_dataset.csv')

if __name__ == '__main__':
    # 强制使用 GPU 2
    sys.argv = [sys.argv[0], '--gpu', '2']
    step8.main()
