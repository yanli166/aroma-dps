"""补训 layer3_ring_fixed 中受 atom_on_ring -1 偏移 bug 影响的 frozen checkpoint.

背景: graphs.py 修复 (2026-09-02) 后, 仅 seed_42 的 checkpoint 重训过,
其余 seeds 仍是 7 月旧 checkpoint (环标记偏移). 本脚本用与 seed_42 相同的
协议 (run_experiment, DEFAULT_PARAMS, n_epochs=200, patience=30) 补训.

支持任意 backbone (MPNN / GAT / GIN / GNN), encoding=label.

用法: python retrain_frozen_mpnn_label.py --seeds 123,456 --model GAT --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/training/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
CODE_END = os.path.join(PROJ_ROOT, "code_end")
OUT = os.path.join(CODE_END, "results", "layer3_ring_fixed")

for p in (PROJ_ROOT, CODE_END):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.tasks import TASKS  # noqa: E402
from ring_encoding_ablation.code.ring_train_eval import (  # noqa: E402
    run_experiment, DEFAULT_PARAMS,
)

TASK_NAMES = ["HOMA", "NICS_1zz", "MBCO"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True, help="comma list, e.g. 123,456")
    parser.add_argument("--model", type=str, default="MPNN", help="backbone name (MPNN / GAT / GIN / GNN)")
    parser.add_argument("--encoding", type=str, default="label", help="encoding (label / mask / pool / none / combined)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    task_list = [t for t in TASKS if t["name"] in TASK_NAMES]
    assert len(task_list) == 3, f"TASKS 中缺少任务: {[t['name'] for t in TASKS]}"

    for seed in seeds:
        seed_dir = os.path.join(OUT, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        for task in task_list:
            print(f"\n{'#'*60}\n# {args.model}_{args.encoding} | seed={seed} | task={task['name']}\n{'#'*60}", flush=True)
            run_experiment(args.model, args.encoding, task, dict(DEFAULT_PARAMS), device,
                           seed_dir, args.n_epochs, args.patience, seed)
    print(f"\n[done] retrain {args.model}_{args.encoding} seeds: {seeds}", flush=True)


if __name__ == "__main__":
    main()
