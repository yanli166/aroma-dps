#!/usr/bin/env python3
"""
Fig.3 launcher: delegates to existing stage scripts.

Usage:
    python experiments/fig3/launcher.py --stage 1 --seed 42 --feature-mode standard
    python experiments/fig3/launcher.py --stage 5 --seed 42 --feature-mode explicit_aromaticity_ablated

This wrapper does NOT duplicate model/graph/split/scaler/metric implementations.
It only configures and launches the existing scripts.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

STAGE_SCRIPTS = {
    1: "0831-end-code/stage1_representation_comparison/code/run_stage1_v2.py",
    2: "0831-end-code/stage2_ring_conditioning/code/ring_conditioning_ablation.py",
    3: "0831-end-code/stage3_mask_pretraining/code/run_mask_pretrain_v2.py",
    4: "0831-end-code/stage4_cross_architecture/code/run_stage4_v2.py",
    5: "0831-end-code/stage5_ring_flag_sensitivity/code/run_ring_flag_sensitivity.py",
    6: "0831-end-code/stage6_final_membership/code/run_final_membership.py",
}

# Seed mapping: display seed -> internal seed
SEED_MAP = {42: 11, 123: 22, 456: 33, 789: 44, 2024: 55}


def main():
    parser = argparse.ArgumentParser(description="Fig.3 experiment launcher")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 123, 456, 789, 2024])
    parser.add_argument("--feature-mode", default="standard",
                        choices=["standard", "explicit_aromaticity_ablated"])
    args = parser.parse_args()

    script = REPO_ROOT / STAGE_SCRIPTS[args.stage]
    internal_seed = SEED_MAP[args.seed]

    cmd = [
        sys.executable, str(script),
        "--seed", str(internal_seed),
        "--feature_mode", args.feature_mode,
    ]

    print(f"Launching Fig.3 Stage {args.stage}: {script.name}")
    print(f"  model_seed={args.seed} (internal={internal_seed})")
    print(f"  feature_mode={args.feature_mode}")
    print(f"  split_seed=2026 (fixed by protocol)")
    print(f"  Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
