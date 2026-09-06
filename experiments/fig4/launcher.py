#!/usr/bin/env python3
"""
Fig.4 launcher: delegates to existing Fig.4 scripts.

Usage:
    python experiments/fig4/launcher.py --experiment scaffold_ood --task HOMA
    python experiments/fig4/launcher.py --experiment external_absolute --task HOMA
    python experiments/fig4/launcher.py --experiment exposure_curve --task HOMA
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FIG4_ROOT = "0901-end-code/fig4_lunci10"

SCRIPTS = {
    "scaffold_ood": f"{FIG4_ROOT}/training/train_scaffold_ood.py",
    "exposure_curve": f"{FIG4_ROOT}/training/train_exposure_curve.py",
    "ring_family_ood": f"{FIG4_ROOT}/training/train_l10_exposure.py",
    "external_absolute": f"{FIG4_ROOT}/evaluation/run_external_absolute_v2.py",
    "external_delta": f"{FIG4_ROOT}/evaluation/run_external_delta.py",
    "hammett": f"{FIG4_ROOT}/analysis/hammett_analysis.py",
    "novelty": f"{FIG4_ROOT}/analysis/novelty_analysis.py",
}

TASKS = ["HOMA", "NICS_1zz", "MCBO"]


def main():
    parser = argparse.ArgumentParser(description="Fig.4 experiment launcher")
    parser.add_argument("--experiment", required=True, choices=list(SCRIPTS.keys()))
    parser.add_argument("--task", default="HOMA", choices=TASKS)
    args = parser.parse_args()

    script = REPO_ROOT / SCRIPTS[args.experiment]

    print(f"Launching Fig.4 {args.experiment}: {script.name}")
    print(f"  task={args.task}")
    print(f"  Protocol: see experiments/fig4/config.yaml")
    print()

    # Pass task as environment variable; scripts read it differently
    env = {**__import__("os").environ, "TASK": args.task}
    subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
