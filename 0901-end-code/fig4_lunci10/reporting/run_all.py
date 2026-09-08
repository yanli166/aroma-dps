"""reporting/run_all.py — Fig.4 lunci10 项目的主入口脚本。

用途:
  1. 列出执行计划 (dependency DAG): 每个 phase 的输入 / 输出 / 依赖。
  2. 提供 invoke_phase("phase_14", dry_run=True) 用于本地 dry-run 模拟。
  3. 调用 register_run() 与 final_summary() 在 registry 中记录每个 run。

约定:
  - 不实际执行训练 (所有子模块均为代码占位 / 真实复用)。
  - 凡是子模块 `if __name__ == '__main__'`, 仍可在隔离环境下单独跑。
  - 此处只 *plan* + *registry appending*, 无副作用训练 / checkpoint 写。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

PROJ_ROOT = Path(_PROJ_ROOT)
FIG4_ROOT = PROJ_ROOT / "0901-end-code" / "fig4_lunci10"
REGISTRY_PATH = PROJ_ROOT / "0901-end-code" / "results" / "fig4_lunci10_final" / "run_registry.csv"

REPORTING_DIR = FIG4_ROOT / "reporting"
if str(REPORTING_DIR.parent) not in sys.path:
    sys.path.insert(0, str(FIG4_ROOT))
if str(REPORTING_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTING_DIR))

from run_registry import register_run, final_summary  # type: ignore  # noqa: E402


# --------------------------------------------------------------------------
# Phase registry — order + dependencies
# --------------------------------------------------------------------------
PHASE_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "phase_01_audit",
        "title": "Phase 1 — External audit",
        "script": "audit/audit_repository.py",
        "depends_on": [],
        "produces": ["00_audit/lunci10_external_manifest.csv"],
        "kind": "audit",
    },
    {
        "id": "phase_02_pairs",
        "title": "Phase 2 — Pair construction",
        "script": "data/build_pairs.py",
        "depends_on": ["phase_01_audit"],
        "produces": ["03_pairwise/lunci10_pair_manifest.csv",
                     "03_pairwise/internal_pair_manifest.csv"],
        "kind": "data",
    },
    {
        "id": "phase_03_external_absolute",
        "title": "Phase 3 — External absolute predictions",
        "script": "evaluation/run_external_absolute.py",
        "depends_on": ["phase_02_pairs"],
        "produces": ["01_external_absolute/lunci10_absolute_predictions.csv"],
        "kind": "evaluation",
    },
    {
        "id": "phase_04_external_delta",
        "title": "Phase 4 — External Δ predictions",
        "script": "evaluation/run_external_delta.py",
        "depends_on": ["phase_03_external_absolute"],
        "produces": ["03_pairwise/lunci10_pair_predictions.csv"],
        "kind": "evaluation",
    },
    {
        "id": "phase_05_to_08_bias_novelty_position_hammett",
        "title": "Phase 5–8 — Bias/novelty/position/Hammett",
        "script": "analysis/{scaffold_bias,novelty_analysis,position_analysis,hammett_analysis}.py",
        "depends_on": ["phase_04_external_delta"],
        "produces": ["04_bias/...", "02_novelty/..."],
        "kind": "analysis",
    },
    {
        "id": "phase_09_internal_pair_train",
        "title": "Phase 9 — Internal pair model training",
        "script": "training/train_internal_pair_models.py",
        "depends_on": ["phase_02_pairs"],
        "produces": ["03_pairwise/fig4c_pairwise_summary.csv"],
        "kind": "training",
    },
    {
        "id": "phase_10_to_13_diagnostics",
        "title": "Phase 10–13 — Scaffold/novelty/position/Hammett diagnostics",
        "script": "analysis/{scaffold_bias,novelty_analysis,position_analysis,hammett_analysis}.py",
        "depends_on": ["phase_09_internal_pair_train"],
        "produces": ["04_bias/..."],
        "kind": "analysis",
    },
    {
        "id": "phase_14_pretraining_control",
        "title": "Phase 14 — Pretraining transfer control",
        "script": "training/train_pretraining_control.py",
        "depends_on": ["phase_09_internal_pair_train"],
        "produces": ["si_pretraining/pretraining_transfer.csv"],
        "kind": "training",
    },
    {
        "id": "phase_15_anchor_robustness",
        "title": "Phase 15 — Anchor robustness",
        "script": "analysis/anchor_analysis.py",
        "depends_on": ["phase_14_pretraining_control",
                       "phase_03_external_absolute"],
        "produces": ["si_anchor/anchor_robustness.csv"],
        "kind": "analysis",
    },
    {
        "id": "phase_16_oov_si_methods",
        "title": "Phase 16 — OOV SI methods (M3/M4/M8)",
        "script": "analysis/oov_si_experiments.py",
        "depends_on": ["phase_09_internal_pair_train"],
        "produces": ["si_methods/m3_m4_m8_summary.csv"],
        "kind": "analysis",
    },
    {
        "id": "phase_final_report",
        "title": "Final — Generate Fig.4 report",
        "script": "reporting/generate_fig4_report.py",
        "depends_on": ["phase_15_anchor_robustness", "phase_16_oov_si_methods"],
        "produces": ["fig4_report.md"],
        "kind": "reporting",
    },
]


def render_dag() -> str:
    """输出文字 ASCII 形式的依赖图。"""
    lines = ["# Fig.4 lunci10 — Phase dependency DAG", ""]
    by_id = {p["id"]: p for p in PHASE_REGISTRY}
    for p in PHASE_REGISTRY:
        deps = p["depends_on"]
        if not deps:
            lines.append(f"- {p['id']}  →  ROOT")
        else:
            for d in deps:
                lines.append(f"- {d}  →  {p['id']}")
    lines.append("")
    lines.append("## Linear Topo")
    lines.append("")
    # crude topo sort
    remaining = {p["id"] for p in PHASE_REGISTRY}
    order: List[str] = []
    while remaining:
        progress = False
        for p in PHASE_REGISTRY:
            if p["id"] not in remaining:
                continue
            if all(d not in remaining for d in p["depends_on"]):
                order.append(p["id"])
                remaining.remove(p["id"])
                progress = True
        if not progress:
            lines.append("(cycle detected)")
            break
    for i, pid in enumerate(order, 1):
        lines.append(f"{i:02d}. {pid} — {by_id[pid]['title']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# invoke_phase — dry-run only (本任务不实际跑训练)
# --------------------------------------------------------------------------
def invoke_phase(phase_id: str, dry_run: bool = True,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对一个 phase 调用 *计划* 操作。

    - dry_run=True (default): 仅记录 run 到 registry, status=plan,
      不实际 import 子模块 (避免运行时硬依赖)。
    - 真实执行留给各 phase 子脚本 (每个子脚本都有自己的
      `if __name__ == "__main__"`) 单独运行。
    """
    phase = next((p for p in PHASE_REGISTRY if p["id"] == phase_id), None)
    if phase is None:
        return {"phase_id": phase_id, "status": "unknown_phase"}

    extra = dict(extra or {})
    extra.update({
        "script": phase["script"],
        "produces": ", ".join(phase["produces"]),
        "depends_on": ", ".join(phase["depends_on"]),
    })
    return {
        "phase_id": phase_id,
        "title": phase["title"],
        "script": phase["script"],
        "dry_run": dry_run,
        "registry_recorded": register_run(
            task=str(extra.get("task", "all")),
            model=str(extra.get("model", "(multiple)")),
            dataset=str(extra.get("dataset", "lunci10")),
            protocol=str(extra.get("protocol", phase["kind"])),
            status=str(extra.get("status", "plan" if dry_run else "running")),
            notes=(f"phase={phase_id}; " + json.dumps(extra, ensure_ascii=False)),
            **extra,
        ) if dry_run else None,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-dag", action="store_true",
                   help="print phase dependency DAG and exit")
    p.add_argument("--plan-all", action="store_true",
                   help="plan all phases (dry_run=True) and record into registry")
    p.add_argument("--summary", action="store_true",
                   help="(re)generate run registry summary md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.show_dag:
        print(render_dag())
        return 0
    if args.plan_all:
        results = []
        for phase in PHASE_REGISTRY:
            results.append(invoke_phase(phase["id"], dry_run=True))
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0
    if args.summary:
        info = final_summary()
        print(json.dumps(info, indent=2))
        return 0
    # default: show plan
    print(render_dag())
    print("")
    print(f"registry: {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
