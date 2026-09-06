"""Task 10 (按协议修正): ring_flag=10 / frozen architecture provenance.

要求:
  - 不得根据 lunci10 performance 决定是否改成 learnable projection
  - 从 Fig.3 已冻结的历史结果/正式 checkpoint 确认最终 architecture
  - 如果 Fig.3 final model 在 external evaluation 之前已正式冻结为 ring_flag=10, 则 Fig.4 继续使用 ring_flag=10
  - 如果此前已正式冻结为 binary membership + learnable projection, 则使用 projection
  - 禁止在 lunci10 上比较两者后重新选择 final model
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

PROJ_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_END = PROJ_ROOT / "code_end"
FIG4_ROOT = PROJ_ROOT / "0901-end-code/fig4_lunci10"
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"

OUT_JSON = AUDIT_OUT / "frozen_architecture_provenance.json"

# Files that document Fig.3 final architecture decision
ARCH_DECISION_FILES = [
    CODE_END / "FINAL_REPORT.md",
    CODE_END / "results" / "final_report.md",
    CODE_END / "aromatic_split" / "code" / "run_aromatic_split.py",
    CODE_END / "ring_encoding_ablation" / "code" / "combined_ragcn.py",
    CODE_END / "ring_encoding_ablation" / "code" / "ring_train_eval.py",
    CODE_END / "layer4_substituent" / "code" / "m6_position_encoding.py",
]


def find_ring_flag_evidence() -> List[Dict[str, Any]]:
    """Grep for ring_flag / learnable_projection evidence in Fig.3 decision files."""
    evidence: List[Dict[str, Any]] = []
    patterns = [
        re.compile(r"ring_flag\s*=\s*(\d+|\w+)"),
        re.compile(r"learnable_projection\s*[:=]\s*(\w+)"),
        re.compile(r"RING_FLAG_VALUE\s*=\s*(\d+|\w+)"),
        re.compile(r"binary[_ ]membership"),
        re.compile(r"learnable[_ ]projection"),
        re.compile(r"ring[_ ]encoding[_ ]ablation"),
    ]
    for p in ARCH_DECISION_FILES:
        if not p.is_file():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        hits = []
        for pat in patterns:
            for m in pat.finditer(txt):
                # extract a 1-line window
                line_start = txt.rfind("\n", 0, m.start()) + 1
                line_end = txt.find("\n", m.end())
                snippet = txt[line_start:line_end if line_end > 0 else m.end() + 100].strip()
                hits.append({"pattern": pat.pattern, "match": m.group(0), "context": snippet[:200]})
        if hits:
            evidence.append({
                "file": str(p.relative_to(PROJ_ROOT)),
                "n_hits": len(hits),
                "hits": hits[:6],
            })
    return evidence


def find_l10_provenance() -> List[Dict[str, Any]]:
    """Find any code that used lunci10 for ring_flag/projection selection (RED FLAG)."""
    red_flags = []
    suspect_patterns = [
        re.compile(r"lunci10.*ring_flag", re.IGNORECASE),
        re.compile(r"lunci10.*learnable_projection", re.IGNORECASE),
        re.compile(r"ring_flag.*lunci10", re.IGNORECASE),
        re.compile(r"projection.*lunci10", re.IGNORECASE),
    ]
    code_root = PROJ_ROOT / "code_end"
    if not code_root.is_dir():
        return red_flags
    for f in code_root.rglob("*.py"):
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        for pat in suspect_patterns:
            for m in pat.finditer(txt):
                line_start = txt.rfind("\n", 0, m.start()) + 1
                line_end = txt.find("\n", m.end())
                snippet = txt[line_start:line_end if line_end > 0 else m.end() + 100].strip()
                red_flags.append({
                    "file": str(f.relative_to(PROJ_ROOT)),
                    "pattern": pat.pattern,
                    "context": snippet[:200],
                })
    return red_flags


def main() -> None:
    print("[frozen provenance] searching Fig.3 decision files for ring_flag / learnable_projection...")
    evidence = find_ring_flag_evidence()
    print(f"  found {len(evidence)} files with hits")

    print("[RED FLAG check] searching for any code that tuned ring_flag using lunci10...")
    rf = find_l10_provenance()
    print(f"  found {len(rf)} suspect matches")

    # Decisional rule (per protocol):
    # If frozen_decision files mention ring_flag=10 as the frozen value AND there is
    # NO evidence of post-lunci10 tuning of ring_flag, use ring_flag=10.
    has_ring_flag_10 = any(
        "10" in h["match"] for ev in evidence for h in ev["hits"] if "ring_flag" in h["pattern"]
    )

    decision = "use_ring_flag_10" if has_ring_flag_10 and not rf else "investigate_before_decision"

    out = {
        "protocol": "frozen_architecture_provenance_check (no lunci10-based re-selection)",
        "evidence": evidence,
        "lunci10_red_flags": rf,
        "decision": decision,
        "rationale": (
            "ring_flag=10 retained per Fig.3 freeze if found in the decision files "
            "and no post-lunci10 tuning evidence is present; otherwise flagged for "
            "manual review"
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_JSON}")
    print()
    print(f"DECISION: {decision}")
    print(f"  has_ring_flag_10={has_ring_flag_10}, n_red_flags={len(rf)}")


if __name__ == "__main__":
    main()