"""Phase 0: 自动发现并审计关键资源 (fig4_v2 Audit Phase).

检查数据/模型/协议。输出:
    - fig4_v2_audit_report.md
    - fig4_v2_audit_report.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# Make project root importable for shared utilities if needed.
PROJ_ROOT = "_PROJ_ROOT"
CODE_END_ROOT = "_PROJ_ROOT + "/code_end""
FIG4_ROOT = Path("_PROJ_ROOT/0901-end-code/fig4_lunci10")
AUDIT_OUT = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/00_audit")

for p in (PROJ_ROOT, CODE_END_ROOT):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


def _load_config(config_path: Path) -> Dict[str, Any]:
    import yaml  # local import to keep module import-light

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------- individual checks ---------------------------

def check_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Data-side checks: Stage-I training data, lunci10, scaffold-OOD lookups."""
    paths = cfg["paths"]
    items: List[Dict[str, Any]] = []

    required_files = {
        "lunci10_test_corrected": paths["lunci10_test"],
        "lunci10_unified": paths["lunci10_unified"],
        "lunci10_begin": paths["lunci10_begin"],
        "lunci10_summary": paths["lunci10_summary"],
        "internal_train_homa": paths["internal_train_homa"],
        "internal_train_mbco": paths["internal_train_mbco"],
        "internal_train_nics": paths["internal_train_nics"],
    }
    for name, p in required_files.items():
        path = Path(p)
        ok = path.is_file()
        items.append({
            "name": name,
            "path": str(path),
            "exists": ok,
            "size_bytes": path.stat().st_size if ok else 0,
        })

    # Lookups
    hammett_path = Path(paths["hammett_module"])
    items.append({
        "name": "hammett_lookup",
        "path": str(hammett_path),
        "exists": hammett_path.is_file(),
        "size_bytes": hammett_path.stat().st_size if hammett_path.is_file() else 0,
    })

    # Anchor/scaffold lookup tables: derived from layer2_gnn if present
    layer2_dir = Path(paths["layer2_gnn_dir"])
    items.append({
        "name": "anchor_lookup_dir",
        "path": str(layer2_dir),
        "exists": layer2_dir.is_dir(),
    })

    all_ok = all(it["exists"] for it in items)
    return {"all_ok": all_ok, "items": items}


def check_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Model-side checks: final RC-GNN, Base GNN, per-task ckpts, Stage-I encoder."""
    paths = cfg["paths"]
    items: List[Dict[str, Any]] = []

    candidate_files = {
        "final_rc_gnn": [
            # try common filenames (.pt and .pth)
            Path(paths["aromatic_split_dir"]) / "rc_gnn_final.pt",
            Path(paths["aromatic_split_dir"]) / "final.pt",
            Path(paths["aromatic_split_dir"]) / "model_best.pt",
            Path(paths["aromatic_split_dir"]) / "rc_gnn_final.pth",
            Path(paths["aromatic_split_dir"]) / "final.pth",
        ],
        "base_gnn_layer2": [
            Path(paths["layer2_gnn_dir"]) / "model.pt",
            Path(paths["layer2_gnn_dir"]) / "best.pt",
            Path(paths["layer2_gnn_dir"]) / "final.pt",
            Path(paths["layer2_gnn_dir"]) / "model.pth",
            Path(paths["layer2_gnn_dir"]) / "best.pth",
        ],
        "ring_layer3": [
            Path(paths["layer3_ring_dir"]) / "model.pt",
            Path(paths["layer3_ring_dir"]) / "best.pt",
            Path(paths["layer3_ring_dir"]) / "final.pt",
            Path(paths["layer3_ring_dir"]) / "model.pth",
            Path(paths["layer3_ring_dir"]) / "best.pth",
        ],
    }

    for name, cands in candidate_files.items():
        existing = [c for c in cands if c.is_file()]
        # additionally, recursively look for any .pt/.pth under the relevant parent dir
        if name == "final_rc_gnn":
            base = Path(paths["aromatic_split_dir"])
        elif name == "base_gnn_layer2":
            base = Path(paths["layer2_gnn_dir"])
        elif name == "ring_layer3":
            base = Path(paths["layer3_ring_dir"])
        else:
            base = None
        if base is not None and base.is_dir():
            for p in base.rglob("*.pth"):
                if str(p) not in existing:
                    existing.append(p)
        items.append({
            "name": name,
            "candidates": [str(c) for c in cands],
            "found": [str(c) for c in existing],
            "exists": len(existing) > 0,
        })

    # Per-task checkpoints — search the split results subfolder
    task_dirs = {
        "HOMA_ckpt": Path(paths["aromatic_split_dir"]) / "HOMA",
        "NICS_1zz_ckpt": Path(paths["aromatic_split_dir"]) / "NICS_1zz",
        "MBCO_ckpt": Path(paths["aromatic_split_dir"]) / "MBCO",
    }
    for name, d in task_dirs.items():
        found = []
        if d.is_dir():
            found = [str(p) for p in d.rglob("*.pt")] + [str(p) for p in d.rglob("*.pth")]
        items.append({
            "name": name,
            "dir": str(d),
            "exists": d.is_dir(),
            "found": found,
        })

    all_ok = all(it["exists"] for it in items)
    return {"all_ok": all_ok, "items": items}


def check_protocol(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Protocol-side checks: seeds, E* protocol, feature_mode, ring_flag_value."""
    seeds = cfg.get("seeds", {})
    model = cfg.get("model", {})
    zero = cfg.get("zero_shot", {})

    split_seed = seeds.get("split_seed")
    model_seeds = seeds.get("model_seeds", [])
    ring_flag_value = model.get("ring_flag_value")
    feature_mode = model.get("feature_mode")
    n_folds = model.get("n_folds")
    tasks = model.get("tasks", [])
    forbidden = zero.get("forbidden_actions", [])

    details = {
        "split_seed": split_seed,
        "model_seeds": model_seeds,
        "n_model_seeds": len(model_seeds) if isinstance(model_seeds, list) else 0,
        "ring_flag_value": ring_flag_value,
        "feature_mode": feature_mode,
        "n_folds": n_folds,
        "tasks": tasks,
        "forbidden_actions": forbidden,
    }

    issues: List[str] = []
    if split_seed != 42:
        issues.append("split_seed != 42 (E* protocol violation)")
    if not isinstance(model_seeds, list) or len(model_seeds) < 3:
        issues.append("model_seeds should be a list with >=3 entries")
    if ring_flag_value != 10:
        issues.append("ring_flag_value != 10 (frozen)")
    if feature_mode not in ("atom_bond", "atom", "bond"):
        issues.append(f"unknown feature_mode: {feature_mode}")

    return {
        "all_ok": len(issues) == 0,
        "details": details,
        "issues": issues,
    }


# --------------------------- report rendering ---------------------------

def render_markdown(audit: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Fig.4 lunci10 — Audit Report (Phase 0)")
    lines.append("")
    lines.append(f"- generated_at: {audit['generated_at']}")
    lines.append(f"- config: `{audit['config_path']}`")
    lines.append("")

    def section(title: str, key: str) -> None:
        block = audit[key]
        ok = block.get("all_ok")
        lines.append(f"## {title} — {'PASS' if ok else 'FAIL'}")
        lines.append("")
        if key == "protocol":
            for k, v in block.get("details", {}).items():
                lines.append(f"- **{k}**: `{v}`")
            if block.get("issues"):
                lines.append("")
                lines.append("**Issues:**")
                for issue in block["issues"]:
                    lines.append(f"- {issue}")
            lines.append("")
            return
        for it in block.get("items", []):
            exists = it.get("exists")
            name = it.get("name", "?")
            if "path" in it:
                lines.append(f"- {'OK' if exists else 'MISSING'} `{name}` -> `{it['path']}`")
            elif "candidates" in it:
                lines.append(
                    f"- {'OK' if exists else 'MISSING'} `{name}` -> found {len(it.get('found', []))} of {len(it.get('candidates', []))}"
                )
            elif "dir" in it:
                lines.append(
                    f"- {'OK' if exists else 'MISSING'} `{name}` -> `{it['dir']}` ({len(it.get('found', []))} ckpts)"
                )
            else:
                lines.append(f"- {'OK' if exists else 'MISSING'} `{name}`")
        lines.append("")

    section("Data & Lookups", "data")
    section("Models & Checkpoints", "models")
    section("Protocol", "protocol")

    overall = (
        audit["data"]["all_ok"]
        and audit["models"]["all_ok"]
        and audit["protocol"]["all_ok"]
    )
    lines.insert(3, f"- overall_status: **{'PASS' if overall else 'FAIL'}**")
    lines.append("---")
    lines.append("")
    lines.append(
        "Note: This audit does NOT touch lunci10 labels for training. "
        "It only verifies that all frozen artifacts exist and the frozen protocol is respected."
    )
    return "\n".join(lines)


def run(config_path: Path | None = None) -> Dict[str, Any]:
    config_path = config_path or (FIG4_ROOT / "configs" / "fig4_lunci10.yaml")
    cfg = _load_config(config_path)

    audit = OrderedDict()
    audit["config_path"] = str(config_path)
    audit["generated_at"] = datetime.utcnow().isoformat() + "Z"
    audit["data"] = check_data(cfg)
    audit["models"] = check_models(cfg)
    audit["protocol"] = check_protocol(cfg)
    audit["overall_ok"] = (
        audit["data"]["all_ok"]
        and audit["models"]["all_ok"]
        and audit["protocol"]["all_ok"]
    )

    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    md_path = AUDIT_OUT / "fig4_v2_audit_report.md"
    json_path = AUDIT_OUT / "fig4_v2_audit_report.json"
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "md": str(md_path),
        "json": str(json_path),
        "overall_ok": audit["overall_ok"],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
