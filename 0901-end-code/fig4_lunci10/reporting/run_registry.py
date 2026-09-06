"""Run registry for Fig.4 lunci10 external generalization project.

设计目标:
  - 统一记录所有训练与测试运行, 不论成功失败, 失败 runs 保留 status=fail。
  - 提供 register_run() 接口 (子模块直接调用)。
  - 提供 final_summary() 接口 (生成 markdown 汇总)。
  - CSV 字段:
        run_id, timestamp, git_commit, task, model, checkpoint,
        dataset, protocol, split_seed, model_seed, pretrained,
        feature_mode, train_N, test_N, MAE, RMSE, R2,
        status, notes

约束:
  - 失败 runs 不许删除 / 覆盖, status 字段诚实记录 'fail'。
  - 文件 atomic: 每次写入用 tmp + replace, 防止并发损坏。
  - git commit 自动探测, 失败则空字符串 + 备注。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/reporting/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
REGISTRY_PATH = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/run_registry.csv"
SUMMARY_PATH = Path(_PROJ_ROOT) / "0901-end-code/results/fig4_lunci10_final/run_registry_summary.md"

REGISTRY_FIELDS: List[str] = [
    "run_id", "timestamp", "git_commit", "task", "model",
    "checkpoint", "dataset", "protocol", "split_seed", "model_seed",
    "pretrained", "feature_mode", "train_N", "test_N",
    "MAE", "RMSE", "R2", "status", "notes",
]


# --------------------------------------------------------------------------
# Git commit 自动探测
# --------------------------------------------------------------------------
def _probe_git_commit(repo_root: Optional[str] = None) -> str:
    repo_root = repo_root or _PROJ_ROOT
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_run_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# Core: register_run() — append-once, atomic
# --------------------------------------------------------------------------
def register_run(
    task: str,
    model: str,
    dataset: str,
    protocol: str,
    *,
    checkpoint: str = "",
    split_seed: int = 42,
    model_seed: int = 42,
    pretrained: bool = False,
    feature_mode: str = "atom_bond",
    train_N: int = 0,
    test_N: int = 0,
    MAE: float = float("nan"),
    RMSE: float = float("nan"),
    R2: float = float("nan"),
    status: str = "ok",
    notes: str = "",
    repo_root: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Append a new run row to run_registry.csv. Returns the run_id."""

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_id = _gen_run_id()
    row: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _utc_now_iso(),
        "git_commit": _probe_git_commit(repo_root=repo_root),
        "task": str(task),
        "model": str(model),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "protocol": str(protocol),
        "split_seed": int(split_seed),
        "model_seed": int(model_seed),
        "pretrained": bool(pretrained),
        "feature_mode": str(feature_mode),
        "train_N": int(train_N),
        "test_N": int(test_N),
        "MAE": float(MAE) if MAE == MAE else "",  # NaN → empty string
        "RMSE": float(RMSE) if RMSE == RMSE else "",
        "R2": float(R2) if R2 == R2 else "",
        "status": status,
        "notes": notes,
    }
    if extra:
        for k, v in extra.items():
            if k in REGISTRY_FIELDS:
                row[k] = v
            # unknown keys ignored to keep schema stable

    # Read existing (if any) so we can append while preserving schema.
    is_new_file = not REGISTRY_PATH.exists()
    if is_new_file:
        # write header + first row atomically
        _write_rows_atomic([row], header=REGISTRY_FIELDS, path=REGISTRY_PATH)
    else:
        # Append (atomic rename)
        with REGISTRY_PATH.open("r", newline="", encoding="utf-8") as f:
            existing_reader = csv.DictReader(f)
            existing_fieldnames = existing_reader.fieldnames or REGISTRY_FIELDS
            # ensure schema compatible: append any missing fields
            fields = list(existing_fieldnames)
            for fld in REGISTRY_FIELDS:
                if fld not in fields:
                    fields.append(fld)
            rows = list(existing_reader)
        rows.append(row)
        _write_rows_atomic(rows, header=fields, path=REGISTRY_PATH)
    return run_id


def _write_rows_atomic(rows: List[Dict[str, Any]],
                       header: List[str],
                       path: Path) -> None:
    """Atomic CSV write: tmpfile + os.replace."""
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in header})
        os.replace(tmp_path_str, path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except Exception:
            pass
        raise


# --------------------------------------------------------------------------
# Final summary — markdown
# --------------------------------------------------------------------------
def final_summary(registry_path: Path = REGISTRY_PATH,
                  summary_path: Path = SUMMARY_PATH) -> Dict[str, Any]:
    if not registry_path.exists():
        return {"exists": False, "n_runs": 0}
    df_src = _read_csv_safe(registry_path)
    if not len(df_src):
        return {"exists": True, "n_runs": 0}

    df_src = df_src.copy()

    n_total = int(len(df_src))
    n_ok = int((df_src["status"] == "ok").sum())
    n_fail = int((df_src["status"].isin(["fail", "stub", "missing_checkpoint",
                                          "missing_layer4_module",
                                          "needs_more_implementation"])).sum())

    lines: List[str] = []
    lines.append("# Fig.4 lunci10 — Run Registry Summary")
    lines.append("")
    lines.append(f"- Total runs: {n_total}")
    lines.append(f"- ok: {n_ok}")
    lines.append(f"- failed/non-ok: {n_fail}")
    lines.append("")

    # status breakdown
    if "status" in df_src.columns:
        lines.append("## Status breakdown")
        lines.append("")
        lines.append(df_src["status"].value_counts().to_frame("n_runs").to_markdown())
        lines.append("")

    # best per (task, model, protocol)
    if {"MAE", "task", "model", "protocol"}.issubset(df_src.columns):
        ok_df = df_src[df_src["status"] == "ok"].copy()
        if len(ok_df) and "MAE" in ok_df.columns:
            try:
                ok_df["MAE_num"] = pd.to_numeric(ok_df["MAE"], errors="coerce")
                best_idx = ok_df.groupby(["task", "protocol"])["MAE_num"].idxmin()
                best = ok_df.loc[best_idx].reset_index(drop=True)
                lines.append("## Best per (task, protocol)")
                lines.append("")
                keep = ["task", "protocol", "model", "MAE", "RMSE", "R2",
                        "pretrained", "checkpoint"]
                keep = [k for k in keep if k in best.columns]
                lines.append(best[keep].to_markdown(index=False))
                lines.append("")
            except Exception:
                pass

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {"exists": True, "n_runs": n_total, "n_ok": n_ok, "n_fail": n_fail}


def _read_csv_safe(path: Path):
    """Lazy pandas import (avoids hard dep at module-import time)."""
    import pandas as pd  # noqa: WPS433  (intentional runtime import)
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# CLI: print path / final_summary / example
# --------------------------------------------------------------------------
def _example_register() -> str:
    return register_run(
        task="HOMA", model="RC_MPNN", dataset="lunci10",
        protocol="external_absolute",
        checkpoint="rc_mpnn_homa_best.pt",
        split_seed=42, model_seed=42, pretrained=False,
        feature_mode="atom_bond",
        train_N=0, test_N=10,
        MAE=0.07, RMSE=0.10, R2=0.85,
        status="ok", notes="example bootstrap run",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", action="store_true",
                   help="(re)generate the summary markdown")
    p.add_argument("--example", action="store_true",
                   help="append an example row (for testing only)")
    p.add_argument("--path", action="store_true",
                   help="print the registry CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.path:
        print(str(REGISTRY_PATH))
    if args.example:
        rid = _example_register()
        print(f"registered example run: {rid}")
    if args.summary:
        info = final_summary()
        print(json.dumps(info, indent=2))
        print(f"summary md: {SUMMARY_PATH}")
    if not (args.path or args.example or args.summary):
        print(f"registry: {REGISTRY_PATH}")
        info = final_summary()
        print(json.dumps(info, indent=2))
