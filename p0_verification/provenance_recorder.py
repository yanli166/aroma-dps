#!/usr/bin/env python3
"""
P0-4: Unified Provenance Recorder

正式新 run 自动保存 git commit、dirty status、dataset hash、split-manifest hash、
split/model seed、config、checkpoint 和软件版本。

设计原则：
- 不修改旧代码，仅提供新模块供正式 run 使用
- 输出为 JSON 文件，放在 result_dir 旁边
- 支持 run_completeness check: expected_runs == completed_runs && missing==0 && failed==0
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# --- Task alias mapping ---
TASK_ALIASES = {
    "MBCO": "MCBO",
    "nMCBO": "MCBO",
    "mcbo": "MCBO",
    "mbco": "MCBO",
}

# Canonical task names
CANONICAL_TASKS = ["HOMA", "NICS_1zz", "MCBO"]


def canonicalize_task_name(name: str) -> str:
    """将旧任务名映射为正式名称。"""
    return TASK_ALIASES.get(name, name)


def task_provenance(name: str) -> Dict[str, str]:
    """返回 task 的 provenance，包含 canonical 和 original_label。

    这样未来可以回答："这个 checkpoint 原始训练文件叫 mbco_value，为什么现在叫 MCBO？"
    """
    canonical = TASK_ALIASES.get(name, name)
    return {
        "canonical": canonical,
        "original_label": name if name != canonical else None,
    }


def get_git_info(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """获取 git 信息。"""
    info = {
        "git_commit": "unknown",
        "git_dirty": None,
        "git_branch": "unknown",
        "git_remote": "unknown",
    }
    try:
        if repo_root is None:
            repo_root = os.getcwd()
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        info["git_dirty"] = bool(dirty)
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True
        ).strip()
        info["git_remote"] = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=repo_root, text=True
        ).strip()
    except Exception as e:
        info["git_error"] = str(e)
    return info


def compute_file_hash(filepath: Union[str, Path], algo: str = "md5") -> str:
    """计算文件 hash。"""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def compute_dataset_hash(dataset_path: Union[str, Path]) -> str:
    """计算数据集文件的 hash。"""
    path = Path(dataset_path)
    if not path.exists():
        return "file_not_found"
    return compute_file_hash(path)


def compute_split_manifest_hash(split_manifest_path: Union[str, Path]) -> str:
    """计算 split manifest 的 hash。"""
    path = Path(split_manifest_path)
    if not path.exists():
        return "file_not_found"
    return compute_file_hash(path)


def compute_checkpoint_hash(checkpoint_path: Union[str, Path]) -> str:
    """计算 checkpoint 文件的 hash。"""
    path = Path(checkpoint_path)
    if not path.exists():
        return "file_not_found"
    return compute_file_hash(path)


def get_package_versions() -> Dict[str, str]:
    """获取关键 Python 包版本。"""
    packages = [
        "torch", "numpy", "pandas", "scikit-learn", "rdkit",
        "scipy", "matplotlib", "tqdm", "yaml",
    ]
    versions = {}
    for pkg in packages:
        try:
            if pkg == "rdkit":
                from rdkit import rdBase
                versions[pkg] = rdBase.rdkitVersion
            elif pkg == "yaml":
                import yaml
                versions[pkg] = yaml.__version__
            elif pkg == "scikit-learn":
                import sklearn
                versions[pkg] = sklearn.__version__
            else:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
        except Exception as e:
            versions[pkg] = f"error: {e}"
    versions["python"] = sys.version
    versions["platform"] = platform.platform()
    return versions


def generate_experiment_id(
    task: str, model: str, split_seed: int, model_seed: int,
    feature_mode: str = "standard", tag: str = ""
) -> str:
    """生成唯一的 experiment_id。"""
    raw = f"{task}_{model}_{split_seed}_{model_seed}_{feature_mode}_{tag}_{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class ProvenanceRecorder:
    """统一 provenance 记录器。"""

    def __init__(
        self,
        result_dir: Union[str, Path],
        repo_root: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = repo_root or os.getcwd()
        self.experiment_id = experiment_id or generate_experiment_id(
            "init", "init", 0, 0
        )
        self._start_time = None
        self._records: List[Dict] = []

    def start_run(
        self,
        task: str,
        model: str,
        split_seed: int,
        model_seed: int,
        config: Dict[str, Any],
        split_manifest_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        feature_mode: str = "standard",
        tag: str = "",
        expected_runs: int = 1,
    ) -> str:
        """开始一次正式 run，返回 experiment_id。"""
        self._start_time = datetime.utcnow()

        # Canonicalize task name with reverse tracking
        task_prov = task_provenance(task)

        # Generate experiment_id
        exp_id = generate_experiment_id(
            task_prov["canonical"], model, split_seed, model_seed, feature_mode, tag
        )
        self.experiment_id = exp_id

        # Collect provenance
        git_info = get_git_info(self.repo_root)
        dataset_hash = compute_dataset_hash(dataset_path) if dataset_path else "N/A"
        split_hash = compute_split_manifest_hash(split_manifest_path) if split_manifest_path else "N/A"
        checkpoint_hash = compute_checkpoint_hash(checkpoint_path) if checkpoint_path else "N/A (not yet saved)"

        record = {
            "experiment_id": exp_id,
            "task": {
                "canonical": task_prov["canonical"],
                "original_label": task_prov["original_label"],
            },
            "model": model,
            "config": config,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "feature_mode": feature_mode,
            "tag": tag,
            "git_commit": git_info["git_commit"],
            "git_dirty": git_info["git_dirty"],
            "git_branch": git_info["git_branch"],
            "git_remote": git_info["git_remote"],
            "dataset_path": str(dataset_path) if dataset_path else "N/A",
            "dataset_hash": dataset_hash,
            "split_manifest_path": str(split_manifest_path) if split_manifest_path else "N/A",
            "split_manifest_hash": split_hash,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "N/A",
            "checkpoint_hash": checkpoint_hash,
            "package_versions": get_package_versions(),
            "start_time": self._start_time.isoformat(),
            "end_time": None,
            "run_status": "running",
            "error_message": "",
            "expected_runs": expected_runs,
        }

        self._records.append(record)
        self._save()
        return exp_id

    def complete_run(
        self,
        experiment_id: str,
        run_status: str = "ok",
        error_message: str = "",
        checkpoint_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """完成一次 run。"""
        for rec in self._records:
            if rec["experiment_id"] == experiment_id:
                rec["end_time"] = datetime.utcnow().isoformat()
                rec["run_status"] = run_status
                rec["error_message"] = error_message
                if checkpoint_path:
                    rec["checkpoint_path"] = str(checkpoint_path)
                    rec["checkpoint_hash"] = compute_checkpoint_hash(checkpoint_path)
                if metrics:
                    rec["metrics"] = metrics
                break
        self._save()

    def _save(self):
        """保存 provenance 到 JSON。"""
        provenance_path = self.result_dir / "provenance.json"
        with open(provenance_path, "w") as f:
            json.dump({
                "records": self._records,
                "generated_at": datetime.utcnow().isoformat(),
            }, f, indent=2, default=str)

    def check_completeness(
        self,
        expected_runs: int,
        task: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """检查 run 完整性。

        正式 summary 只有在 expected_runs == completed_runs
        && missing_runs == 0 && failed_runs == 0 时才允许生成。
        """
        records = self._records
        if task:
            # task is now a dict {canonical, original_label}; match on canonical
            records = [
                r for r in records
                if (r["task"]["canonical"] if isinstance(r.get("task"), dict) else r.get("task")) == task
            ]
        if model:
            records = [r for r in records if r["model"] == model]

        completed = [r for r in records if r["run_status"] in ("ok", "fail")]
        ok_runs = [r for r in completed if r["run_status"] == "ok"]
        failed_runs = [r for r in completed if r["run_status"] == "fail"]
        running = [r for r in records if r["run_status"] == "running"]

        result = {
            "expected_runs": expected_runs,
            "completed_runs": len(completed),
            "ok_runs": len(ok_runs),
            "failed_runs": len(failed_runs),
            "missing_runs": expected_runs - len(completed),
            "running_runs": len(running),
            "all_complete": len(completed) == expected_runs and len(running) == 0,
            "can_generate_summary": (
                len(completed) == expected_runs
                and len(failed_runs) == 0
                and len(running) == 0
                and len(records) == expected_runs
            ),
        }
        return result

    def save_summary(
        self,
        summary_data: Dict[str, Any],
        filename: str = "summary_with_provenance.json",
    ):
        """保存带 provenance 的 summary。

        只有在 can_generate_summary == True 时才允许调用。
        """
        completeness = self.check_completeness(
            self._records[0].get("expected_runs", 1) if self._records else 1
        )
        if not completeness["can_generate_summary"]:
            raise RuntimeError(
                f"Cannot generate summary: {completeness}"
            )

        git_info = get_git_info(self.repo_root)
        summary = {
            "summary_data": summary_data,
            "provenance": {
                "git_commit": git_info["git_commit"],
                "git_dirty": git_info["git_dirty"],
                "generated_at": datetime.utcnow().isoformat(),
                "n_runs": len(self._records),
                "experiment_ids": [r["experiment_id"] for r in self._records],
            },
            "completeness_check": completeness,
        }
        summary_path = self.result_dir / filename
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary_path


def load_provenance(result_dir: Union[str, Path]) -> Optional[Dict]:
    """加载已有的 provenance 文件。"""
    path = Path(result_dir) / "provenance.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# --- CLI for testing ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Provenance Recorder Test")
    parser.add_argument("--result-dir", default="/tmp/provenance_test", help="Output dir")
    parser.add_argument("--task", default="MBCO", help="Task name (test alias mapping)")
    parser.add_argument("--model", default="RC-GNN", help="Model name")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=42)
    args = parser.parse_args()

    recorder = ProvenanceRecorder(
        result_dir=args.result_dir,
        repo_root="/home/ubuntu/aroma-dps",
    )

    # Simulate a run
    exp_id = recorder.start_run(
        task=args.task,
        model=args.model,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        config={"hidden_dim": 128, "n_layers": 3, "lr": 0.001},
        dataset_path="/home/ubuntu/aroma-dps-code/code_end/data1_end/collet_homa_0716.csv",
        feature_mode="standard",
        tag="test_run",
        expected_runs=1,
    )
    print(f"Started run: {exp_id}")

    # Simulate completion
    time.sleep(0.1)
    recorder.complete_run(
        experiment_id=exp_id,
        run_status="ok",
        metrics={"test_mae": 0.109, "test_r2": 0.991, "test_rmse": 0.215},
    )
    print(f"Completed run: {exp_id}")

    # Check completeness
    completeness = recorder.check_completeness(expected_runs=1)
    print(f"\nCompleteness: {json.dumps(completeness, indent=2)}")

    if completeness["can_generate_summary"]:
        summary_path = recorder.save_summary(
            {"best_model": "RC-GNN", "best_mae": 0.109}
        )
        print(f"\nSummary saved to: {summary_path}")

    # Show provenance
    prov = load_provenance(args.result_dir)
    if prov:
        print(f"\nProvenance records:")
        for rec in prov["records"]:
            print(f"  exp_id={rec['experiment_id']}")
            # task is now a dict: {canonical: "MCBO", original_label: "MBCO"}
            task_info = rec["task"]
            if isinstance(task_info, dict):
                print(f"  task: canonical={task_info['canonical']} original_label={task_info['original_label']}")
            else:
                print(f"  task={task_info}")
            print(f"  model={rec['model']}")
            print(f"  split_seed={rec['split_seed']} model_seed={rec['model_seed']}")
            print(f"  git_commit={rec['git_commit'][:12]} dirty={rec['git_dirty']}")
            print(f"  dataset_hash={rec['dataset_hash']}")
            print(f"  run_status={rec['run_status']}")
            print(f"  start={rec['start_time']} end={rec['end_time']}")
