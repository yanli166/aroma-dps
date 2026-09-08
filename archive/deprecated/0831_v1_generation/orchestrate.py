#!/usr/bin/env python
"""
4-GPU 并行全流程编排脚本 (从零启动, 修复版)

GPU 分配:
  Phase 1: Stage 1ml (CPU) + Stage 1gnn (GPU 0,1,2 按模型) + Stage 4 (GPU 3)
  Phase 2: Stage 2 (GPU 0,1,2 按任务) | Stage 4 继续 (GPU 3)
  Phase 3: Stage 3 (GPU 0,1,2 按任务) | Stage 4 继续 (GPU 3)
  Phase 4: 合并 + 聚合 + 显著性检验

依赖关系:
  Stage 1ml  — 独立
  Stage 1gnn — 独立
  Stage 2    — 依赖 Stage 1gnn (选最佳 backbone)
  Stage 3    — 依赖 Stage 2    (选最佳 config)
  Stage 4    — 独立 (固定 base/ring_conditioned 配置)
"""
import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"

PYTHON = sys.executable
TASKS = ["HOMA", "NICS_1zz", "MBCO"]


def launch(cmd_args, log_name):
    """启动子进程, 返回 Popen 对象"""
    log_path = LOGS / log_name
    f = open(log_path, "w")
    p = subprocess.Popen(cmd_args, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"  [启动] PID={p.pid} -> {log_name}")
    return p


def wait_procs(procs_dict, label):
    """等待多个 Popen 进程完成"""
    pending = dict(procs_dict)
    while pending:
        for name, p in list(pending.items()):
            if p.poll() is not None:
                rc = p.returncode
                status = "完成" if rc == 0 else f"退出码={rc}"
                print(f"  [{status}] {name} (PID {p.pid})")
                del pending[name]
        if pending:
            alive = ", ".join(f"{k}(PID {v.pid})" for k, v in pending.items())
            print(f"  [等待] {label}: {alive} ...", flush=True)
            time.sleep(60)
    print(f"  [全部完成] {label}")


def merge_csvs(main_dir, sub_dirs):
    """合并多个子目录的 per_seed_results.csv 到主目录"""
    main = RESULTS / main_dir
    main.mkdir(parents=True, exist_ok=True)
    dfs = []
    for sub in sub_dirs:
        csv = RESULTS / sub / "per_seed_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            print(f"  [merge] {sub}: {len(df)} 行")
            dfs.append(df)
        else:
            print(f"  [merge] {sub}: 无 CSV (跳过)")
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out = main / "per_seed_results.csv"
        merged.to_csv(out, index=False)
        print(f"  [merge] -> {out}: {len(merged)} 行")
    else:
        print(f"  [merge] 警告: {main_dir} 无可合并 CSV")


def main():
    print("=" * 60)
    print("4-GPU 并行全流程编排 (修复版: parent-molecule split + fold-内预训练)")
    print("=" * 60, flush=True)

    # ====== Phase 1: Stage 1ml + Stage 1gnn + Stage 4 ======
    print("\n>>> [Phase 1] Stage 1ml (CPU) + Stage 1gnn (GPU 0,1,2) + Stage 4 (GPU 3)")

    # Stage 1ml: CPU only (CUDA_VISIBLE_DEVICES="" 隐藏 GPU)
    env_ml = os.environ.copy()
    env_ml["CUDA_VISIBLE_DEVICES"] = ""
    log_f = open(LOGS / "stage1ml.log", "w")
    p_s1ml = subprocess.Popen(
        [PYTHON, "-u", "multiseed/run_multiseed.py", "--stage", "1ml", "--gpu", "0"],
        stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env_ml)
    print(f"  [启动] Stage 1ml (CPU) PID={p_s1ml.pid}")

    # Stage 1gnn: 3 GPU 按模型分组
    s1gnn_groups = [("GNN,GIN", 0, "g0"), ("GAT,MPNN", 1, "g1"), ("GraphSAGE,DMPNN", 2, "g2")]
    s1gnn_procs = {}
    for models, gpu, tag in s1gnn_groups:
        p = launch(
            [PYTHON, "-u", "multiseed/run_multiseed.py", "--stage", "1gnn",
             "--models", models, "--gpu", str(gpu),
             "--output_root", str(RESULTS / f"stage1_gnn_{tag}")],
            f"stage1gnn_{tag}.log")
        s1gnn_procs[f"1gnn_{tag}"] = p

    # Stage 4: GPU 3 (全部 backbones, 独立)
    p_s4 = launch(
        [PYTHON, "-u", "multiseed/run_multiseed.py", "--stage", "4",
         "--backbones", "MPNN,GIN,GAT,DMPNN", "--gpu", "3",
         "--output_root", str(RESULTS / "stage4_g3")],
        "stage4_g3.log")
    s4_proc = {"stage4": p_s4}

    # 等待 Stage 1gnn 完成
    print("\n  [等待] Stage 1gnn (3 GPU) ...")
    wait_procs(s1gnn_procs, "Stage 1gnn")

    # 合并 Stage 1gnn
    print("  [合并] Stage 1gnn 结果...")
    merge_csvs("stage1_gnn", ["stage1_gnn_g0", "stage1_gnn_g1", "stage1_gnn_g2"])

    # ====== Phase 2: Stage 2 ======
    print("\n>>> [Phase 2] Stage 2 (GPU 0,1,2 按任务) | Stage 4 继续 (GPU 3)")
    s2_procs = {}
    for i, task in enumerate(TASKS):
        p = launch(
            [PYTHON, "-u", "multiseed/run_multiseed.py", "--stage", "2",
             "--tasks", task, "--gpu", str(i),
             "--output_root", str(RESULTS / f"stage2_g{i}")],
            f"stage2_{task.lower()}.log")
        s2_procs[f"stage2_{task}"] = p

    wait_procs(s2_procs, "Stage 2")
    merge_csvs("stage2_ring_conditioning", ["stage2_g0", "stage2_g1", "stage2_g2"])

    # ====== Phase 3: Stage 3 ======
    print("\n>>> [Phase 3] Stage 3 (GPU 0,1,2 按任务) | Stage 4 继续 (GPU 3)")
    s3_procs = {}
    for i, task in enumerate(TASKS):
        p = launch(
            [PYTHON, "-u", "multiseed/run_multiseed.py", "--stage", "3",
             "--tasks", task, "--gpu", str(i),
             "--output_root", str(RESULTS / f"stage3_g{i}")],
            f"stage3_{task.lower()}.log")
        s3_procs[f"stage3_{task}"] = p

    wait_procs(s3_procs, "Stage 3")
    merge_csvs("stage3_mask_pretraining", ["stage3_g0", "stage3_g1", "stage3_g2"])

    # ====== Phase 4: 等待 Stage 4 + Stage 1ml ======
    print("\n>>> [Phase 4] 等待 Stage 4 + Stage 1ml 完成...")
    wait_procs(s4_proc, "Stage 4")
    merge_csvs("stage4_cross_arch", ["stage4_g3"])

    # 等待 Stage 1ml
    s1ml_proc = {"stage1ml": p_s1ml}
    wait_procs(s1ml_proc, "Stage 1ml")

    # ====== Phase 5: 聚合 + 显著性检验 ======
    print("\n>>> [Phase 5] 全局聚合 + 显著性检验")
    subprocess.run([PYTHON, "multiseed/run_multiseed.py", "--aggregate_only"], cwd=str(ROOT))
    subprocess.run([PYTHON, "stats/significance_test.py"], cwd=str(ROOT))

    print("\n" + "=" * 60)
    print("全部实验完成!")
    print(f"  结果目录: {RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
