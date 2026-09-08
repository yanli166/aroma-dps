#!/usr/bin/env python3
"""
lunci10 并行 g16 优化运行器
- 改写 .gjf 的 %mem / %nprocshared 为指定值
- 用 ProcessPoolExecutor 并行运行 g16
- 每个 worker 独立 GAUSS_SCRDIR，退出自动清理
- 失败重试一次
"""
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============== 配置 ==============
GJF_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/aroma-dps-code/lunci10/gaussian_inputs")
SCRATCH_BASE = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci10/scratch")
G16_BIN = "/home/ubuntu/apps/g16/g16"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MEM_GB = int(sys.argv[4]) if len(sys.argv) > 4 else 20
PARALLEL = int(sys.argv[5]) if len(sys.argv) > 5 else 12
MAX_RETRY = 1
# ==================================

MEM_LINE = f"%mem={MEM_GB}GB"
NPROC_LINE = f"%nprocshared={NPROC}"


def rewrite_gjf(gjf_path: Path) -> None:
    """改写 .gjf 前两行为新的 %mem / %nprocshared"""
    txt = gjf_path.read_text()
    lines = txt.split("\n")
    # 找到 %mem 和 %nprocshared 行并替换
    new_lines = []
    mem_done = nproc_done = False
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("%mem=") and not mem_done:
            new_lines.append(MEM_LINE)
            mem_done = True
        elif s.startswith("%nprocshared") and not nproc_done:
            new_lines.append(NPROC_LINE)
            nproc_done = True
        else:
            new_lines.append(ln)
    if not mem_done:
        new_lines.insert(0, MEM_LINE)
    if not nproc_done:
        new_lines.insert(1 if mem_done else 0, NPROC_LINE)
    gjf_path.write_text("\n".join(new_lines))


def run_one_g16(gjf_path: str, attempt: int = 0) -> tuple:
    """运行一个 g16 任务，返回 (gjf_name, success, msg, attempt)"""
    gjf = Path(gjf_path)
    name = gjf.stem
    log_path = gjf.with_suffix(".log")

    # 已存在 .log 且正常完成则跳过
    if log_path.exists() and log_path.stat().st_size > 1000:
        tail = log_path.read_text()[-2000:]
        if "Normal termination" in tail:
            return (name, True, "already_done", attempt)

    # 创建专属 scratch
    SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
    job_scratch = Path(subprocess.check_output(
        ["mktemp", "-d", "-p", str(SCRATCH_BASE), f"g16_{name}_XXXXXX"]
    ).decode().strip())

    try:
        env = os.environ.copy()
        env["GAUSS_SCRDIR"] = str(job_scratch)
        env["GAUSS_EXEDIR"] = "/home/ubuntu/apps/g16"
        env["OMP_NUM_THREADS"] = str(NPROC)
        env["MKL_NUM_THREADS"] = str(NPROC)
        env["PGI_FASTMATH_CPU"] = "sandybridge"

        # 运行 g16
        with open(log_path, "w") as logf:
            result = subprocess.run(
                [G16_BIN],
                stdin=open(gjf, "r"),
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(GJF_DIR),
                env=env,
                timeout=3600,  # 单任务最长 1 小时
            )

        # 检查结果
        if result.returncode != 0:
            return (name, False, f"exit={result.returncode}", attempt)

        # 验证 Normal termination
        if log_path.exists():
            tail = log_path.read_text()[-2000:]
            if "Normal termination" in tail:
                return (name, True, "ok", attempt)
            else:
                return (name, False, "no_normal_termination", attempt)
        else:
            return (name, False, "no_log", attempt)
    except subprocess.TimeoutExpired:
        return (name, False, "timeout", attempt)
    except Exception as e:
        return (name, False, f"exc:{e}", attempt)
    finally:
        # 清理 scratch
        try:
            shutil.rmtree(job_scratch, ignore_errors=True)
        except Exception:
            pass


def main():
    # 1. 收集所有 .gjf
    gjf_files = sorted(GJF_DIR.glob("*.gjf"))
    total = len(gjf_files)
    print(f"[{time.strftime('%H:%M:%S')}] Total gjf: {total}, parallel={PARALLEL}, nproc={NPROC}, mem={MEM_GB}GB", flush=True)

    # 2. 改写所有 .gjf
    print(f"[{time.strftime('%H:%M:%S')}] Rewriting %mem / %nprocshared ...", flush=True)
    for g in gjf_files:
        rewrite_gjf(g)
    print(f"[{time.strftime('%H:%M:%S')}] Rewrite done.", flush=True)

    # 3. 第一轮并行
    print(f"[{time.strftime('%H:%M:%S')}] Round 1 starting ...", flush=True)
    success = set()
    failed = []
    done_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=PARALLEL) as ex:
        futures = {ex.submit(run_one_g16, str(g), 0): g.stem for g in gjf_files}
        for fut in as_completed(futures):
            name, ok, msg, attempt = fut.result()
            done_count += 1
            if ok:
                success.add(name)
            else:
                failed.append((name, msg, attempt))
            if done_count % 20 == 0 or done_count == total:
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(f"[{time.strftime('%H:%M:%S')}] {done_count}/{total} done, "
                      f"success={len(success)}, fail={len(failed)}, "
                      f"rate={rate:.2f}/s, ETA={eta/60:.1f}min | last={name}:{msg}", flush=True)

    # 4. 失败重试
    for retry in range(1, MAX_RETRY + 2):
        if not failed:
            break
        print(f"[{time.strftime('%H:%M:%S')}] Retry {retry}: {len(failed)} failed tasks", flush=True)
        retry_list = failed
        failed = []
        with ProcessPoolExecutor(max_workers=PARALLEL) as ex:
            futures = {ex.submit(run_one_g16, str(GJF_DIR / f"{n}.gjf"), retry): n
                       for n, _, _ in retry_list}
            for fut in as_completed(futures):
                name, ok, msg, attempt = fut.result()
                if ok:
                    success.add(name)
                    print(f"[{time.strftime('%H:%M:%S')}] Retry OK: {name}", flush=True)
                else:
                    failed.append((name, msg, attempt))
                    print(f"[{time.strftime('%H:%M:%S')}] Retry FAIL: {name}:{msg}", flush=True)

    # 5. 汇总
    elapsed = time.time() - start_time
    print("=" * 60, flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] FINISHED: {len(success)}/{total} success, "
          f"{len(failed)} failed, total time={elapsed/60:.1f}min", flush=True)
    if failed:
        print("Failed list:", flush=True)
        for n, msg, _ in failed:
            print(f"  {n}: {msg}", flush=True)
        # 保存失败列表
        with open(GJF_DIR / "g16_failed.txt", "w") as f:
            for n, msg, _ in failed:
                f.write(f"{n}\t{msg}\n")


if __name__ == "__main__":
    main()
