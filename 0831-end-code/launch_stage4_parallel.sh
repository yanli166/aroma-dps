#!/bin/bash
# Stage 4 并行加速脚本
# 将剩余 Stage 4 任务按 backbone 分配到 4 个 GPU:
#   GPU 0: MPNN   (seeds 123,456,789,2024)  -> stage4_g0_parallel
#   GPU 1: GIN    (seeds 123,456,789,2024)  -> stage4_g1_parallel
#   GPU 2: GAT    (seeds 123,456,789,2024)  -> stage4_g2_parallel
#   GPU 3: 继续原进程 (全部 backbone, 已有 seed_42 部分结果)
#
# seed_42 已由 GPU3 完成 (HOMA/NICS_1zz 全部, MBCO 6/8), 故新 worker 跳过 seed_42
# DMPNN 仅由 GPU3 处理 (PyG 模型, 需独立数据加载)
#
# 预计: 每 GPU ~24 configs × ~40min = ~16h
set -e

source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
conda activate torch_env

ROOT="${SCRIPT_DIR}/last_end_code"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON=$(which python)

SEEDS="123,456,789,2024"

echo "============================================================"
echo "Stage 4 并行加速启动 $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU 0: MPNN  (seeds $SEEDS)"
echo "  GPU 1: GIN   (seeds $SEEDS)"
echo "  GPU 2: GAT   (seeds $SEEDS)"
echo "  GPU 3: 继续原进程 (全部 backbone)"
echo "============================================================"

# GPU 0: MPNN
nohup $PYTHON -u "$ROOT/multiseed/run_multiseed.py" \
    --stage 4 --backbones MPNN --seeds "$SEEDS" --gpu 0 \
    --output_root "$RESULTS/stage4_g0_parallel" \
    > "$LOGS/stage4_g0_parallel.log" 2>&1 &
PID_G0=$!
echo "  [启动] GPU 0 MPNN: PID=$PID_G0"

# GPU 1: GIN
nohup $PYTHON -u "$ROOT/multiseed/run_multiseed.py" \
    --stage 4 --backbones GIN --seeds "$SEEDS" --gpu 1 \
    --output_root "$RESULTS/stage4_g1_parallel" \
    > "$LOGS/stage4_g1_parallel.log" 2>&1 &
PID_G1=$!
echo "  [启动] GPU 1 GIN: PID=$PID_G1"

# GPU 2: GAT
nohup $PYTHON -u "$ROOT/multiseed/run_multiseed.py" \
    --stage 4 --backbones GAT --seeds "$SEEDS" --gpu 2 \
    --output_root "$RESULTS/stage4_g2_parallel" \
    > "$LOGS/stage4_g2_parallel.log" 2>&1 &
PID_G2=$!
echo "  [启动] GPU 2 GAT: PID=$PID_G2"

echo ""
echo "PIDs: G0=$PID_G0 G1=$PID_G1 G2=$PID_G2"
echo "GPU 3 原进程继续运行 (PID 2397280)"
echo ""
echo "日志:"
echo "  tail -f $LOGS/stage4_g0_parallel.log"
echo "  tail -f $LOGS/stage4_g1_parallel.log"
echo "  tail -f $LOGS/stage4_g2_parallel.log"
echo ""
echo "完成后运行合并脚本:"
echo "  python $ROOT/merge_stage4_parallel.py"
