#!/bin/bash
# Stage4 (v2) 并行启动: 每个 backbone 各占一个 GPU, 写入 results_v2/stage4/partial/{BB}/
set -e
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
conda activate torch_env

ROOT="${SCRIPT_DIR}/0831-end-code"
RESULTS="$ROOT/results_v2/stage4"
PYTHON="$ROOT/stage4_cross_architecture/code/run_stage4_v2.py"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

SEED=11
EPOCHS=30
PATIENCE=8
PIDS=""

declare -A BB_GPU=( [MPNN]=0 [GIN]=1 [GAT]=2 [DMPNN]=3 )
for BB in MPNN GIN GAT DMPNN; do
  GPU=${BB_GPU[$BB]}
  OUT="$RESULTS/partial/$BB"
  LOG="$RESULTS/logs/stage4_${BB}.log"
  mkdir -p "$OUT" "$RESULTS/logs"
  nohup python -u "$PYTHON" --output_root "$OUT" \
      --backbones "$BB" --tasks HOMA,NICS_1zz,MBCO \
      --seed $SEED --n_epochs $EPOCHS --patience $PATIENCE --gpu $GPU \
      > "$LOG" 2>&1 &
  pid=$!
  PIDS="$PIDS $pid"
  echo "[启动] $BB -> GPU$GPU PID=$pid log=$LOG"
done

echo ""
echo "PIDs:$PIDS"
echo "全部完成后再运行聚合:"
echo "  python $ROOT/stage4_cross_architecture/code/aggregate_stage4_v2.py"