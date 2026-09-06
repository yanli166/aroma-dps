#!/bin/bash
# Optuna超参调优 - Top 3模型并行
set -e
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." cd ${SCRIPT_DIR}cd ${SCRIPT_DIR} pwd)"
DATASET=${SCRIPT_DIR}/nics-nics1zz-out-no3.csv
N_TRIALS=15
N_EPOCHS=50

# Top 3模型 (按R²排名)
# 1. gnn-label (R²=0.9786)
# 2. graphsage-label (R²=0.9769)  
# 3. mpnn-label (R²=0.9761)

echo "开始 Optuna 超参调优 (3个模型并行, 每个15 trials, 50 epochs/trial)..."

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u unified_models/optimize.py \
    --model gnn --mode label \
    --dataset_path $DATASET \
    --n_trials $N_TRIALS --n_epochs $N_EPOCHS \
    --study_name gnn_label_opt \
    > ${SCRIPT_DIR}/0427_unified_results/optuna_gnn_label.log 2>&1 &
PID1=$!

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u unified_models/optimize.py \
    --model graphsage --mode label \
    --dataset_path $DATASET \
    --n_trials $N_TRIALS --n_epochs $N_EPOCHS \
    --study_name graphsage_label_opt \
    > ${SCRIPT_DIR}/0427_unified_results/optuna_graphsage_label.log 2>&1 &
PID2=$!

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u unified_models/optimize.py \
    --model mpnn --mode label \
    --dataset_path $DATASET \
    --n_trials $N_TRIALS --n_epochs $N_EPOCHS \
    --study_name mpnn_label_opt \
    > ${SCRIPT_DIR}/0427_unified_results/optuna_mpnn_label.log 2>&1 &
PID3=$!

echo "启动: gnn-label(PID=$PID1), graphsage-label(PID=$PID2), mpnn-label(PID=$PID3)"

wait $PID1 || echo "WARN: gnn-label optuna failed"
wait $PID2 || echo "WARN: graphsage-label optuna failed"
wait $PID3 || echo "WARN: mpnn-label optuna failed"

echo ""
echo "============================================"
echo "Optuna 超参调优完成!"
echo "============================================"

# 输出最佳结果
for study in gnn_label_opt graphsage_label_opt mpnn_label_opt; do
    BEST_FILE=${SCRIPT_DIR}/unified_optuna_${study}/best_parameters.csv
    if [ -f "$BEST_FILE" ]; then
        echo ""
        echo "=== $study 最佳参数 ==="
        cat $BEST_FILE
    fi
done
