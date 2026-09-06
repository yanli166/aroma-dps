#!/bin/bash
# 并行运行所有15个模型 (5种模型 × 3种编码方式)
# 每批3个模型并行，每个模型使用4个CPU线程

set -e
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." cd ${SCRIPT_DIR}cd ${SCRIPT_DIR} pwd)"

DATASET=${SCRIPT_DIR}/nics-nics1zz-out-no3.csv
RESULTS_ROOT=${SCRIPT_DIR}/0427_unified_results
EPOCHS=200
BATCH_SIZE=64
HIDDEN_DIM=128
N_CONV=3
N_HIDDEN=2
LR=0.001
PATIENCE=30

mkdir -p $RESULTS_ROOT

# 所有15个实验 (按速度排序: 快的先跑)
EXPERIMENTS=(
    "gnn label"
    "gnn mask"
    "gnn pool"
    "graphsage label"
    "graphsage mask"
    "graphsage pool"
    "mpnn label"
    "mpnn mask"
    "mpnn pool"
    "gin label"
    "gin mask"
    "gin pool"
    "gat label"
    "gat mask"
    "gat pool"
)

run_experiment() {
    local model=$1
    local mode=$2
    local OUT_DIR=$RESULTS_ROOT/${model}_${mode}
    local LOG_FILE=$OUT_DIR/training.log

    if [ -f "$OUT_DIR/summary.csv" ]; then
        echo "SKIP ${model}-${mode} (已完成)"
        return 0
    fi

    echo "START ${model}-${mode}"
    mkdir -p $OUT_DIR

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u unified_models/train.py \
        --model $model \
        --mode $mode \
        --dataset_path $DATASET \
        --output_dir $OUT_DIR \
        --n_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --n_conv_layers $N_CONV \
        --n_hidden_layers $N_HIDDEN \
        --learning_rate $LR \
        --patience $PATIENCE \
        --splitter random \
        --seed 42 \
        --n_threads 4 \
        > $LOG_FILE 2>&1

    echo "DONE ${model}-${mode}"
}

# 每3个并行运行
BATCH_SIZE_PARALLEL=3
COUNT=0
PIDS=()

for exp in "${EXPERIMENTS[@]}"; do
    model=$(echo $exp | awk '{print $1}')
    mode=$(echo $exp | awk '{print $2}')

    run_experiment $model $mode &
    PIDS+=($!)
    COUNT=$((COUNT + 1))

    if [ $((COUNT % BATCH_SIZE_PARALLEL)) -eq 0 ]; then
        for pid in "${PIDS[@]}"; do
            wait $pid || echo "WARN: process $pid failed"
        done
        PIDS=()
        echo "--- 批次完成 ($COUNT / ${#EXPERIMENTS[@]}) ---"
    fi
done

# 等待剩余的
for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARN: process $pid failed"
done

echo ""
echo "============================================"
echo "所有15个模型训练完成!"
echo "============================================"
echo ""
echo "结果汇总:"
for exp in "${EXPERIMENTS[@]}"; do
    model=$(echo $exp | awk '{print $1}')
    mode=$(echo $exp | awk '{print $2}')
    SUMMARY=$RESULTS_ROOT/${model}_${mode}/summary.csv
    if [ -f "$SUMMARY" ]; then
        VAL_R2=$(grep 'final_val_r2' $SUMMARY | cut -d',' -f2)
        VAL_MAE=$(grep 'final_val_mae' $SUMMARY | cut -d',' -f2)
        MAX_R2=$(grep 'max_val_r2' $SUMMARY | cut -d',' -f2)
        echo "  ${model}-${mode}: Val R²=${VAL_R2}, Max R²=${MAX_R2}, MAE=${VAL_MAE}"
    else
        echo "  ${model}-${mode}: 未完成"
    fi
done
