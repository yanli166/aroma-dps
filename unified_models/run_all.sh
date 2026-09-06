#!/bin/bash
# 运行所有15个模型 (5种模型 × 3种编码方式)
# 使用统一的超参数配置

set -e
export MPLBACKEND=Agg

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

# 所有模型和编码方式的组合
MODELS=("gat" "gin" "gnn" "mpnn" "graphsage")
MODES=("label" "mask" "pool")

for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        OUT_DIR=$RESULTS_ROOT/${model}_${mode}
        LOG_FILE=$OUT_DIR/training.log

        if [ -f "$OUT_DIR/summary.csv" ]; then
            echo "跳过 ${model}-${mode} (已完成)"
            continue
        fi

        echo "============================================"
        echo "运行 ${model}-${mode}..."
        echo "============================================"
        mkdir -p $OUT_DIR

        python unified_models/train.py \
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
            > $LOG_FILE 2>&1

        echo "${model}-${mode} 完成"
    done
done

echo ""
echo "============================================"
echo "所有模型训练完成!"
echo "============================================"
