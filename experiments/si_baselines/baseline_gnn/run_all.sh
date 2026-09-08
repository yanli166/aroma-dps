#!/bin/bash
# 第二层: 通用GNN基线 完整运行脚本
# 6个模型 × 3任务 × 5折CV
set -e
export MPLBACKEND=Agg
source /home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch_env

cd /home/ubuntu/aroma-dps-code
RESULTS=/home/ubuntu/aroma-dps-code/baseline_gnn/results
mkdir -p $RESULTS

python -u baseline_gnn/code/gnn_train_eval.py \
    --output_dir $RESULTS \
    --models all \
    --tasks all \
    --n_epochs 200 \
    --patience 30 \
    --seed 42 \
    --gpu 0 \
    2>&1 | tee $RESULTS/run.log

echo "第二层GNN基线实验完成"
