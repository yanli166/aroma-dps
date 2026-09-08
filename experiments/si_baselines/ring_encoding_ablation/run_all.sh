#!/bin/bash
# 第三层: 三类环信息编码范式对比实验 完整运行脚本
# top5模型 × (label/mask/pool/combined) × 3任务 × 5折CV
set -e
export MPLBACKEND=Agg
source /home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch_env

cd /home/ubuntu/aroma-dps-code
RESULTS=/home/ubuntu/aroma-dps-code/ring_encoding_ablation/results
mkdir -p $RESULTS

# 依赖第二层结果文件 all_gnn_summary.csv 自动选取 top5
# 使用 GPU 2 (GPU 0 给GNN基线, GPU 1 给ML树模型)
python -u ring_encoding_ablation/code/ring_train_eval.py \
    --output_dir $RESULTS \
    --gnn_results /home/ubuntu/aroma-dps-code/baseline_gnn/results/all_gnn_summary.csv \
    --tasks all \
    --n_epochs 200 \
    --patience 30 \
    --seed 42 \
    --gpu 2 \
    2>&1 | tee $RESULTS/run.log

echo "第三层环编码消融实验完成"
