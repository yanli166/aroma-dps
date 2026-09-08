#!/bin/bash
# 第一层: 传统ML基线 完整运行脚本
# 9个模型 × 3任务 × 5折CV
# CatBoost/XGBoost/LightGBM 使用 GPU 1 加速, sklearn 模型使用 CPU
set -e
export MPLBACKEND=Agg
source /home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch_env

cd /home/ubuntu/aroma-dps-code
RESULTS=/home/ubuntu/aroma-dps-code/baseline_traditional_ml/results
mkdir -p $RESULTS

python -u baseline_traditional_ml/code/ml_train_eval.py \
    --output_dir $RESULTS \
    --seed 42 \
    --tasks all \
    --gpu 1 \
    2>&1 | tee $RESULTS/run.log

echo "第一层传统ML实验完成"
