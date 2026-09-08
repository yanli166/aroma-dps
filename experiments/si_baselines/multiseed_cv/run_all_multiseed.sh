#!/bin/bash
# 多种子 × 5折CV 实验总调度
#
# 5 个种子: 42, 123, 456, 789, 2024
# 4 个 GPU 并行:
#   GPU 0: Layer 1 ML (CPU为主, 快速完成)
#   GPU 1: Layer 2 GNN (7模型 × 3任务 × 5种子)
#   GPU 2: Layer 3 Ring (5模型 × 4编码 × 3任务 × 5种子)
#   GPU 3: Layer 2 备用 (分担部分模型) / Layer 3 备用
#
# 预计耗时:
#   Layer 1: ~30 分钟
#   Layer 2: ~3-5 小时 (单GPU), 多GPU分担可缩短
#   Layer 3: ~6-10 小时 (单GPU), 多GPU分担可缩短

set -e

CONDA_SH=/home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh
PROJ_ROOT=/home/ubuntu/aroma-dps-code
OUT_ROOT=$PROJ_ROOT/new5zhecv
LOG_DIR=$OUT_ROOT/logs

source $CONDA_SH && conda activate torch_env

SEEDS="42,123,456,789,2024"
EPOCHS=200
PATIENCE=30

mkdir -p $LOG_DIR

echo "============================================"
echo "多种子实验开始: $(date)"
echo "种子: $SEEDS"
echo "============================================"

# ---- Layer 1: ML (GPU 0, CPU为主) ----
echo "[Layer 1 ML] 启动..."
python $OUT_ROOT/code/run_multiseed.py --layer 1 --gpu 0 --seeds $SEEDS \
    > $LOG_DIR/layer1_ml.log 2>&1 &
PID_L1=$!
echo "  PID=$PID_L1, 日志: $LOG_DIR/layer1_ml.log"

# ---- Layer 2: GNN ----
# 拆分模型到不同 GPU 以并行:
#   GPU 0: GNN, GIN (Layer1完成后接力)
#   GPU 1: GAT, MPNN
#   GPU 2: GraphSAGE, AttentiveFP
#   GPU 3: DMPNN
echo "[Layer 2 GNN] 启动 (4 GPU 并行)..."
python $OUT_ROOT/code/run_multiseed.py --layer 2 --gpu 0 --seeds $SEEDS --models GNN,GIN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer2_gnn_part1 \
    > $LOG_DIR/layer2_gnn_gpu0.log 2>&1 &
PID_L2A=$!

python $OUT_ROOT/code/run_multiseed.py --layer 2 --gpu 1 --seeds $SEEDS --models GAT,MPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer2_gnn_part2 \
    > $LOG_DIR/layer2_gnn_gpu1.log 2>&1 &
PID_L2B=$!

python $OUT_ROOT/code/run_multiseed.py --layer 2 --gpu 2 --seeds $SEEDS --models GraphSAGE,AttentiveFP \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer2_gnn_part3 \
    > $LOG_DIR/layer2_gnn_gpu2.log 2>&1 &
PID_L2C=$!

python $OUT_ROOT/code/run_multiseed.py --layer 2 --gpu 3 --seeds $SEEDS --models DMPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer2_gnn_part4 \
    > $LOG_DIR/layer2_gnn_gpu3.log 2>&1 &
PID_L2D=$!

echo "  GPU0: GNN,GIN (PID=$PID_L2A)"
echo "  GPU1: GAT,MPNN (PID=$PID_L2B)"
echo "  GPU2: GraphSAGE,AttentiveFP (PID=$PID_L2C)"
echo "  GPU3: DMPNN (PID=$PID_L2D)"

# ---- Layer 3: Ring Encoding ----
# Layer 3 依赖 Layer 2 结果选 top5, 但我们直接用全部5个自定义模型
# 拆分模型到不同 GPU:
#   GPU 0: GNN (Layer2完成后接力, 这里先启动其他)
#   GPU 1: GIN, GAT
#   GPU 2: MPNN
#   GPU 3: GraphSAGE
echo "[Layer 3 Ring] 启动 (4 GPU 并行)..."
python $OUT_ROOT/code/run_multiseed.py --layer 3 --gpu 0 --seeds $SEEDS --models GNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer3_ring_part1 \
    > $LOG_DIR/layer3_ring_gpu0.log 2>&1 &
PID_L3A=$!

python $OUT_ROOT/code/run_multiseed.py --layer 3 --gpu 1 --seeds $SEEDS --models GIN,GAT \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer3_ring_part2 \
    > $LOG_DIR/layer3_ring_gpu1.log 2>&1 &
PID_L3B=$!

python $OUT_ROOT/code/run_multiseed.py --layer 3 --gpu 2 --seeds $SEEDS --models MPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer3_ring_part3 \
    > $LOG_DIR/layer3_ring_gpu2.log 2>&1 &
PID_L3C=$!

python $OUT_ROOT/code/run_multiseed.py --layer 3 --gpu 3 --seeds $SEEDS --models GraphSAGE \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/results/layer3_ring_part4 \
    > $LOG_DIR/layer3_ring_gpu3.log 2>&1 &
PID_L3D=$!

echo "  GPU0: GNN (PID=$PID_L3A)"
echo "  GPU1: GIN,GAT (PID=$PID_L3B)"
echo "  GPU2: MPNN (PID=$PID_L3C)"
echo "  GPU3: GraphSAGE (PID=$PID_L3D)"

# ---- 等待所有任务完成 ----
echo ""
echo "等待所有任务完成..."
wait $PID_L1 && echo "[Layer 1 ML] 完成" || echo "[Layer 1 ML] 异常退出"
wait $PID_L2A && echo "[Layer 2 GPU0] 完成" || echo "[Layer 2 GPU0] 异常"
wait $PID_L2B && echo "[Layer 2 GPU1] 完成" || echo "[Layer 2 GPU1] 异常"
wait $PID_L2C && echo "[Layer 2 GPU2] 完成" || echo "[Layer 2 GPU2] 异常"
wait $PID_L2D && echo "[Layer 2 GPU3] 完成" || echo "[Layer 2 GPU3] 异常"
wait $PID_L3A && echo "[Layer 3 GPU0] 完成" || echo "[Layer 3 GPU0] 异常"
wait $PID_L3B && echo "[Layer 3 GPU1] 完成" || echo "[Layer 3 GPU1] 异常"
wait $PID_L3C && echo "[Layer 3 GPU2] 完成" || echo "[Layer 3 GPU2] 异常"
wait $PID_L3D && echo "[Layer 3 GPU3] 完成" || echo "[Layer 3 GPU3] 异常"

# ---- 合并分片结果并聚合 ----
echo ""
echo "============================================"
echo "合并分片结果并聚合"
echo "============================================"

python3 -c "
import pandas as pd
import os

PROJ_ROOT = '/home/ubuntu/aroma-dps-code'
OUT_ROOT = os.path.join(PROJ_ROOT, 'new5zhecv')

# Layer 2: 合并 4 个分片
l2_parts = []
for i in range(1, 5):
    p = os.path.join(OUT_ROOT, 'results', f'layer2_gnn_part{i}', 'per_seed_results.csv')
    if os.path.exists(p):
        l2_parts.append(pd.read_csv(p))
if l2_parts:
    df_l2 = pd.concat(l2_parts, ignore_index=True)
    l2_dir = os.path.join(OUT_ROOT, 'results', 'layer2_gnn')
    os.makedirs(l2_dir, exist_ok=True)
    df_l2.to_csv(os.path.join(l2_dir, 'per_seed_results.csv'), index=False)
    print(f'Layer 2 合并: {len(df_l2)} 行')

# Layer 3: 合并 4 个分片
l3_parts = []
for i in range(1, 5):
    p = os.path.join(OUT_ROOT, 'results', f'layer3_ring_part{i}', 'per_seed_results.csv')
    if os.path.exists(p):
        l3_parts.append(pd.read_csv(p))
if l3_parts:
    df_l3 = pd.concat(l3_parts, ignore_index=True)
    l3_dir = os.path.join(OUT_ROOT, 'results', 'layer3_ring')
    os.makedirs(l3_dir, exist_ok=True)
    df_l3.to_csv(os.path.join(l3_dir, 'per_seed_results.csv'), index=False)
    print(f'Layer 3 合并: {len(df_l3)} 行')
"

# 聚合所有层
python $OUT_ROOT/code/run_multiseed.py --aggregate_only

echo ""
echo "============================================"
echo "全部完成: $(date)"
echo "结果目录: $OUT_ROOT/results/"
echo "============================================"
