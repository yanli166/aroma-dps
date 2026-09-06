#!/bin/bash
# 多种子 × 5折CV 实验总调度 (修复版)
#
# 修复项:
#   - 所有代码从 code_end/ 加载, 不改动原始代码
#   - 5 个种子: 42, 123, 456, 789, 2024
#   - Layer 3 增加 'none' 编码 (M3: 无环信息 ablation 对照)
#   - 统计检验 (M4) 在实验完成后自动运行
#
# 4 GPU 并行策略:
#   GPU 0: Layer 1 ML (CPU为主, 快速完成) + Layer 2 GNN,GIN
#   GPU 1: Layer 2 GAT,MPNN + Layer 3 GNN
#   GPU 2: Layer 2 GraphSAGE,AttentiveFP + Layer 3 GIN,GAT
#   GPU 3: Layer 2 DMPNN + Layer 3 MPNN,GraphSAGE
#
# 预计耗时:
#   Layer 1: ~30 分钟
#   Layer 2: ~3-5 小时 (4 GPU 并行)
#   Layer 3: ~6-10 小时 (4 GPU 并行, 含 none 编码)

set -e

CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
PROJ_ROOT=${SCRIPT_DIR}
CODE_ROOT=$PROJ_ROOT/code_end
OUT_ROOT=$CODE_ROOT/results
LOG_DIR=$OUT_ROOT/logs

source $CONDA_SH && conda activate torch_env

SEEDS="42,123,456,789,2024"
EPOCHS=200
PATIENCE=30

# 设置环境变量 (m4: 路径配置化)
export AROMA_DATA_ROOT=${SCRIPT_DIR}/unified_models
export AROMA_PROJ_ROOT=$PROJ_ROOT
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

mkdir -p $LOG_DIR

echo "============================================"
echo "多种子实验开始 (修复版): $(date)"
echo "种子: $SEEDS"
echo "代码目录: $CODE_ROOT"
echo "结果目录: $OUT_ROOT"
echo "============================================"

# ---- Layer 1: ML (GPU 0, CPU为主) ----
echo "[Layer 1 ML] 启动..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 1 --gpu 0 --seeds $SEEDS \
    --output_root $OUT_ROOT/layer1_ml \
    > $LOG_DIR/layer1_ml.log 2>&1 &
PID_L1=$!
echo "  PID=$PID_L1, 日志: $LOG_DIR/layer1_ml.log"

# ---- Layer 2: GNN (4 GPU 并行) ----
echo "[Layer 2 GNN] 启动 (4 GPU 并行)..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 2 --gpu 0 --seeds $SEEDS --models GNN,GIN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer2_gnn_part1 \
    > $LOG_DIR/layer2_gnn_gpu0.log 2>&1 &
PID_L2A=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 2 --gpu 1 --seeds $SEEDS --models GAT,MPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer2_gnn_part2 \
    > $LOG_DIR/layer2_gnn_gpu1.log 2>&1 &
PID_L2B=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 2 --gpu 2 --seeds $SEEDS --models GraphSAGE,AttentiveFP \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer2_gnn_part3 \
    > $LOG_DIR/layer2_gnn_gpu2.log 2>&1 &
PID_L2C=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 2 --gpu 3 --seeds $SEEDS --models DMPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer2_gnn_part4 \
    > $LOG_DIR/layer2_gnn_gpu3.log 2>&1 &
PID_L2D=$!

echo "  GPU0: GNN,GIN (PID=$PID_L2A)"
echo "  GPU1: GAT,MPNN (PID=$PID_L2B)"
echo "  GPU2: GraphSAGE,AttentiveFP (PID=$PID_L2C)"
echo "  GPU3: DMPNN (PID=$PID_L2D)"

# ---- Layer 3: Ring Encoding (4 GPU 并行) ----
# M3: 包含 'none' 编码 (无环信息 ablation 对照)
echo "[Layer 3 Ring] 启动 (4 GPU 并行, 含 none 编码)..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 3 --gpu 0 --seeds $SEEDS --models GNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer3_ring_part1 \
    > $LOG_DIR/layer3_ring_gpu0.log 2>&1 &
PID_L3A=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 3 --gpu 1 --seeds $SEEDS --models GIN,GAT \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer3_ring_part2 \
    > $LOG_DIR/layer3_ring_gpu1.log 2>&1 &
PID_L3B=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 3 --gpu 2 --seeds $SEEDS --models MPNN \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer3_ring_part3 \
    > $LOG_DIR/layer3_ring_gpu2.log 2>&1 &
PID_L3C=$!

python $CODE_ROOT/multiseed/run_multiseed.py --layer 3 --gpu 3 --seeds $SEEDS --models GraphSAGE \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer3_ring_part4 \
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

python $CODE_ROOT/multiseed/run_multiseed.py --aggregate_only

# ---- M4: 统计显著性检验 ----
echo ""
echo "============================================"
echo "运行统计显著性检验 (M4)"
echo "============================================"

python $CODE_ROOT/stats/significance_test.py \
    --results_root $OUT_ROOT

echo ""
echo "============================================"
echo "全部完成: $(date)"
echo "结果目录: $OUT_ROOT"
echo "============================================"
