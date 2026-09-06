#!/bin/bash
# 完整训练 + 四种集外测试 (0716 新数据)
#
# 三层训练 (单种子 seed=42):
#   GPU 0: Layer 1 ML (9模型 × 3任务)
#   GPU 1: Layer 2 GNN (7模型 × 3任务)
#   GPU 2: Layer 3 Ring (5模型 × 5编码 × 3任务, 含 none)
#
# 四种集外测试 (GPU 3):
#   1. scaffold: 骨架拆分 (DeepChem)
#   2. ring_type: 芳环类型拆分
#   3. lunci6: lunci6 集外测试
#   4. lunci78: lunci78 集外测试

set -e

CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
PROJ_ROOT=${SCRIPT_DIR}
CODE_ROOT=$PROJ_ROOT/code_end
OUT_ROOT=$CODE_ROOT/results
LOG_DIR=$OUT_ROOT/logs

source $CONDA_SH && conda activate torch_env

SEED=42
EPOCHS=200
PATIENCE=30

export AROMA_DATA_ROOT=${SCRIPT_DIR}/unified_models
export DATA1_END_DIR=$CODE_ROOT/data1_end
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

mkdir -p $LOG_DIR

echo "============================================"
echo "完整训练 + 集外测试 (0716新数据): $(date)"
echo "种子: $SEED"
echo "============================================"

# ---- Layer 1: ML (GPU 0) ----
echo "[Layer 1 ML] 启动 (GPU 0)..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 1 --gpu 0 --seeds $SEED \
    --output_root $OUT_ROOT/layer1_ml \
    > $LOG_DIR/layer1_ml.log 2>&1 &
PID_L1=$!

# ---- Layer 2: GNN (GPU 1) ----
echo "[Layer 2 GNN] 启动 (GPU 1)..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 2 --gpu 1 --seeds $SEED \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer2_gnn \
    > $LOG_DIR/layer2_gnn.log 2>&1 &
PID_L2=$!

# ---- Layer 3: Ring Encoding (GPU 2) ----
echo "[Layer 3 Ring] 启动 (GPU 2)..."
python $CODE_ROOT/multiseed/run_multiseed.py --layer 3 --gpu 2 --seeds $SEED \
    --n_epochs $EPOCHS --patience $PATIENCE \
    --output_root $OUT_ROOT/layer3_ring \
    > $LOG_DIR/layer3_ring.log 2>&1 &
PID_L3=$!

# ---- 泛化测试 (GPU 3, 与训练并行) ----
echo "[泛化测试] 启动 (GPU 3)..."
python $CODE_ROOT/generalization_test/code/run_all.py \
    --gpu 3 --tasks all \
    --test_types scaffold,ring_type,lunci6,lunci78 \
    --n_epochs $EPOCHS --patience $PATIENCE --seed $SEED \
    --output_dir $OUT_ROOT/generalization \
    > $LOG_DIR/generalization.log 2>&1 &
PID_GEN=$!

echo ""
echo "PID: L1=$PID_L1, L2=$PID_L2, L3=$PID_L3, GEN=$PID_GEN"
echo "等待所有任务完成..."

wait $PID_L1 && echo "[Layer 1 ML] 完成" || echo "[Layer 1 ML] 异常"
wait $PID_L2 && echo "[Layer 2 GNN] 完成" || echo "[Layer 2 GNN] 异常"
wait $PID_L3 && echo "[Layer 3 Ring] 完成" || echo "[Layer 3 Ring] 异常"
wait $PID_GEN && echo "[泛化测试] 完成" || echo "[泛化测试] 异常"

# ---- 生成 md 报告 ----
echo ""
echo "============================================"
echo "生成汇总报告"
echo "============================================"

python $CODE_ROOT/generate_report.py --results_root $OUT_ROOT

echo ""
echo "============================================"
echo "全部完成: $(date)"
echo "结果目录: $OUT_ROOT"
echo "报告文件: $OUT_ROOT/final_report.md"
echo "============================================"
