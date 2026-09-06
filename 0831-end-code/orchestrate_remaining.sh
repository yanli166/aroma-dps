#!/bin/bash
# ===========================================================================
# 剩余实验编排脚本 (父脚本已终止后接管)
#
# 当前状态:
#   - Stage 1ml (PID 2319412) 正在运行 (CPU-only)
#   - Stage 1gnn GPU0 (PID 2291512) 正在运行
#   - Stage 1gnn GPU1 (PID 2291513) 正在运行
#   - Stage 1gnn GPU2 (PID 2291514) 正在运行
#   - Stage 4 已终止, 需要重新拆分启动
#
# 执行流程:
#   1. 启动 4 个 Stage 4 进程 (1 backbone/GPU, 与 Stage 1gnn 共享 GPU 0-2)
#   2. 等待 Stage 1gnn 完成 → 合并结果 → 启动 Stage 2 (3 GPU 按任务)
#   3. 等待 Stage 2 完成 → 合并结果 → 启动 Stage 3 (3 GPU 按任务)
#   4. 等待 Stage 3 + Stage 4 (4 进程) + Stage 1ml 全部完成
#   5. 合并所有结果 → 聚合 → 显著性检验
# ===========================================================================
set -u

cd ${SCRIPT_DIR}/last_end_code
export PYTHONPATH="${SCRIPT_DIR}/last_end_code:${PYTHONPATH:-}"
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null
conda activate torch_env 2>/dev/null

PYTHON=python
RESULTS=${SCRIPT_DIR}/last_end_code/results
LOGS=${SCRIPT_DIR}/last_end_code/logs
mkdir -p "${LOGS}"

# 已有进程 PID
PID_S1ML=2319412
PID_S1GNN_0=2291512
PID_S1GNN_1=2291513
PID_S1GNN_2=2291514

echo "============================================================"
echo "剩余实验编排启动"
echo "  已有进程: Stage1ml=${PID_S1ML}, Stage1gnn=${PID_S1GNN_0},${PID_S1GNN_1},${PID_S1GNN_2}"
echo "============================================================"

# 合并函数
merge_per_seed() {
    local stage_dir="$1"
    shift
    local main_csv="${RESULTS}/${stage_dir}/per_seed_results.csv"
    local tmp="${RESULTS}/${stage_dir}/_merge_tmp.csv"
    : > "${tmp}"
    local first=1
    for sub in "$@"; do
        local csv="${RESULTS}/${sub}/per_seed_results.csv"
        if [ -f "$csv" ]; then
            if [ $first -eq 1 ]; then
                cat "$csv" > "${tmp}"
                first=0
            else
                tail -n +2 "$csv" >> "${tmp}"
            fi
            echo "  [merge] 合并 ${sub}/per_seed_results.csv ($(wc -l < "$csv") 行)"
        fi
    done
    if [ $first -eq 0 ]; then
        mv "${tmp}" "${main_csv}"
        echo "  [merge] -> ${main_csv} ($(wc -l < "${main_csv}") 行)"
    else
        rm -f "${tmp}"
        echo "  [merge] 警告: 无可合并的 CSV (${stage_dir})"
    fi
}

# ============================================================
# Step 1: 启动 4 个 Stage 4 进程 (1 backbone/GPU)
# ============================================================
echo -e "\n>>> [Step 1] 启动 Stage 4 (4 backbone × 4 GPU 并行)"

declare -A S4_PIDS
S4_SUBS=("stage4_mpnn" "stage4_gin" "stage4_gat" "stage4_dmpnn")
S4_GPUS=(0 1 2 3)
S4_BACKBONES=("MPNN" "GIN" "GAT" "DMPNN")

for i in 0 1 2 3; do
    sub=${S4_SUBS[$i]}
    gpu=${S4_GPUS[$i]}
    bb=${S4_BACKBONES[$i]}
    echo "  [启动] Stage 4 ${bb} -> GPU${gpu} (log: logs/stage4_${bb,,}.log)"
    ${PYTHON} -u multiseed/run_multiseed.py --stage 4 \
        --backbones ${bb} --gpu ${gpu} \
        --output_root "${RESULTS}/${sub}" \
        > "${LOGS}/stage4_${bb,,}.log" 2>&1 &
    S4_PIDS[${bb}]=$!
done
echo "  Stage 4 PIDs: MPNN=${S4_PIDS[MPNN]} GIN=${S4_PIDS[GIN]} GAT=${S4_PIDS[GAT]} DMPNN=${S4_PIDS[DMPNN]}"

# ============================================================
# Step 2: 等待 Stage 1gnn 完成 → 合并 → 启动 Stage 2
# ============================================================
echo -e "\n>>> [Step 2] 等待 Stage 1gnn 完成..."
wait ${PID_S1GNN_0} 2>/dev/null; echo "  [完成] Stage 1gnn GPU0 (PID ${PID_S1GNN_0})"
wait ${PID_S1GNN_1} 2>/dev/null; echo "  [完成] Stage 1gnn GPU1 (PID ${PID_S1GNN_1})"
wait ${PID_S1GNN_2} 2>/dev/null; echo "  [完成] Stage 1gnn GPU2 (PID ${PID_S1GNN_2})"

echo "  [合并] Stage 1gnn 结果..."
merge_per_seed "stage1_gnn" "stage1_gnn_g0" "stage1_gnn_g1" "stage1_gnn_g2"

# 启动 Stage 2 (3 GPU 按任务)
echo -e "\n>>> [Step 2b] 启动 Stage 2 (3 GPU 按任务)"
TASKS=("HOMA" "NICS_1zz" "MBCO")
GPUS=(0 1 2)
declare -A S2_PIDS
S2_SUBS=("stage2_g0" "stage2_g1" "stage2_g2")
for i in 0 1 2; do
    g=${GPUS[$i]}
    t=${TASKS[$i]}
    sub=${S2_SUBS[$i]}
    echo "  [启动] Stage 2 ${t} -> GPU${g} (log: logs/stage2_${t,,}.log)"
    ${PYTHON} -u multiseed/run_multiseed.py --stage 2 \
        --tasks ${t} --gpu ${g} \
        --output_root "${RESULTS}/${sub}" \
        > "${LOGS}/stage2_${t,,}.log" 2>&1 &
    S2_PIDS[${t}]=$!
done

# ============================================================
# Step 3: 等待 Stage 2 完成 → 合并 → 启动 Stage 3
# ============================================================
echo -e "\n>>> [Step 3] 等待 Stage 2 完成..."
for t in "${TASKS[@]}"; do
    wait ${S2_PIDS[$t]} 2>/dev/null
    echo "  [完成] Stage 2 ${t} (PID ${S2_PIDS[$t]})"
done

echo "  [合并] Stage 2 结果..."
merge_per_seed "stage2_ring_conditioning" "${S2_SUBS[@]}"

# 启动 Stage 3 (3 GPU 按任务)
echo -e "\n>>> [Step 3b] 启动 Stage 3 (3 GPU 按任务)"
declare -A S3_PIDS
S3_SUBS=("stage3_g0" "stage3_g1" "stage3_g2")
for i in 0 1 2; do
    g=${GPUS[$i]}
    t=${TASKS[$i]}
    sub=${S3_SUBS[$i]}
    echo "  [启动] Stage 3 ${t} -> GPU${g} (log: logs/stage3_${t,,}.log)"
    ${PYTHON} -u multiseed/run_multiseed.py --stage 3 \
        --tasks ${t} --gpu ${g} \
        --output_root "${RESULTS}/${sub}" \
        > "${LOGS}/stage3_${t,,}.log" 2>&1 &
    S3_PIDS[${t}]=$!
done

# ============================================================
# Step 4: 等待所有剩余进程完成
# ============================================================
echo -e "\n>>> [Step 4] 等待 Stage 3 + Stage 4 + Stage 1ml 全部完成..."

# Stage 3
for t in "${TASKS[@]}"; do
    wait ${S3_PIDS[$t]} 2>/dev/null
    echo "  [完成] Stage 3 ${t} (PID ${S3_PIDS[$t]})"
done
echo "  [合并] Stage 3 结果..."
merge_per_seed "stage3_mask_pretraining" "${S3_SUBS[@]}"

# Stage 4 (4 个 backbone)
for bb in "${S4_BACKBONES[@]}"; do
    wait ${S4_PIDS[$bb]} 2>/dev/null
    echo "  [完成] Stage 4 ${bb} (PID ${S4_PIDS[$bb]})"
done
echo "  [合并] Stage 4 结果..."
merge_per_seed "stage4_cross_arch" "${S4_SUBS[@]}"

# Stage 1ml
echo "  [等待] Stage 1ml (PID ${PID_S1ML})..."
wait ${PID_S1ML} 2>/dev/null
echo "  [完成] Stage 1ml (PID ${PID_S1ML})"

# ============================================================
# Step 5: 全局聚合 + 显著性检验
# ============================================================
echo -e "\n>>> [Step 5] 全局聚合 + 显著性检验"
${PYTHON} multiseed/run_multiseed.py --aggregate_only
${PYTHON} stats/significance_test.py

echo -e "\n============================================================"
echo "全部实验完成!"
echo "  结果目录: ${RESULTS}"
echo "  日志目录: ${LOGS}"
echo "============================================================"
