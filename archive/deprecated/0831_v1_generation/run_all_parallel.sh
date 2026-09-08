#!/bin/bash
# ===========================================================================
# 多种子 × 5折CV 全流程运行脚本 (4-GPU 并行版)
#
# GPU 分配策略:
#   Phase 1: Stage 1ml (CPU/GPU3) + Stage 1gnn (GPU0,1,2 按模型分组) + Stage 4 (GPU3)
#   Phase 2: Stage 2 (GPU0,1,2 按任务分组) | Stage 4 继续 (GPU3)
#   Phase 3: Stage 3 (GPU0,1,2 按任务分组) | Stage 4 继续 (GPU3)
#   Phase 4: 合并 + 聚合 + 显著性检验
#
# 依赖关系:
#   Stage 1ml  — 独立
#   Stage 1gnn — 独立
#   Stage 2    — 依赖 Stage 1gnn (选最佳 backbone)
#   Stage 3    — 依赖 Stage 2    (选最佳 config)
#   Stage 4    — 独立 (固定 base/ring_conditioned 配置)
#
# 用法:
#   bash run_all_parallel.sh
#   GPU_OFFSET=1 bash run_all_parallel.sh   # 从 GPU 1 开始分配
# ===========================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate torch_env 2>/dev/null || echo "[提示] 未激活 torch_env"
fi

PYTHON="${PYTHON:-python}"
GPU_OFFSET="${GPU_OFFSET:-0}"
# 4 张 GPU 编号
G0=$((GPU_OFFSET + 0))
G1=$((GPU_OFFSET + 1))
G2=$((GPU_OFFSET + 2))
G3=$((GPU_OFFSET + 3))

RESULTS="${SCRIPT_DIR}/results"
LOGS="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS}"

echo "============================================================"
echo "4-GPU 并行全流程启动"
echo "  GPU 分配: Stage1gnn/Stage2/Stage3 -> GPU ${G0},${G1},${G2} | Stage4/Stage1ml -> GPU ${G3}"
echo "  PYTHON=${PYTHON} | PYTHONPATH=${PYTHONPATH}"
echo "  工作目录=${SCRIPT_DIR}"
echo "============================================================"

# 并行进程注册
PIDS=""
wait_all() {
    for pid in $1; do
        wait "$pid" 2>/dev/null || true
    done
}

# 合并多个 GPU 子目录的 per_seed_results.csv 到主目录
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
            echo "  [merge] 合并 ${sub}/per_seed_results.csv"
        fi
    done
    if [ $first -eq 0 ]; then
        mv "${tmp}" "${main_csv}"
        echo "  [merge] -> ${main_csv}"
    else
        rm -f "${tmp}"
        echo "  [merge] 无可合并的 CSV (${stage_dir})"
    fi
}

# ============================================================
# Phase 1: Stage 1ml (CPU/后台) + Stage 1gnn (3 GPU 并行) + Stage 4 (GPU3)
# ============================================================
echo -e "\n>>> [Phase 1] Stage 1ml + Stage 1gnn (3 GPU) + Stage 4 (GPU${G3})"

# --- Stage 1ml: 传统ML (GPU3 仅 CatBoost 用, 主要 CPU 计算) ---
echo "  [启动] Stage 1ml -> GPU${G3} (log: logs/stage1ml.log)"
${PYTHON} multiseed/run_multiseed.py --stage 1ml --gpu ${G3} \
    > "${LOGS}/stage1ml.log" 2>&1 &
PID_S1ML=$!

# --- Stage 1gnn: 6 模型分 3 组, 每组 1 GPU ---
# GPU0: GNN, GIN | GPU1: GAT, MPNN | GPU2: GraphSAGE, DMPNN
S1GNN_SUBS=("stage1_gnn_g0" "stage1_gnn_g1" "stage1_gnn_g2")
echo "  [启动] Stage 1gnn GNN,GIN     -> GPU${G0} (log: logs/stage1gnn_g0.log)"
${PYTHON} multiseed/run_multiseed.py --stage 1gnn --models GNN,GIN --gpu ${G0} \
    --output_root "${RESULTS}/${S1GNN_SUBS[0]}" \
    > "${LOGS}/stage1gnn_g0.log" 2>&1 &
PID_S1GNN_0=$!

echo "  [启动] Stage 1gnn GAT,MPNN    -> GPU${G1} (log: logs/stage1gnn_g1.log)"
${PYTHON} multiseed/run_multiseed.py --stage 1gnn --models GAT,MPNN --gpu ${G1} \
    --output_root "${RESULTS}/${S1GNN_SUBS[1]}" \
    > "${LOGS}/stage1gnn_g1.log" 2>&1 &
PID_S1GNN_1=$!

echo "  [启动] Stage 1gnn GraphSAGE,DMPNN -> GPU${G2} (log: logs/stage1gnn_g2.log)"
${PYTHON} multiseed/run_multiseed.py --stage 1gnn --models GraphSAGE,DMPNN --gpu ${G2} \
    --output_root "${RESULTS}/${S1GNN_SUBS[2]}" \
    > "${LOGS}/stage1gnn_g2.log" 2>&1 &
PID_S1GNN_2=$!

# --- Stage 4: 独立, GPU3 上运行 (与 Stage 1ml 共享, ML 占用小) ---
# Stage 4 有 4 backbones × 2 configs × 3 tasks × 5 seeds = 120 runs, 最耗时, 尽早启动
S4_SUBS=("stage4_cross_arch_g3")
echo "  [启动] Stage 4 全部 backbones -> GPU${G3} (log: logs/stage4_g3.log)"
${PYTHON} multiseed/run_multiseed.py --stage 4 --backbones MPNN,GIN,GAT,DMPNN --gpu ${G3} \
    --output_root "${RESULTS}/${S4_SUBS[0]}" \
    > "${LOGS}/stage4_g3.log" 2>&1 &
PID_S4=$!

# 等待 Stage 1gnn 全部完成
echo -e "\n  [等待] Stage 1gnn 3 个 GPU 进程完成..."
wait_all "${PID_S1GNN_0} ${PID_S1GNN_1} ${PID_S1GNN_2}"
echo "  [完成] Stage 1gnn 全部 GPU 进程结束"

# 合并 Stage 1gnn 结果到主目录
echo "  [合并] Stage 1gnn 结果..."
merge_per_seed "stage1_gnn" "${S1GNN_SUBS[@]}"

# 合并后的聚合
echo "  [聚合] Stage 1gnn..."
${PYTHON} multiseed/run_multiseed.py --aggregate_only 2>&1 | grep -A5 "stage1_gnn" || true

# ============================================================
# Phase 2: Stage 2 (3 GPU 按任务分组) | Stage 4 继续 (GPU3)
# ============================================================
echo -e "\n>>> [Phase 2] Stage 2 (3 GPU 按任务) | Stage 4 继续 (GPU${G3})"

S2_SUBS=("stage2_ring_conditioning_g0" "stage2_ring_conditioning_g1" "stage2_ring_conditioning_g2")
TASKS=("HOMA" "NICS_1zz" "MBCO")
GPUS=($G0 $G1 $G2)
for i in 0 1 2; do
    g=${GPUS[$i]}
    echo "  [启动] Stage 2 ${TASKS[$i]} -> GPU${g} (log: logs/stage2_g${i}.log)"
    ${PYTHON} multiseed/run_multiseed.py --stage 2 --tasks ${TASKS[$i]} --gpu ${g} \
        --output_root "${RESULTS}/${S2_SUBS[$i]}" \
        > "${LOGS}/stage2_g${i}.log" 2>&1 &
    eval "PID_S2_${i}=$!"
done

echo -e "\n  [等待] Stage 2 3 个 GPU 进程完成..."
wait_all "$(for i in 0 1 2; do eval echo \$PID_S2_${i}; done)"
echo "  [完成] Stage 2 全部 GPU 进程结束"

# 合并 Stage 2 结果
echo "  [合并] Stage 2 结果..."
merge_per_seed "stage2_ring_conditioning" "${S2_SUBS[@]}"

# ============================================================
# Phase 3: Stage 3 (3 GPU 按任务分组) | Stage 4 继续 (GPU3)
# ============================================================
echo -e "\n>>> [Phase 3] Stage 3 (3 GPU 按任务) | Stage 4 继续 (GPU${G3})"

S3_SUBS=("stage3_mask_pretraining_g0" "stage3_mask_pretraining_g1" "stage3_mask_pretraining_g2")
for i in 0 1 2; do
    g=${GPUS[$i]}
    echo "  [启动] Stage 3 ${TASKS[$i]} -> GPU${g} (log: logs/stage3_g${i}.log)"
    ${PYTHON} multiseed/run_multiseed.py --stage 3 --tasks ${TASKS[$i]} --gpu ${g} \
        --output_root "${RESULTS}/${S3_SUBS[$i]}" \
        > "${LOGS}/stage3_g${i}.log" 2>&1 &
    eval "PID_S3_${i}=$!"
done

echo -e "\n  [等待] Stage 3 3 个 GPU 进程完成..."
wait_all "$(for i in 0 1 2; do eval echo \$PID_S3_${i}; done)"
echo "  [完成] Stage 3 全部 GPU 进程结束"

# 合并 Stage 3 结果
echo "  [合并] Stage 3 结果..."
merge_per_seed "stage3_mask_pretraining" "${S3_SUBS[@]}"

# ============================================================
# 等待 Stage 1ml 和 Stage 4 完成
# ============================================================
echo -e "\n>>> [等待] Stage 1ml 和 Stage 4 完成..."
wait_all "${PID_S1ML}"
echo "  [完成] Stage 1ml"
wait_all "${PID_S4}"
echo "  [完成] Stage 4"

# 合并 Stage 4 结果
echo "  [合并] Stage 4 结果..."
merge_per_seed "stage4_cross_arch" "${S4_SUBS[@]}"

# ============================================================
# Phase 4: 全局聚合 + 显著性检验
# ============================================================
echo -e "\n>>> [Phase 4] 全局聚合 + 显著性检验"
${PYTHON} multiseed/run_multiseed.py --aggregate_only
${PYTHON} stats/significance_test.py

echo -e "\n============================================================"
echo "4-GPU 并行全流程完成"
echo "  结果目录: ${RESULTS}"
echo "  日志目录: ${LOGS}"
echo "============================================================"
