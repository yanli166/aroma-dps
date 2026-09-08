#!/bin/bash
# ===========================================================================
# 0831-end-code Fig.3 正式复现: 5 seeds × 双轨 feature_mode × 6 stages
# bug修复后 (graphs.py 2026-09-02) 重新生成全部结果
#
# 依赖关系:
#   Stage 1 — 独立
#   Stage 4 — 独立
#   Stage 5 — 独立
#   Stage 2 — 依赖 Stage 1 (选最佳 backbone)
#   Stage 3 — 依赖 Stage 2 (选最佳 config)
#   Stage 6 — 依赖 Stage 5 (ring_flag 决定)
#
# 用法: bash run_all_v2_parallel.sh
# ===========================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"
SEEDS=(11 22 33 44 55)
LOG_DIR="${SCRIPT_DIR}/logs_v2_rerun"
mkdir -p "${LOG_DIR}"

# 备份旧结果
if [ -d "${SCRIPT_DIR}/results_v2" ] && [ ! -d "${SCRIPT_DIR}/results_v2_backup_pre_bugfix" ]; then
    echo "[backup] 备份旧 results_v2 到 results_v2_backup_pre_bugfix"
    cp -r "${SCRIPT_DIR}/results_v2" "${SCRIPT_DIR}/results_v2_backup_pre_bugfix"
fi

# 清理旧结果 (仅 seed 子目录, 保留 splits)
for stage in stage1 stage2 stage3 stage4 final_membership ring_flag_sensitivity; do
    dir="${SCRIPT_DIR}/results_v2/${stage}"
    if [ -d "$dir" ]; then
        find "$dir" -name "seed_*" -type d -exec rm -rf {} + 2>/dev/null || true
        find "$dir" -name "*.csv" -exec rm -f {} + 2>/dev/null || true
        find "$dir" -name "*.json" -exec rm -f {} + 2>/dev/null || true
    fi
done
# 清理 stage4/partial
rm -rf "${SCRIPT_DIR}/results_v2/stage4/partial" 2>/dev/null || true

echo "[cleanup] 旧 seed 结果已清理"
echo "[start] 5 seeds × 双轨 × 6 stages 并行启动"
echo ""

# --- Phase 1: 独立 stages (Stage 1, 4, 5) ---
# GPU 0: Stage 1 (5 seeds × 2 modes)
# GPU 1: Stage 4 (5 seeds × 2 modes)
# GPU 2: Stage 5 (5 seeds × 2 modes)
# GPU 3: Stage 1 第二批 (交替使用)

run_stage1() {
    local gpu=$1
    for seed in "${SEEDS[@]}"; do
        for mode in standard explicit_aromaticity_ablated; do
            echo "[Stage1] seed=${seed} mode=${mode} gpu=${gpu}" | tee -a "${LOG_DIR}/stage1.log"
            CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} \
                stage1_representation_comparison/code/run_stage1_v2.py \
                --seed ${seed} --feature_mode ${mode} --gpu 0 \
                2>&1 | tee -a "${LOG_DIR}/stage1_s${seed}_${mode}.log"
        done
    done
}

run_stage4() {
    local gpu=$1
    for seed in "${SEEDS[@]}"; do
        for mode in standard explicit_aromaticity_ablated; do
            echo "[Stage4] seed=${seed} mode=${mode} gpu=${gpu}" | tee -a "${LOG_DIR}/stage4.log"
            CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} \
                stage4_cross_architecture/code/run_stage4_v2.py \
                --seed ${seed} --feature_mode ${mode} --gpu 0 \
                2>&1 | tee -a "${LOG_DIR}/stage4_s${seed}_${mode}.log"
        done
    done
}

run_stage5() {
    local gpu=$1
    for seed in "${SEEDS[@]}"; do
        for mode in standard explicit_aromaticity_ablated; do
            echo "[Stage5] seed=${seed} mode=${mode} gpu=${gpu}" | tee -a "${LOG_DIR}/stage5.log"
            CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} \
                stage5_ring_flag_sensitivity/code/run_ring_flag_sensitivity.py \
                --seed ${seed} --feature_mode ${mode} --gpu 0 \
                2>&1 | tee -a "${LOG_DIR}/stage5_s${seed}_${mode}.log"
        done
    done
}

# 启动 Phase 1 (3 GPU 并行)
run_stage1 0 &
PID_S1=$!
run_stage4 1 &
PID_S4=$!
run_stage5 2 &
PID_S5=$!

echo "[Phase1] Stage1(PID=$PID_S1) GPU0 | Stage4(PID=$PID_S4) GPU1 | Stage5(PID=$PID_S5) GPU2"

# 等待 Stage 1 完成
wait $PID_S1
echo "[Phase1] Stage 1 完成, 启动 Stage 2"

# --- Phase 2: Stage 2 (依赖 Stage 1) ---
run_stage2() {
    local gpu=$1
    for seed in "${SEEDS[@]}"; do
        for mode in standard explicit_aromaticity_ablated; do
            echo "[Stage2] seed=${seed} mode=${mode} gpu=${gpu}" | tee -a "${LOG_DIR}/stage2.log"
            CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} \
                stage2_ring_conditioning/code/ring_conditioning_ablation.py \
                --seed ${seed} --feature_mode ${mode} --gpu 0 \
                2>&1 | tee -a "${LOG_DIR}/stage2_s${seed}_${mode}.log"
        done
    done
}

# Stage 2 在 Stage 1 完成后启动 (GPU 0 空闲)
run_stage2 0 &
PID_S2=$!
echo "[Phase2] Stage2(PID=$PID_S2) GPU0"

# 等待 Stage 2 完成
wait $PID_S2
echo "[Phase2] Stage 2 完成, 启动 Stage 3"

# --- Phase 3: Stage 3 (依赖 Stage 2, 单轨) ---
# run_mask_pretrain_v2.py = publication protocol (get_final_splits, SPLIT_SEED=2026).
# The legacy run_pretrain_eval.py used canonical_splits(seed=model_seed) and is
# NON-publication; it stays available under archive/deprecated/.
for seed in "${SEEDS[@]}"; do
    echo "[Stage3] seed=${seed} gpu=0" | tee -a "${LOG_DIR}/stage3.log"
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} \
        stage3_mask_pretraining/code/run_mask_pretrain_v2.py \
        --seed ${seed} --gpu 0 \
        2>&1 | tee -a "${LOG_DIR}/stage3_s${seed}.log"
done
echo "[Phase3] Stage 3 完成"

# 等待 Stage 5 完成
wait $PID_S5
echo "[Phase1] Stage 5 完成, 启动 Stage 6"

# --- Phase 4: Stage 6 (依赖 Stage 5) ---
run_stage6() {
    local gpu=$1
    for seed in "${SEEDS[@]}"; do
        for mode in standard explicit_aromaticity_ablated; do
            echo "[Stage6] seed=${seed} mode=${mode} gpu=${gpu}" | tee -a "${LOG_DIR}/stage6.log"
            CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} \
                stage6_final_membership/code/run_final_membership.py \
                --seed ${seed} --feature_mode ${mode} --gpu 0 \
                2>&1 | tee -a "${LOG_DIR}/stage6_s${seed}_${mode}.log"
        done
    done
}

# Stage 6 在 Stage 5 完成后启动
run_stage6 0 &
PID_S6=$!
echo "[Phase4] Stage6(PID=$PID_S6) GPU0"

# 等待 Stage 4 完成
wait $PID_S4
echo "[Phase1] Stage 4 完成"

# 等待 Stage 6 完成
wait $PID_S6
echo "[Phase4] Stage 6 完成"

echo ""
echo "============================================================"
echo "全部 6 stages × 5 seeds × 2 feature_modes 完成!"
echo "结果目录: ${SCRIPT_DIR}/results_v2/"
echo "日志目录: ${LOG_DIR}/"
echo "============================================================"
