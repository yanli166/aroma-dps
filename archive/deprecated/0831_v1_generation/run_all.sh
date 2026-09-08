#!/bin/bash
# ===========================================================================
# 多种子 × 5折CV 全流程运行脚本 (四阶段)
#
# 顺序执行:
#   Stage 1: 传统ML + GNN基线
#   Stage 2: 环条件化消融 (依赖 Stage1 GNN 结果选 backbone)
#   Stage 3: Ring Masking 预训练 (依赖 Stage2 结果选最佳配置)
#   Stage 4: 跨架构验证
#   聚合所有阶段 + 显著性检验
#
# 用法:
#   bash run_all.sh
#   bash run_all.sh --gpu 1          # 透传参数到各阶段 (注意: 仅 --gpu 会被透传)
# ===========================================================================

set -u  # 引用未定义变量报错

# 切换到脚本所在目录 (last_end_code 根目录), 保证相对路径可用
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 将 last_end_code 根目录加入 PYTHONPATH, 使 common / models / stageX_* 可导入
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 激活 conda 环境 (torch_env); 若不存在则回退到当前 python
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate torch_env 2>/dev/null || echo "[提示] 未激活 torch_env, 使用默认 python"
fi

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"

echo "============================================================"
echo "全流程启动 | PYTHONPATH=${PYTHONPATH}"
echo "             PYTHON=${PYTHON} | GPU=${GPU}"
echo "             工作目录=${SCRIPT_DIR}"
echo "============================================================"

# ---------- Stage 1: 传统ML + GNN基线 ----------
echo -e "\n>>> [Stage 1ml] 传统ML基线 (多种子)"
${PYTHON} multiseed/run_multiseed.py --stage 1ml --gpu "${GPU}"

echo -e "\n>>> [Stage 1gnn] GNN基线 (多种子)"
${PYTHON} multiseed/run_multiseed.py --stage 1gnn --gpu "${GPU}"

# ---------- Stage 2: 环条件化消融 (依赖 Stage1 GNN 结果选 backbone) ----------
echo -e "\n>>> [Stage 2] 环条件化消融 (自动选 backbone)"
${PYTHON} multiseed/run_multiseed.py --stage 2 --gpu "${GPU}"

# ---------- Stage 3: Ring Masking 预训练 (依赖 Stage2 结果选最佳配置) ----------
echo -e "\n>>> [Stage 3] Ring Masking 预训练"
${PYTHON} multiseed/run_multiseed.py --stage 3 --gpu "${GPU}"

# ---------- Stage 4: 跨架构验证 ----------
echo -e "\n>>> [Stage 4] 跨架构验证"
${PYTHON} multiseed/run_multiseed.py --stage 4 --gpu "${GPU}"

# ---------- 聚合所有阶段 ----------
echo -e "\n>>> [聚合] 汇总各阶段多种子结果 (mean ± std)"
${PYTHON} multiseed/run_multiseed.py --aggregate_only

# ---------- 显著性检验 ----------
echo -e "\n>>> [统计] 显著性检验 (paired t-test + Wilcoxon)"
${PYTHON} stats/significance_test.py

echo -e "\n============================================================"
echo "全流程完成"
echo "============================================================"
