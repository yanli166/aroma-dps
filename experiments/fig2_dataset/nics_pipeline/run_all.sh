#!/bin/bash
# 端到端主控脚本: 从Gaussian优化结果生成 HOMA/MBCO/NICS 汇总CSV — config-driven
#
# 流程:
#   步骤 1-4: HOMA/MBCO (环信息, formchk, Multiwfn, 解析)
#   步骤 5-7: NICS_ZZ   (生成Multiwfn输入, 运行Multiwfn, 解析) — 需NMR log已存在
#
# 前提条件:
#   - 数据集已在 config.py 中注册
#   - input_dir 含 .chk/.gjf/.log (优化计算完成)
#   - csv_path  SMILES CSV (含 'no','SMILES' 列)
#   - Gaussian: /home/ubuntu/apps/g16/
#   - Multiwfn: /home/ubuntu/data_90/alldata_in_3090/Multiwfn/
#   - Python (pandas, rdkit): /home/ubuntu/apps/anaconda3/bin/python3
#
# NICS计算需手动投递 nics_gjf/*.gjf, NMR log完成后(nics_gjf/*-nics.log)再运行 --nics
#
# 用法:
#   bash run_all.sh <dataset>           # HOMA/MBCO 流程 (步骤1-4)
#   bash run_all.sh <dataset> --nics    # + NICS解析 (步骤5-7, 需 *-nics.log 已存在)
#   bash run_all.sh <dataset> --full    # 步骤1-7 (NICS部分需NMR log已存在)
#
# 示例:
#   bash run_all.sh lunci9
#   bash run_all.sh lunci9 --nics

set -e

if [ "$#" -lt 1 ]; then
    echo "用法: bash run_all.sh <dataset> [--nics|--full]"
    echo "  (无参数)   仅 HOMA/MBCO (步骤1-4)"
    echo "  --nics     仅 NICS解析 (步骤5-7, 需 *-nics.log 已存在)"
    echo "  --full     全流程 (步骤1-7)"
    echo ""
    echo "已注册数据集:"
    python3 "$(cd "$(dirname "$0")" && pwd)/config.py"
    exit 1
fi

DATASET=$1
MODE="${2:-homa_mbco}"
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/ubuntu/apps/anaconda3/bin/python3

# 校验数据集已注册
if ! python3 "$PIPELINE_DIR/config.py" datasets | tr ' ' '\n' | grep -qx "$DATASET"; then
    echo "错误: 未知数据集 '$DATASET'"
    echo "已注册: $(python3 "$PIPELINE_DIR/config.py" datasets)"
    exit 1
fi

run_homa_mbco() {
    echo ""
    echo "============================================================"
    echo "Step 1: 准备输入 (环信息, HOMA/MBCO txt, NICS GJF)"
    echo "============================================================"
    $PYTHON "$PIPELINE_DIR/01_prepare_inputs.py" "$DATASET"

    echo ""
    echo "============================================================"
    echo "Step 2: formchk (chk → fchk)"
    echo "============================================================"
    bash "$PIPELINE_DIR/02_run_formchk.sh" "$DATASET"

    echo ""
    echo "============================================================"
    echo "Step 3a: Multiwfn HOMA"
    echo "============================================================"
    bash "$PIPELINE_DIR/03_run_multiwfn.sh" "$DATASET" homa

    echo ""
    echo "============================================================"
    echo "Step 3b: Multiwfn MBCO"
    echo "============================================================"
    bash "$PIPELINE_DIR/03_run_multiwfn.sh" "$DATASET" mbco

    echo ""
    echo "============================================================"
    echo "Step 4: 解析HOMA/MBCO → ${DATASET}-homa-mbco-summary.csv"
    echo "============================================================"
    $PYTHON "$PIPELINE_DIR/04_parse_homa_mbco.py" "$DATASET"
}

run_nics() {
    echo ""
    echo "============================================================"
    echo "Step 5: 生成Multiwfn NICS_ZZ输入"
    echo "============================================================"
    $PYTHON "$PIPELINE_DIR/05_gen_nics_multiwfn_input.py" "$DATASET"

    echo ""
    echo "============================================================"
    echo "Step 6: 运行Multiwfn NICS_ZZ (function 25 option 4)"
    echo "============================================================"
    bash "$PIPELINE_DIR/06_run_multiwfn_nics.sh" "$DATASET"

    echo ""
    echo "============================================================"
    echo "Step 7: 解析NICS → ${DATASET}-homa-mbco-nics-final.csv"
    echo "============================================================"
    $PYTHON "$PIPELINE_DIR/07_parse_nics.py" "$DATASET"
}

case "$MODE" in
    --nics)
        run_nics
        ;;
    --full)
        run_homa_mbco
        run_nics
        ;;
    ""|homa_mbco|--homa-mbco)
        run_homa_mbco
        ;;
    *)
        echo "未知模式: $MODE (支持: --nics, --full, 或留空)"
        exit 1
        ;;
esac

echo ""
echo "############################################################"
echo "# $DATASET 完成! (模式: $MODE)"
echo "# 输出目录: $(python3 "$PIPELINE_DIR/config.py" "$DATASET" out_dir)"
echo "############################################################"
