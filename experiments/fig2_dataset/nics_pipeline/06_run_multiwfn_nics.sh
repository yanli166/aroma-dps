#!/bin/bash
# Step 6: 运行Multiwfn NICS_ZZ计算 (function 25 option 4) — config-driven
# 用法: bash 06_run_multiwfn_nics.sh <dataset> [并行数]
#       bash 06_run_multiwfn_nics.sh lunci9 8

set -e

if [ "$#" -lt 1 ]; then
    echo "用法: bash 06_run_multiwfn_nics.sh <dataset> [并行数]"
    exit 1
fi
DATASET="${1:?用法: bash 06_run_multiwfn_nics.sh <dataset> [并行数]}"
MAX_JOBS="${2:-8}"
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "$PIPELINE_DIR/config.py" --shell "$DATASET")"

# Multiwfn路径
MULTIWFN="/home/ubuntu/data_90/alldata_in_3090/Multiwfn/multiwfn"
export Multiwfnpath="/home/ubuntu/data_90/alldata_in_3090/Multiwfn"

INPUT_DIR="$MULTIWFN_INPUT_DIR"
OUTPUT_DIR="$MULTIWFN_OUTPUT_DIR"
FCHK_SEARCH_DIRS="$FCHK_SEARCH_DIRS"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Step 6: 运行Multiwfn NICS_ZZ ($DATASET)"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  fchk搜索目录: $FCHK_SEARCH_DIRS"
echo "  并行数: $MAX_JOBS"
echo "============================================================"

# 查找fchk文件的函数 (支持多目录搜索)
find_fchk() {
    local mol_id="$1"
    for dir in $FCHK_SEARCH_DIRS; do
        local fchk="$dir/${mol_id}.fchk"
        if [ -f "$fchk" ]; then
            echo "$fchk"
            return 0
        fi
    done
    return 1
}

# 并行运行
running=0
total=0
failed=0

for input_file in "$INPUT_DIR"/*-nics-input.txt; do
    [ -f "$input_file" ] || continue
    mol_id=$(basename "$input_file" -nics-input.txt)
    total=$((total + 1))

    fchk_file=$(find_fchk "$mol_id")
    if [ -z "$fchk_file" ]; then
        echo "警告: 未找到 ${mol_id}.fchk, 跳过"
        failed=$((failed + 1))
        continue
    fi

    output_file="$OUTPUT_DIR/${mol_id}-nics-output.txt"

    # 并行控制
    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi

    (
        "$MULTIWFN" "$fchk_file" < "$input_file" > "$output_file" 2>/dev/null
    ) &
    running=$((running + 1))
done

wait

# 统计结果
success=$(ls "$OUTPUT_DIR"/*-nics-output.txt 2>/dev/null | wc -l)
echo ""
echo "完成: $success/$total 成功, $failed 跳过"
echo "输出: $OUTPUT_DIR"
echo ""
echo "下一步: python3 07_parse_nics.py $DATASET"
