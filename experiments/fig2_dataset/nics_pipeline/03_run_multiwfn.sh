#!/bin/bash
# Step 3: 批量运行Multiwfn计算HOMA或MBCO — config-driven
# 用法: bash 03_run_multiwfn.sh <dataset> homa|mbco
#       bash 03_run_multiwfn.sh lunci9 homa

if [ "$#" -lt 2 ] || { [ "$2" != "homa" ] && [ "$2" != "mbco" ]; }; then
    echo "用法: bash 03_run_multiwfn.sh <dataset> homa|mbco"
    exit 1
fi
DATASET=$1
TYPE=$2  # homa 或 mbco
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "$PIPELINE_DIR/config.py" --shell "$DATASET")"
TXT_DIR=$([ "$TYPE" = "homa" ] && echo "$HOMA_TXT_DIR" || echo "$MBCO_TXT_DIR")
OUT_DIR=$([ "$TYPE" = "homa" ] && echo "$OUTHOMA_DIR" || echo "$OUTMBCO_DIR")
mkdir -p "$OUT_DIR"

export Multiwfnpath=/home/ubuntu/data_90/alldata_in_3090/Multiwfn
export PATH=$PATH:$Multiwfnpath

count=0
total=$(ls "$TXT_DIR"/*.txt 2>/dev/null | wc -l)
echo "共 $total 个${TYPE^^}输入文件需要处理 ($DATASET)"

for fchk_file in "$FCHK_DIR"/*.fchk; do
    [ -f "$fchk_file" ] || continue
    [ -s "$fchk_file" ] || continue  # 跳过空文件
    filename=$(basename "$fchk_file" .fchk)

    for ring_id in $(seq 1 20); do
        input_filename="${TYPE}-${filename}-ring${ring_id}.txt"
        input_txt="$TXT_DIR/$input_filename"
        [ -f "$input_txt" ] || continue

        output_txt="$OUT_DIR/$input_filename-out.txt"
        if [ -f "$output_txt" ] && [ -s "$output_txt" ]; then
            count=$((count + 1))
            continue
        fi
        multiwfn "$fchk_file" < "$input_txt" > "$output_txt" 2>/dev/null
        count=$((count + 1))
        if [ $((count % 50)) -eq 0 ]; then
            echo "进度: $count / $total"
        fi
    done
done

echo "${TYPE^^}计算完成: $count / $total ($DATASET)"
