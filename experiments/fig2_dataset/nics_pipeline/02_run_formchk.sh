#!/bin/bash
# Step 2: 批量formchk (chk → fchk) — config-driven
# 用法: bash 02_run_formchk.sh <dataset>
#       bash 02_run_formchk.sh lunci9
# 注意: 不使用 set -e, 因 g16.profile 含非交互式下返回非零的命令

if [ "$#" -lt 1 ]; then
    echo "用法: bash 02_run_formchk.sh <dataset>"
    exit 1
fi
DATASET=$1
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

# 从config.py加载路径
eval "$(python3 "$PIPELINE_DIR/config.py" --shell "$DATASET")"
mkdir -p "$FCHK_DIR"

export g16root=/home/ubuntu/apps
source /home/ubuntu/apps/g16/bsd/g16.profile

echo "=== Step 2: formchk ($DATASET): $INPUT_DIR → $FCHK_DIR ==="
count=0
total=0
for chk_file in "$INPUT_DIR"/*.chk; do
    [ -f "$chk_file" ] || continue
    total=$((total + 1))
    basename=$(basename "$chk_file" .chk)
    fchk_file="$FCHK_DIR/${basename}.fchk"
    if [ -f "$fchk_file" ] && [ -s "$fchk_file" ]; then
        count=$((count + 1))
        continue  # 已存在且非空,跳过
    fi
    formchk "$chk_file" "$fchk_file" 2>/dev/null
    count=$((count + 1))
    if [ $((count % 20)) -eq 0 ]; then
        echo "进度: $count / $total"
    fi
done

valid=$(find "$FCHK_DIR" -maxdepth 1 -name "*.fchk" -size +0c | wc -l)
echo "$DATASET done: $valid valid fchk files (processed $count / $total)"
echo "Step 2 完成。下一步: bash 03_run_multiwfn.sh $DATASET homa && bash 03_run_multiwfn.sh $DATASET mbco"
