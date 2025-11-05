#!/bin/bash

# 设置包含.wfn文件的文件夹路径
INPUT_FOLDER="/home/ubuntu/cal/DPSCAL/lunci3/gjf-1"
OUTPUT_FOLDER="/home/ubuntu/cal/DPSCAL/lunci3/OUT_mppsdp"
# 遍历文件夹中的所有.wfn文件
for wfn_file in "$INPUT_FOLDER"/*.fchk; do
    # 提取文件名（不含扩展名）
    filename=$(basename "$wfn_file" .fchk)
    # 构建对应的homa-*-*.txt文件名
    homa_filename="mpp-$filename.txt"
    input_txt="/home/ubuntu/cal/DPSCAL/lunci3/21_mppsdp.txt"
    if [ ! -f "$input_txt" ]; then
        continue # 如果文件不存在，跳过当前Ring_ID的处理
    fi
    # 构建输出文件的完整路径
    output_txt="$OUTPUT_FOLDER/$homa_filename-out.txt"
    
    # 检查对应的homa-*-*.txt文件是否存在
    if [ -f "$input_txt" ]; then
        # 执行Multiwfn操作并将输出重定向到out.txt文件
        multiwfn "$wfn_file" < "$input_txt" > "$output_txt"
        echo "Processed $wfn_file with ring_id $ring_id"
    else
        echo "Parameter file not found for $wfn_file with ring_id $ring_id"
    fi
    
done
