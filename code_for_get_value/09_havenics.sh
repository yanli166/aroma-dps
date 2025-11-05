#!/bin/bash

# 设置包含.fchk文件的文件夹路径
INPUT_FOLDER="/home/ubuntu/cal/DPSCAL/lunci3/NICS1/"
TXT_FOLDER="/home/ubuntu/cal/DPSCAL/lunci3/NICSTXT/"  # TXT文件所在的目录
OUTPUT_FOLDER="/home/ubuntu/cal/DPSCAL/lunci3/NICSOUT1ZZ"  # 输出文件夹路径

# 确保输出文件夹存在
mkdir -p "$OUTPUT_FOLDER"

# 遍历文件夹中的所有.fchk文件
for fchk_file in "$INPUT_FOLDER"/*.fchk; do
    # 提取fchk文件名（不含扩展名）
    fchk_filename=$(basename "$fchk_file" .fchk)
    
    # 提取对应的TXT文件名
    # 假设TXT文件的命名规则是：去掉fchk文件名中的"B"前缀，并使用"nics-"作为前缀
    txt_filename="nics-${fchk_filename#B}.txt"
    txt_file="$TXT_FOLDER/$txt_filename"
    
    # 检查对应的TXT文件是否存在
    if [ ! -f "$txt_file" ]; then
        echo "Warning: No TXT file found for $fchk_file."
        continue
    fi
    
    # 构建输出文件的完整路径
    output_txt="$OUTPUT_FOLDER/${txt_filename%.*}-out.txt"
    
    # 执行Multiwfn操作并将输出重定向到output_txt文件
    multiwfn "$fchk_file" < "$txt_file" > "$output_txt"
    echo "Processed $fchk_file"
done