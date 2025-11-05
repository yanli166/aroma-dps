#!/bin/bash

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OMP_STACKSIZE=20G
ulimit -s unlimited

# 指定包含XYZ文件的目录
xyz_dir="/home/ubuntu/cal/DPSCAL/lunci3/xtb-1/"
# 指定存放优化结果的根目录
results_root_dir="/home/ubuntu/cal/DPSCAL/lunci3/xtb-out/"

# 确保根目录存在
mkdir -p "$results_root_dir"

# 遍历目录中的所有XYZ文件
for xyz_file in "$xyz_dir"/*.xyz; do
    if [ -f "$xyz_file" ]; then
        # 提取文件名（不包含路径）
        base_name=$(basename "$xyz_file" .xyz)
        
        # 为每个文件创建一个同名的子文件夹
        output_dir="$results_root_dir/$base_name"
        mkdir -p "$output_dir"
        
        # 将XYZ文件移动到子文件夹
        mv "$xyz_file" "$output_dir/"
        
        # 提交XTB作业
        cd "$output_dir"
        xtb "$base_name.xyz" --chrg 0 --uhf 0 --gfn 2 --opt > "${base_name}_xtb.out"
        cd -
    else
        echo "Warning: $xyz_file is not a regular file or does not exist."
    fi
done




