#!/bin/bash

# 定义 Gaussian 执行命令
G16_CMD="/home/ubuntu/apps/g16/g16"

# 获取当前目录下所有 .gjf 文件并排序
GJF_FILES=($(ls *.gjf | sort))

# 提交 Gaussian 计算任务
for GJF_FILE in "${GJF_FILES[@]}"
do
  # 提取文件名（不包含扩展名）
  BASENAME=$(basename "$GJF_FILE" .gjf)
  echo "Submitting job for input file $GJF_FILE"
  
  # 执行 Gaussian 计算
  $G16_CMD < "$GJF_FILE" > "${BASENAME}.log"
  
  # 检查 Gaussian 是否正常完成
  if grep -q "Normal termination" "${BASENAME}.log"; then
    echo "Job for $GJF_FILE completed successfully."
  else
    echo "Job for $GJF_FILE did not terminate normally. Check the log file."
  fi
done