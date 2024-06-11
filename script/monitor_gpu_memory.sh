#!/bin/bash

# 定义输出文件名
output_file="gpu_memory_usage.csv"

# 获取初始时间戳
start_time=$(date +%s)

# 无限循环，按需可以设置持续时间或条件退出
while true
do
    # 获取当前时间戳
    current_time=$(date +%s)
    
    # 计算从脚本开始到当前的时间差（秒）
    elapsed_seconds=$((current_time - start_time))
    
    # 获取显存使用情况
    memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    
    # 将时间戳和显存使用量追加到文件中
    echo "${elapsed_seconds},${memory_usage}" >> $output_file
    
    # 休眠一段时间，这里是1/10秒，即100毫秒
    sleep 0.1
done