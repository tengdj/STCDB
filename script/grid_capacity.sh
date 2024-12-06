#!/bin/bash

# 遍历 grid_capacity 从 10 到 100，每次增加 10
for i in $(seq 10 10 100); do
    # 构建输出文件名，使用 i 标志不同
    output_file="grid_capacity_out_${i}.txt"
    error_file="grid_capacity_err_${i}.txt"

    # 执行命令，并将标准输出和错误输出重定向到文件
    ../build/pipeline -d 1000 -o 10000000 -g -p -u -m 5 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity "$i" \
        1>"$output_file" 2>"$error_file"
done

echo "所有任务已完成。"
