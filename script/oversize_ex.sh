#!/bin/bash

#kv_restriction = G_bytes * 33554432; //half GB

# 定义数组
#values=(0.00000575557 0.00000724534 0.0000095889 0.0000133709 0.0000192537 0.0000301749 0.000053588 0.000103177 0.000204741 0.00044563 0.00233805)

# 遍历索引
for i in $(seq 0 1 10); do
    # 获取数组中的值
    #value=${values[$i]}

    output_file="oversize${i}percent_out.txt"
    error_file="oversize${i}percent_err.txt"

#walk_rate refers to area_reistriction
    ../build/pipeline -d 2000 -o 20000000 --load_data  -g -p -f "$i" --CTF_count 256 \
    -m 5 -r 2 --memTable_capacity 2 --G_bytes 6 --grid_capacity 50 --zone_capacity 20 \
        1>"$output_file" 2>"$error_file"
done

echo "所有任务已完成。"


