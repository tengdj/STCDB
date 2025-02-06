#!/bin/bash

#kv_restriction = G_bytes * 33554432; //half GB

values=(6)

for i in "${values[@]}"; do

    output_file="six_GB_${i}out.txt"
    error_file="six_GB_${i}err.txt"

    ../build/pipeline -d 2000 -o 20000000 --load_data  -g -p -m 5 -r 2 --memTable_capacity 2 --G_bytes "$i" --grid_capacity 50 --zone_capacity 20 \
        1>"$output_file" 2>"$error_file"
done

echo "所有任务已完成。"