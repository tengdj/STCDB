#!/bin/bash

for i in $(seq 50000000 10000000 50000000); do

    output_file="objects_out_${i}.txt"
    error_file="objects_err_${i}.txt"


    ../build/pipeline -d 1000 -o "$i" -g -p -m 5 -r 1 --memTable_capacity 2 --G_bytes 2 --grid_capacity 50 --zone_capacity 20 \
        1>"$output_file" 2>"$error_file"
done

echo "所有任务已完成。"

