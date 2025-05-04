#!/bin/bash


for i in $(seq 10 10 100); do

    output_file="grid_capacity_out_${i}.txt"
    error_file="grid_capacity_err_${i}.txt"


    ../build/pipeline -d 1000 -o 10000000 -g -p -u -m 5 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity "$i" \
        1>"$output_file" 2>"$error_file"
done

echo "finished"
