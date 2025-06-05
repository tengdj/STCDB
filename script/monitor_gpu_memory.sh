#!/bin/bash


output_file="gpu_memory_usage.csv"


start_time=$(date +%s)


while true
do

    current_time=$(date +%s)
    

    elapsed_seconds=$((current_time - start_time))
    

    memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    

    echo "${elapsed_seconds},${memory_usage}" >> $output_file
    

    sleep 0.1
done