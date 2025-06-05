#!/bin/bash


iostat -dx 1 > /home/xiang/1vs/iostat.log &
iostat_pid=$!


../build/rocksdb &
program_pid=$!


wait $program_pid


kill $iostat_pid