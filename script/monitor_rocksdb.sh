#!/bin/bash

# 启动 iostat
iostat -dx 1 > /home/xiang/1vs/iostat.log &
iostat_pid=$!

# 启动你的程序
../build/rocksdb &
program_pid=$!

# 等待程序结束
wait $program_pid

# 停止 iostat
kill $iostat_pid