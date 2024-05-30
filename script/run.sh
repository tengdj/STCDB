#!/bin/bash

for i in {4..10}; do
    echo "Run #$i"

    # 动态调整 --sstable_count 参数
    DURATION=$((400 + 250*(i-1)))
    MEMTABLE_CAPACITY=$((2 * i))

    # 运行程序并将输出重定向到文件
    ../build/pipeline --map_path ../data/streets.map \
                      --meta_path ../data/chicago.mt \
                      -d $DURATION \
                      -o 10000000 \
                      -m 5 \
                      -g 1 > stdout_$i.txt 2> stderr_$i.txt \
                      -t ../100_10000000.tr \
                      --sstable_count 50 \
                      --memTable_capacity $MEMTABLE_CAPACITY

    # 检查程序是否成功执行
    if [ $? -eq 0 ]; then
        echo "Run #$i: Program executed successfully."
    else
        echo "Run #$i: Program failed to execute. Check stdout_$i.txt and stderr_$i.txt for details."
    fi
done
