#!/bin/bash

for i in {1..10}; do
    echo "Run #$i"


    DURATION=$((400 + 250*(i-1)))
    MEMTABLE_CAPACITY=$((2 * i))


    ../build/pipeline --map_path ../data/streets.map \
                      --meta_path ../data/chicago.mt \
                      -d $DURATION \
                      -o 10000000 \
                      -m 5 \
                      -g 1 > stdout_$i.txt 2> stderr_$i.txt \
                      -t ../100_10000000.tr \
                      --sstable_count 50 \
                      --memTable_capacity $MEMTABLE_CAPACITY


    if [ $? -eq 0 ]; then
        echo "Run #$i: Program executed successfully."
    else
        echo "Run #$i: Program failed to execute. Check stdout_$i.txt and stderr_$i.txt for details."
    fi
done
