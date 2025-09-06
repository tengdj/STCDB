#!/bin/bash

for i in $(seq 0 1 7); do
    sudo mount /dev/nvme"$i"n1 /data3/ssd"$i"
done

echo "mount finish"

