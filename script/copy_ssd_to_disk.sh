##!/bin/bash
#
##for i in $(seq 0 1 7); do
##    sudo cp -r -v /data3/ssd"$i"/* /data/copy_of_ssd
##done
##
##echo "copy finished"
#
#
#
## 设置目标目录（确保目录存在）
#RAID0_DIR="/data3/raid0_num0"
#RAID1_DIR="/data3/raid0_num1"
#
## 遍历当前目录下所有以 SSTable_ 开头的文件
#for file in /data/copy_of_ssd/SSTable_*; do
#    # 检查是否为文件
#    if [ -f "$file" ]; then
#        # 使用正则提取文件名中间和末尾的数字
#        # 假设文件名格式是 SSTable_<中间数字>-<末尾数字>
#        if [[ $file =~ SSTable_([0-9]+)-([0-9]+) ]]; then
#            middle_number="${BASH_REMATCH[1]}"
#            end_number="${BASH_REMATCH[2]}"
#
#            # 根据末尾数字的奇偶性进行分类
#            if (( end_number % 2 == 0 )); then
#                #echo "偶数文件: $file -> 复制到 $RAID0_DIR"
#                cp "$file" "$RAID0_DIR/"
#            else
#                #echo "奇数文件: $file -> 复制到 $RAID1_DIR"
#                cp "$file" "$RAID1_DIR/"
#            fi
#        fi
#    fi
#done
#
#echo "文件分类完成！"
