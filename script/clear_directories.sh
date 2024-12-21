#!/bin/sh

# 定义目录的基础路径
base_path="/data3"

dir_path0="/data3/raid0_num0"
dir_path1="/data3/raid0_num1"

if [ -d "$dir_path0" ]; then
    rm -f "${dir_path0}/temp_ctf"*
fi

if [ -d "$dir_path1" ]; then
    rm -f "${dir_path1}/temp_ctf"*
fi

echo "完成清空操作。"

## 循环从 0 到 7
#for i in $(seq 0 7)
#do
#    # 构建目录路径
#    dir_path="${base_path}/ssd${i}"
#    #echo "处理目录：$dir_path"
#
#    # 检查目录是否存在
#    if [ -d "$dir_path" ]; then
#        #echo "清空目录：$dir_path"
#        # 清空目录内容
#        #echo "${dir_path}/temp_ctf"*
#        rm -f "${dir_path}/temp_ctf"*
#    #else
#        #echo "目录不存在：$dir_path"
#    fi
#done
#
#echo "完成清空操作。"
