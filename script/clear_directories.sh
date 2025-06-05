#!/bin/sh


base_path="/data3"

dir_path0="/data3/raid0_num0"
dir_path1="/data3/raid0_num1"

if [ -d "$dir_path0" ]; then
    rm -f "${dir_path0}/temp_ctf"*
fi

if [ -d "$dir_path1" ]; then
    rm -f "${dir_path1}/temp_ctf"*
fi

echo "clear finished."


#for i in $(seq 0 7)
#do

#    dir_path="${base_path}/ssd${i}"
#    #echo "target ：$dir_path"
#

#    if [ -d "$dir_path" ]; then
#        #echo "claer：$dir_path"

#        #echo "${dir_path}/temp_ctf"*
#        rm -f "${dir_path}/temp_ctf"*
#    #else
#        #echo "canot find：$dir_path"
#    fi
#done
#
#echo "finished."
