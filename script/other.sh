
#for i in $(seq 5 5 50); do
#    output_file="zone_capacity_out_${i}.txt"
#    error_file="zone_capacity_err_${i}.txt"
#
#    ../build/pipeline -d 1000 -o 10000000 -g -p -m 5 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity 50 --zone_capacity "$i" --refine_size 8 \
#        1>"$output_file" 2>"$error_file"
#done
#echo "zone_capacity finish"

#for i in $(seq 10 1 10); do
#    output_file="max_distance_out_${i}.txt"
#    error_file="max_distance_err_${i}.txt"
#
##--refine_size 32
#    ../build/pipeline -d 100 -o 10000000 -g -p -m 5 -r "$i" --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity 50 --zone_capacity 20 --refine_size 32 -b 40000000 \
#        1>"$output_file" 2>"$error_file"
#done
#echo "max_distance reachable distance finish"

#for i in $(seq 1 1 10); do
#    output_file="min_duration_out_${i}.txt"
#    error_file="min_duration_err_${i}.txt"
#
#    ../build/pipeline -d 1000 -o 10000000 -g -p -m "$i" -r 2 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity 50 --zone_capacity 20 \
#        1>"$output_file" 2>"$error_file"
#done
#echo "min_duration Minimum contact duration finish"

#for i in $(seq 30000000 10000000 70000000); do
#    output_file="meeting_buckets_out_${i}.txt"
#    error_file="meeting_buckets_err_${i}.txt"
#
#    ../build/pipeline -d 1000 -o 10000000 -g -p -m 5 -r 2 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity 50 --zone_capacity 20 -b "$i" \
#        1>"$output_file" 2>"$error_file"
#done
#
#echo "meeting_buckets finish"

../build/pipeline -d 1000 -o 10000000 -g -p -m 5 -r 2 --memTable_capacity 2 --G_bytes 2 --load_data --grid_capacity 50 --zone_capacity 20 -b 40000000 --disable_dynamic_schema \
1>rebuild_out.txt 2>rebuild_err.txt