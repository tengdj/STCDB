#include "../tracing/trace.h"

//inline int get_rand_number(int max_value){
//    return rand()%max_value+1;
//}
//
//inline struct timeval get_cur_time(){
//    struct timeval t1;
//    gettimeofday(&t1, NULL);
//    return t1;
//}
//
//inline double get_time_elapsed(struct timeval &t1, bool update_start = false){
//    struct timeval t2;
//    double elapsedTime;
//    gettimeofday(&t2, NULL);
//    // compute and print the elapsed time in millisec
//    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
//    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
//    if(update_start){
//        t1 = get_cur_time();
//    }
//    return elapsedTime;
//}

int main(){
    __uint128_t **h_keys = new __uint128_t *[5];
    for(int i=0;i<5;i++){
        h_keys[i] = new __uint128_t[44739243];
        for(int j=0;j<44739243;j++){
            h_keys[i][j] = get_rand_number(1<<30)*get_rand_number(1<<30)*get_rand_number(1<<30);
        }
    }
    //不用排序，模拟这个过程比较次数是一样的，没必要排序。

    uint offset = 0;
    box a;
    //merge sort
    ofstream SSTable_of;
    //bool of_open = false;
    uint kv_count = 0;
    uint sst_count = 0;
    uint sst_capacity = 218454;     //218454   10G /1024
    key_value *temp_kvs = new key_value[sst_capacity];
    uint *key_index = new uint[5]{0};
    int finish = 0;
    double bg_merge_time = 0,bg_open_time=0,bg_flush_time=0;
    struct timeval bg_start = get_cur_time();
    while(finish<5){
        if(kv_count==0){
            SSTable_of.open("../../store/SSTable_"+to_string(sst_count), ios::out | ios::trunc);
            assert(SSTable_of.is_open());
            bg_open_time += get_time_elapsed(bg_start,true);
        }
        finish = 0;
        __uint128_t temp_key = (__uint128_t)1<<126;
        uint take_id =0;
        for(int i=0;i<5; i++){
//            if( bench->h_keys[offset+i][key_index[i]] == 0){              //empty kv
//                finish++;
//                continue;
//            }
            if(key_index[i]>= 44739243){              //empty kv
                finish++;
                continue;
            }
            if( temp_key > h_keys[offset+i][key_index[i]] ){
                temp_key = h_keys[offset+i][key_index[i]];
                take_id = i;
            }
        }
        if(finish<5){
            h_keys[offset+take_id][key_index[take_id]] = 0;                                     //init
            //box temp_box = bench->h_box_block[offset+take_id][bench->h_values[offset+take_id][key_index[take_id]]];               //bench->  i find the right 2G, then in box_block[ h_values ]
            key_index[take_id]++;                                                   // one while, one kv
//            print_128(temp_key);
//            cout<< ": "<< temp_box->low[0] << endl;
            temp_kvs[kv_count].key = temp_key;
            temp_kvs[kv_count].value = a;     //box
            kv_count++;
        }
        if(kv_count==sst_capacity||finish==5){
            bg_merge_time += get_time_elapsed(bg_start,true);
            SSTable_of.write((char *)temp_kvs, sizeof(key_value)*sst_capacity);
            SSTable_of.flush();
            SSTable_of.close();
            bg_flush_time += get_time_elapsed(bg_start,true);
            sst_count++;
            kv_count = 0;
        }
    }
    fprintf(stdout,"\tmerge sort:\t%.2f\n",bg_merge_time);
    fprintf(stdout,"\tflush:\t%.2f\n",bg_flush_time);
    fprintf(stdout,"\topen:\t%.2f\n",bg_open_time);

}