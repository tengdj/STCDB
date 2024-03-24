/*
 * working_bench.h
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_STEPMERGE_H_
#define SRC_TRACING_STEPMERGE_H_

#include "../util/util.h"
#include "../util/config.h"
#include "../util/query_context.h"
#include "../geometry/geometry.h"
#include "../index/QTree.h"

typedef struct key_value{
    uint64_t key;
    __uint128_t value;
}key_value;                         //sizeof(key_value)==32, beacuse 8+16=24, but 24<2*16=32

class SSTable{

public:
    key_value *kv = NULL;
    uint SSTable_kv_capacity = 327680;              //67108864 * 5 / 1024 = 327,680 (Round up)

    ~SSTable();
    bool search_SSTable(uint pid, bool search_multi, uint &search_multi_length, uint *search_multi_pid);
};

class sorted_run {                          //10G

public:
    SSTable * sst = NULL;
    uint SSTable_count = 0;
    //uint SSTable_size = 10*1024*1024;       //10M   useless
    uint *first_pid = NULL;
    unsigned short * wids = NULL;
    unsigned char * bitmaps = NULL;
    uint start_time_min = 0;
    uint start_time_max = 0;
    uint end_time_min = 0;
    uint end_time_max = 0;

    ~sorted_run();
    void print_meta(){
        fprintf(stdout,"SSTable_count:%d start_time:%d~%d,end_time:%d~%d\n",SSTable_count,start_time_min,start_time_max,end_time_min,end_time_max);
    }
    //bool search_in_disk(uint big_sort_id, uint pid);
};

#endif