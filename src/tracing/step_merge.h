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
}key_value;

class SSTable{

public:
    key_value *kv = NULL;
    uint SSTable_kv_capacity = 218454;              //44739243 * 5 / 1024 = 218454 (Round up)

    ~SSTable();
    bool search_SSTable(uint pid, bool search_multi, uint &search_multi_length, uint *search_multi_pid);
};

class sorted_run {                          //10G

public:
    SSTable * sst = NULL;
    uint SSTable_count = 0;
    //uint SSTable_size = 10*1024*1024;       //10M   useless
    uint *first_pid = NULL;
    uint timestamp_min = 0;
    uint timestamp_max = 0;

    ~sorted_run();
    void print_meta(){
        fprintf(stdout,"SSTable_count:%d timestamp_min:%d timestamp_max:%d\n",SSTable_count,timestamp_min,timestamp_max);
    }
    //bool search_in_disk(uint big_sort_id, uint pid);
};

#endif