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
    __uint128_t key;
    box value;
}key_value;

typedef struct SSTable{
    key_value kv[218454];                  //44739243 * 5 / 1024 = 218454 (Round up)
}SSTable;

class sorted_run {                          //10G

public:
    SSTable * sst = NULL;
    uint SSTable_count = 1024;
    uint SSTable_size = 10*1024*1024;       //useless
    uint SSTable_kv_capacity = 218454;
    uint *first_pid = NULL;
    uint timestamp_min = 0;
    uint timestamp_max = 0;

    void print_meta(){
        fprintf(stdout,"SSTable_count:%d timestamp_min:%d timestamp_max:%d\n",SSTable_count,timestamp_min,timestamp_max);
    }
};

#endif
