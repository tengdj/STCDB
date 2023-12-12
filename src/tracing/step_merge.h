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

class sorted_run {

public:
    SSTable * sst = NULL;
    uint SSTable_count = 1024;
    uint SSTable_size = 10*1024*1024;
    uint *first_pid = NULL;
    uint timestamp_max = 0;
    uint timestamp_min = 0;
};

#endif
