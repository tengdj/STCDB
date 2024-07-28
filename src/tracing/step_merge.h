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
#include "../index/RTree.h"
#include "../cuda/cuda_util.cuh"

typedef struct oversize_buffer{
    uint oversize_kv_count = 0;
    __uint128_t * keys = NULL;
    f_box * boxes = NULL;
}oversize_buffer;

typedef struct key_value{
    uint64_t key;
    uint64_t value;
}key_value;                         //sizeof(key_value)==32, beacuse 8+16=24, but 24<2*16=32

class CTF{                          //contact tracing file
public:
    __uint128_t * keys = NULL;
    //key_value *kv = NULL;
    //uint SSTable_kv_capacity = 327680;              //67108864 * 5 / 1024 = 327,680 (Round up)

    ~CTF();
    uint search_SSTable(uint64_t wp, bool search_multi, uint SSTable_kv_capacity, uint &search_multi_length, uint *search_multi_pid);
    //uint search_SSTable(uint64_t wp, uint SSTable_kv_capacity, vector<__uint128_t> & v_keys, vector<uint> & v_indices);
};

class CTB {             //contact tracing block

public:
    CTF * ctfs = NULL;
    uint SSTable_count = 0;
    uint start_time_min = 0;
    uint start_time_max = 0;
    uint end_time_min = 0;
    uint end_time_max = 0;
    uint64_t *first_widpid = NULL;
    unsigned short * sids = NULL;
    unsigned char * bitmaps = NULL;
    box * bitmap_mbrs = NULL;
    uint * CTF_capacity = NULL;
    oversize_buffer o_buffer;
    RTree<short *, double, 2, double> *box_rtree = NULL;

    ~CTB();
    void print_meta(){
        fprintf(stdout,"start_time:%d~%d,end_time:%d~%d\n",start_time_min,start_time_max,end_time_min,end_time_max);
    }
    //uint search_in_sorted_run(uint big_sort_id, uint pid);
};

#endif