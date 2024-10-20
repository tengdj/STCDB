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

class oversize_buffer {             //contact tracing block
public:
    uint oversize_kv_count = 0;
    __uint128_t * keys = NULL;
    f_box * boxes = NULL;

    ~oversize_buffer();
    uint search_buffer(uint32_t oid, time_query * tq);
};

class CTF{                          //contact tracing file
public:
    __uint128_t * keys = NULL;
    //uint SSTable_kv_capacity = 327680;              //67108864 * 5 / 1024 = 327,680 (Round up)

    ~CTF();
    uint search_SSTable(uint64_t wp, bool search_multi, uint SSTable_kv_capacity, uint &search_multi_length, uint *search_multi_pid);
    //uint search_SSTable(uint64_t wp, uint SSTable_kv_capacity, vector<__uint128_t> & v_keys, vector<uint> & v_indices);
};

class CTB {             //contact tracing block

public:
    CTF * ctfs = NULL;
    uint CTF_count = 0;
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
};


#endif