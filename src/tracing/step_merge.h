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
#include "../index/bplustree.h"
//#include "../cuda/cuda_util.cuh"

class old_oversize_buffer {             //contact tracing block
public:
    uint oversize_kv_count = 0;
    __uint128_t * keys = NULL;
    f_box * boxes = NULL;

    ~old_oversize_buffer();
    //uint search_buffer(uint32_t oid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid);
};

class oversize_buffer {             //contact tracing block
public:
    uint oversize_kv_count = 0;
    __uint128_t * keys = NULL;
    f_box * boxes = NULL;
    unsigned char * o_bitmaps = NULL;
    uint start_time_min = 0;
    uint start_time_max = 0;
    uint end_time_min = 0;
    uint end_time_max = 0;

    ~oversize_buffer();
    uint search_buffer(uint32_t oid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid);
    uint o_time_search(time_query * tq);

    void print_buffer();
    void write_o_buffer(box map_mbr, uint bit_count);

};

class CTF{                          //contact tracing file
public:
    f_box ctf_mbr;
    __uint128_t * keys = nullptr;
    unsigned char * bitmap = nullptr;
    uint CTF_kv_capacity = 0;
    uint start_time_min = 0;
    uint start_time_max = 0;
    uint end_time_min = 0;
    uint end_time_max = 0;
    uint16_t key_bit = 0;
    uint16_t id_bit = 0;
    uint16_t duration_bit = 0;
    uint16_t end_bit = 0;
    uint16_t low_x_bit = 0;
    uint16_t low_y_bit = 0;
    uint16_t edge_bit = 0;
    uint16_t mbr_bit = 0;
    uint16_t x_grid = 0;
    uint16_t y_grid = 0;
    uint16_t ctf_bitmap_size = 0;          //bytes
//    ~CTF();

    void eight_parallel();
    void get_ctf_bits(box map_mbr, configuration * config);
    uint search_SSTable(uint pid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid);
    uint time_search(time_query * tq);
    __uint128_t serial_key(uint64_t pid, uint64_t target, uint64_t duration, uint64_t end, __uint128_t value_mbr);
    void print_key(__uint128_t key);
    void parse_key(__uint128_t key, uint &pid, uint &target, uint &duration, uint &end, uint64_t value_mbr);
    void print_ctf_meta();
    void dump(const string& path);
    box new_parse_mbr(uint64_t value_mbr);
    f_box new_parse_mbr_f_box(uint64_t value_mbr);
    void transfer_all_in_one();
    int binary_search(uint oid);
    uint ctf_get_key_oid(__uint128_t temp_128);

};


class CTB {             //contact tracing block
public:
    CTF * ctfs = NULL;
    uint CTF_count = 0;
    uint start_time_min = 0;
    uint start_time_max = 0;
    uint end_time_min = 0;
    uint end_time_max = 0;
    uint64_t *first_widpid = NULL;          //useless
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
