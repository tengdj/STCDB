/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "step_merge.h"

oversize_buffer::~oversize_buffer() {
    delete []keys;
    delete []boxes;
}

//total binary search must be faster
uint oversize_buffer::search_buffer(uint32_t oid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid) {
//    uint count = 0;
//    //cout<<"into search_SSTable"<<endl;
//    int find = -1;
//    int low = 0;
//    int high = oversize_kv_count - 1;
//    int mid;
//    uint64_t temp_oid;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        temp_oid = get_key_oid(keys[mid]);
//        if (temp_oid == oid){
//            find = mid;
//            high = mid - 1;
//        }
//        else if (temp_oid > oid){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        //cout<<"cannot find"<<endl;
//        return 0;
//    }
//    //cout<<"exactly find"<<endl;
//    uint cursor = find;
//    while(cursor < oversize_kv_count){
//        temp_oid = get_key_oid(keys[cursor]);
//        if(temp_oid == oid){
//            if(tq->abandon || tq->check_key_time(keys[cursor])) {
//                count++;
//                //cout<<get_key_target(keys[cursor])<<endl;
//                if (search_multi) {
//                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
//                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
//                }
//            }
//        }
//        else break;
//        cursor++;
//    }
//    //cout<<"find !"<<endl;
//    return count;
    return 0;
}

void CTF::eight_parallel() {
    assert(key_bit == 0);
    uint temp_bit = id_bit * 2 + duration_bit + end_bit + mbr_bit;
    assert(temp_bit <= 128);
    key_bit = temp_bit / 8 * 8;
    if(key_bit < temp_bit) key_bit += 8;
}

void CTF::get_ctf_bits(box map_mbr, configuration * config){
    id_bit = min_bits_to_store(config->num_objects);
    duration_bit = min_bits_to_store(config->max_meet_time - 1);
    end_bit = duration_bit;     //end is always rest in the range of duration

    //m granularity, is 0.00001
    //left bottom float to 0.00001 granularity  ~=  (map_mbr.high[0] - map_mbr.low[0]) / 30000   ~=   (map_mbr.high[1] - map_mbr.low[1]) / 42000
    uint low_x_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * 30000;
    uint low_y_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * 42000;
    low_x_bit = min_bits_to_store(low_x_grid - 1);
    low_y_bit = min_bits_to_store(low_y_grid - 1);

    //box, 1m granularity
    int x_restriction = 0.008 / (map_mbr.high[0] - map_mbr.low[0]) * 30000;
    int y_restriction = 0.008 / (map_mbr.high[1] - map_mbr.low[1]) * 42000;
    edge_bit = min_bits_to_store((uint)max(x_restriction, y_restriction) - 1);
    mbr_bit = low_x_bit + low_y_bit + 2 * edge_bit;
    assert(mbr_bit <= 64);
    eight_parallel();

    //42000m high, 30000m width, 100m granularity
    int old_width_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * 300;
    int old_high_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * 420;
//    x_grid = max(old_width_grid, 64);
//    y_grid = max(old_high_grid, 64);
    x_grid = old_width_grid;
    y_grid = old_high_grid;
    assert(x_grid >= 1 && x_grid <= 256);
    assert(y_grid >= 1 && y_grid <= 256);
    ctf_bitmap_size = (old_width_grid * old_high_grid + 7) / 8;
    assert(ctf_bitmap_size <= 256 * 256 / 8);

}

__uint128_t CTF::serial_key(uint64_t pid, uint64_t target, uint64_t duration, uint64_t end, __uint128_t value_mbr){
    __uint128_t temp_key = ((__uint128_t)pid << (id_bit + duration_bit + end_bit + mbr_bit)) + ((__uint128_t)target << (duration_bit + end_bit + mbr_bit))
                            + ((__uint128_t)duration << (end_bit + mbr_bit)) + ((__uint128_t)end << (mbr_bit)) + value_mbr;
    return temp_key;
}

void CTF::print_key(__uint128_t key){
    __uint128_t value_mbr = key & ((__uint128_t(1) << mbr_bit) - 1);
    key >>= mbr_bit;
    uint64_t end = key & ((1ULL << end_bit) - 1);
    key >>= end_bit;
    uint64_t duration = key & ((1ULL << duration_bit) - 1);
    key >>= duration_bit;
    uint64_t target = key & ((1ULL << id_bit) - 1);
    key >>= id_bit;
    uint64_t pid = key;

    std::cout << "Parsed Key: " << std::endl;
    std::cout << "  PID: " << pid << std::endl;
    std::cout << "  Target: " << target << std::endl;
    std::cout << "  Duration: " << duration << std::endl;
    std::cout << "  End: " << end << std::endl;
    //std::cout << "  Value MBR: " << value_mbr << std::endl;
}

void CTF::parse_key(__uint128_t key, uint &pid, uint &target, uint &duration, uint &end, uint64_t value_mbr){
    value_mbr = key & ((uint64_t(1) << mbr_bit) - 1);
    key >>= mbr_bit;
    end = key & ((1ULL << end_bit) - 1);
    key >>= end_bit;
    duration = key & ((1ULL << duration_bit) - 1);
    key >>= duration_bit;
    target = key & ((1ULL << id_bit) - 1);
    key >>= id_bit;
    pid = key;
}

box CTF::new_parse_mbr(uint64_t value_mbr){
    uint64_t mask_edge = (1ULL << edge_bit) - 1;
    uint64_t mask_low_y = (1ULL << low_y_bit) - 1;
    uint64_t mask_low_x = (1ULL << low_x_bit) - 1;

    uint64_t y = value_mbr & mask_edge;  
    value_mbr >>= edge_bit;
    uint64_t x = value_mbr & mask_edge;  
    value_mbr >>= edge_bit;
    uint64_t low1 = value_mbr & mask_low_y; 
    value_mbr >>= low_y_bit;
    uint64_t low0 = value_mbr & mask_low_x;

    box b;
    b.low[0] = ctf_mbr.low[0] + (double)low0 / mask_low_x * (ctf_mbr.high[0] - ctf_mbr.low[0]);
    b.low[1] = ctf_mbr.low[1] + (double)low1 / mask_low_y * (ctf_mbr.high[1] - ctf_mbr.low[1]);
    b.high[0] = ctf_mbr.low[0] + (double)x / mask_edge * (ctf_mbr.high[0] - ctf_mbr.low[0]);
    b.high[1] = ctf_mbr.low[1] + (double)y / mask_edge * (ctf_mbr.high[1] - ctf_mbr.low[1]);
    return b;
}

//range query
uint CTF::search_SSTable(uint pid, time_query * tq, bool search_multi, uint SSTable_kv_capacity, atomic<long long> &search_count, uint *search_multi_pid) {
//    uint count = 0;
//    //cout<<"into search_SSTable"<<endl;
//    int find = -1;
//    int low = 0;
//    int high = SSTable_kv_capacity - 1;
//    int mid;
//    uint temp_pid;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        temp_pid = get_key_oid(keys[mid]);
//        if (temp_pid == pid){
//            find = mid;
//            high = mid - 1;
//        }
//        else if (temp_pid > pid){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        //cout<<"cannot find"<<endl;
//        return 0;
//    }
//    //cout<<"exactly find"<<endl;
//    uint cursor = find;
//    while(cursor < SSTable_kv_capacity){
//        temp_pid = get_key_oid(keys[cursor]);
//        if(temp_pid == pid){
//            if(tq->abandon || tq->check_key_time(keys[cursor])) {
//                count++;
//                //cout<< temp_pid << "-" << get_key_target(keys[cursor]) << endl;
//                if (search_multi) {
//                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
//                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
//                }
//            }
//        }
//        else break;
//        cursor++;
//    }
//    //cout<<"find !"<<endl;
//    return count;
    return 0;
}


void CTF::print_ctf_meta() {
    std::cout << "CTF Metadata:" << std::endl;

    std::cout << "CTF MBR: ["
              << ctf_mbr.low[0] << ", " << ctf_mbr.low[1] << "] -> ["
              << ctf_mbr.high[0] << ", " << ctf_mbr.high[1] << "]" << std::endl;

    std::cout << "CTF KV Capacity: " << CTF_kv_capacity << std::endl;

    std::cout << "Start Time Min: " << start_time_min << std::endl;
    std::cout << "Start Time Max: " << start_time_max << std::endl;
    std::cout << "End Time Min: " << end_time_min << std::endl;
    std::cout << "End Time Max: " << end_time_max << std::endl;

    std::cout << "Key Bit: " << key_bit << std::endl;
    std::cout << "ID Bit: " << id_bit << std::endl;
    std::cout << "Duration Bit: " << duration_bit << std::endl;
    std::cout << "End Bit: " << end_bit << std::endl;
    std::cout << "Low X Bit: " << low_x_bit << std::endl;
    std::cout << "Low Y Bit: " << low_y_bit << std::endl;
    std::cout << "Edge Bit: " << edge_bit << std::endl;
    std::cout << "MBR Bit: " << mbr_bit << std::endl;

    std::cout << "X Grid: " << x_grid << std::endl;
    std::cout << "Y Grid: " << y_grid << std::endl;

    std::cout << "CTF Bitmap Size (in bytes): " << ctf_bitmap_size << std::endl;

}

void CTF:: dump(const string& path){
    struct timeval start_time = get_cur_time();
    ofstream wf(path.c_str(), ios::out|ios::binary|ios::trunc);
    wf.write((char *)this, sizeof(CTF));
    wf.write((char *)bitmap, ctf_bitmap_size);
    wf.close();
}

CTB::~CTB(){
    if(ctfs)
        delete []ctfs;
    if(sids)
        delete []sids;
    if(box_rtree)
        delete box_rtree;
}

