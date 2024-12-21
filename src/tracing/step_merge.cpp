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
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = oversize_kv_count - 1;
    int mid;
    uint64_t temp_oid;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_oid = get_key_oid(keys[mid]);
        if (temp_oid == oid){
            find = mid;
            high = mid - 1;
        }
        else if (temp_oid > oid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        //cout<<"cannot find"<<endl;
        return 0;
    }
    //cout<<"exactly find"<<endl;
    uint cursor = find;
    while(cursor < oversize_kv_count){
        temp_oid = get_key_oid(keys[cursor]);
        if(temp_oid == oid){
            if(tq->abandon || tq->check_key_time(keys[cursor])) {
                count++;
                //cout<<get_key_target(keys[cursor])<<endl;
                if (search_multi) {
                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
                }
            }
        }
        else break;
        cursor++;
    }
    //cout<<"find !"<<endl;
    return count;
}

CTF::~CTF(){
    delete []keys;
}

//range query
uint CTF::search_SSTable(uint pid, time_query * tq, bool search_multi, uint SSTable_kv_capacity, atomic<long long> &search_count, uint *search_multi_pid) {
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = SSTable_kv_capacity - 1;
    int mid;
    uint temp_pid;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_pid = get_key_oid(keys[mid]);
        if (temp_pid == pid){
            find = mid;
            high = mid - 1;
        }
        else if (temp_pid > pid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        //cout<<"cannot find"<<endl;
        return 0;
    }
    //cout<<"exactly find"<<endl;
    uint cursor = find;
    while(cursor < SSTable_kv_capacity){
        temp_pid = get_key_oid(keys[cursor]);
        if(temp_pid == pid){
            if(tq->abandon || tq->check_key_time(keys[cursor])) {
                count++;
                //cout<< temp_pid << "-" << get_key_target(keys[cursor]) << endl;
                if (search_multi) {
                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
                }
            }
        }
        else break;
        cursor++;
    }
    //cout<<"find !"<<endl;
    return count;
}



CTB::~CTB(){
    if(ctfs)
        delete []ctfs;
    if(first_widpid)
        delete []first_widpid;
    if(sids)
        delete []sids;
    if(bitmaps)
        delete []bitmaps;
    if(bitmap_mbrs)
        delete []bitmap_mbrs;
    if(CTF_capacity)
        delete []CTF_capacity;
    if(box_rtree)
        delete box_rtree;
}