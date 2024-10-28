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
            break;
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
    while(temp_oid == oid && cursor >= 1){
        cursor--;
        temp_oid = get_key_oid(keys[cursor]);
    }
    if(temp_oid == oid && cursor == 0){
        if(tq->abandon || tq->check_key_time(keys[cursor])) {
            count++;
            long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
            if(search_multi){
                search_multi_pid[search_multi_length] = get_key_target(keys[0]);
            }
        }

        //cout<<oid<<endl;
    }
    while(cursor+1<oversize_kv_count){
        cursor++;
        temp_oid = get_key_oid(keys[cursor]);
        if(temp_oid == oid){
            if(tq->abandon || tq->check_key_time(keys[cursor])){
                count++;
                long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
                if(search_multi){
                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
                }
            }

            //cout<<oid<<"-"<<get_key_target(keys[cursor])<<endl;
        }
        else break;
    }
    //cout<<"find !"<<endl;
    return count;
}

CTF::~CTF(){
    delete []keys;
}

//range query
uint CTF::search_SSTable(uint64_t wp, time_query * tq, bool search_multi, uint SSTable_kv_capacity, atomic<long long> &search_count, uint *search_multi_pid) {
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = SSTable_kv_capacity - 1;
    int mid;
    uint64_t temp_wp;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_wp = keys[mid] >> (OID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
        if (temp_wp == wp){
            find = mid;
            break;
        }
        else if (temp_wp > wp){
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
    while(temp_wp == wp && cursor >= 1){
        cursor--;
        temp_wp = keys[cursor] >> (OID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
    }
    if(temp_wp == wp && cursor == 0){
        count++;
        //cout<<wp<<endl;
        if(tq->abandon || tq->check_key_time(keys[cursor])) {
            count++;
            long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
            if(search_multi){
                search_multi_pid[search_multi_length] = get_key_target(keys[0]);
            }
        }
    }
    while(cursor+1<SSTable_kv_capacity){
        cursor++;
        temp_wp = keys[cursor] >> (OID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
        if(temp_wp == wp){
            count++;
            //cout<<get_key_target(keys[cursor])<<endl;
            long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
            if(search_multi){
                search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
            }
        }
        else break;
    }
    //cout<<"find !"<<endl;
    return count;
}

//uint SSTable::search_SSTable(uint64_t wp, uint SSTable_kv_capacity, vector<__uint128_t> & v_keys, vector<uint> & v_indices){
//    uint count = 0;
//    cout<<"into search_SSTable"<<endl;
//    int find = -1;
//    int low = 0;
//    int high = SSTable_kv_capacity - 1;
//    int mid;
//    uint64_t temp_wp;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        temp_wp = keys[mid] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
//        if (temp_wp == wp){
//            find = mid;
//            break;
//        }
//        else if (temp_wp > wp){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        cout<<"cannot find"<<endl;
//        return 0;
//    }
//    cout<<"exactly find"<<endl;
//    uint cursor = find;
//    while(temp_wp == wp && cursor >= 1){
//        cursor--;
//        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
//    }
//    if(temp_wp == wp && cursor == 0){
//        count++;
//        v_keys.push_back(keys[cursor]);
//        v_indices.push_back(cursor);
//    }
//    while(cursor+1<SSTable_kv_capacity){
//        cursor++;
//        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
//        if(temp_wp == wp){
//            count++;
//            v_keys.push_back(keys[cursor]);
//            v_indices.push_back(cursor);
//        }
//        else break;
//    }
//    //cout<<"find !"<<endl;
//    return count;
//}



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

//uint sorted_run::search_in_sorted_run(uint big_sort_id, uint pid){                              //this pointer refers to a single sorted_run
//
//    //cout<<"big_sorted_run_num:"<<i<<endl;
//    uint64_t wp = pid;
//
//    wp += ((uint64_t)wids[pid] << PID_BIT);
//    //cout<<"wp: "<<wp<<endl;
//
//    if(!sst){
//        cout << "sth wrong" << endl;
//        return 0;
//    }
//
//    //high level binary search
//    int find = -1;
//    int low = 0;
//    int high = SSTable_count - 1;
//    int mid;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        //cout << first_widpid[mid] << endl;
//        if (first_widpid[mid] == wp){
//            find = mid;
//            break;
//        }
//        else if (first_widpid[mid] > wp){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
////                cout<<"not find in first_widpid"<<endl;
////                cout << low << "-" << first_widpid[low] << endl;
////                cout << mid << "-" << first_widpid[mid] << endl;
////                cout << high << "-" << first_widpid[high] << endl;
//        if(high<0){
//            high = 0;
//        }
//        if(!sst[high].keys){
//            //cout<<"new SSTables keys"<<high<<endl;
//            string filename = "../store/SSTable_"+to_string(i)+"-"+to_string(high);
//            //cout<<filename<<endl;
//            read_sst.open(filename);                   //final place is not high+1, but high
//            assert(read_sst.is_open());
//            sst[high].keys = new __uint128_t [SSTable_kv_capacity];
//            read_sst.read((char *)sst[high].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
//            read_sst.close();
//        }
//        uint target_count = sst[high].search_SSTable(wp,search_multi,SSTable_kv_capacity,search_multi_length,search_multi_pid);
//        id_find_count += target_count;
//        if(target_count){
//            ret = true;
//        }
//        else{
//            id_not_find_count++;
//        }
//        continue;
//    }
//    //cout<<"high level binary search finish and find"<<endl;
//
//    //for the case, there are many SSTables that first_widpid==wp
//    //find start and end
//    uint pid_start = find;
//    while(pid_start>=1){
//        pid_start--;
//        if(first_widpid[pid_start] != wp){
//            break;
//        }
//    }
//    if(!sst[pid_start].keys){
//        read_sst.open("../store/SSTable_"+to_string(i)+"-"+to_string(pid_start));
//        assert(read_sst.is_open());
//        sst[pid_start].keys = new __uint128_t[SSTable_kv_capacity];
//        read_sst.read((char *)sst[pid_start].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
//        read_sst.close();
//    }
//    sst[pid_start].search_SSTable(wp,search_multi,SSTable_kv_capacity,search_multi_length,search_multi_pid);
//    uint cursor = pid_start+1;
//    uint temp_pid;
//    while(true) {
//        read_sst.open("../store/SSTable_" + to_string(i) + "-" + to_string(cursor));
//        assert(read_sst.is_open());
//        if(!sst[cursor].keys){
//            sst[cursor].keys = new __uint128_t[SSTable_kv_capacity];
//        }
//        read_sst.read((char *) sst[cursor].keys, sizeof(__uint128_t) * SSTable_kv_capacity);
//        read_sst.close();
//        if (cursor + 1 < config->SSTable_count) {
//            if (first_widpid[cursor + 1] != wp) {               //must shut down in this cursor
//                cout<<"case 1"<<endl;
//                uint index = 0;
//                while (index <= SSTable_kv_capacity - 1) {
//                    temp_pid = get_key_pid(sst[cursor].keys[index]) ;
//                    if (temp_pid == pid) {
//                        id_find_count++;
//                        //cout << sst[cursor].keys[index] << endl;
//                        if(search_multi){
//                            search_multi_pid[search_multi_length] = get_key_pid(sst[cursor].keys[index]) ;
//                            search_multi_length++;
//                        }
//                    } else break;
//                    index++;
//                }
//                break;
//            }
//            if (first_widpid[cursor + 1] == wp) {               //mustn't shut down in this cursor
//                for (uint j = 0; j < SSTable_kv_capacity; j++) {
//                    //cout << sst[cursor].keys[j] << endl;
//                    if(search_multi){
//                        search_multi_pid[search_multi_length] = get_key_pid(sst[cursor].keys[j]) ;
//                        search_multi_length++;
//                    }
//                }
//            }
//            cursor++;
//        } else {                                           // cursor is the last one, same too bg_run->first_widpid[cursor+1]!=pid
//            uint index = 0;
//            while (index <= SSTable_kv_capacity - 1) {
//                temp_pid = get_key_pid(sst[cursor].keys[index]);
//                cout<<"temp_pid: "<<temp_pid<<endl;
//                if (temp_pid == pid) {
//                    //cout << sst[cursor].keys[index] << endl;
//                    if(search_multi){
//                        search_multi_pid[search_multi_length] = get_key_pid(sst[cursor].keys[index]);
//                        search_multi_length++;
//                    }
//                } else break;
//                index++;
//            }
//            break;
//        }
//    }
//    ret = true;
//
//
//
//}
