/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "step_merge.h"

SSTable::~SSTable(){
    delete []keys;
}

//range query
uint SSTable::search_SSTable(uint64_t wp, bool search_multi, uint SSTable_kv_capacity, uint &search_multi_length, uint *search_multi_pid) {
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = SSTable_kv_capacity - 1;
    int mid;
    uint64_t temp_wp;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_wp = keys[mid] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
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
        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
    }
    if(temp_wp == wp && cursor == 0){
        count++;
        //cout<<wp<<endl;
        if(search_multi){
            search_multi_pid[search_multi_length] = get_key_pid(keys[0]);
            search_multi_length++;
        }
    }
    while(cursor+1<SSTable_kv_capacity){
        cursor++;
        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
        if(temp_wp == wp){
            count++;
            //cout<<get_key_target(keys[cursor])<<endl;
            if(search_multi){
                search_multi_pid[search_multi_length] = get_key_pid(keys[cursor]);
                search_multi_length++;
            }
        }
        else break;
    }
    //cout<<"find !"<<endl;
    return count;
}

uint SSTable::search_SSTable(uint64_t wp, uint SSTable_kv_capacity, vector<__uint128_t> & v_keys, vector<uint> & v_indices){
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = SSTable_kv_capacity - 1;
    int mid;
    uint64_t temp_wp;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_wp = keys[mid] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
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
        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
    }
    if(temp_wp == wp && cursor == 0){
        count++;
        v_keys.push_back(keys[cursor]);
        v_indices.push_back(cursor);
    }
    while(cursor+1<SSTable_kv_capacity){
        cursor++;
        temp_wp = keys[cursor] >> (PID_BIT + MBR_BIT + DURATION_BIT + END_BIT);
        if(temp_wp == wp){
            count++;
            v_keys.push_back(keys[cursor]);
            v_indices.push_back(cursor);
        }
        else break;
    }
    //cout<<"find !"<<endl;
    return count;
}



sorted_run::~sorted_run(){
    delete []sst;
    delete []first_widpid;
}

//bool sorted_run::search_in_disk(uint big_sort_id, uint pid){                              //this pointer refers to a single sorted_run
//    cout<<"into disk"<<endl;
//    sst = new SSTable[SSTable_count];                   //maybe useful later, should not delete after this func
//    ifstream read_sst;
//
//    //high level binary search
//    int find = -1;
//    int low = 0;
//    int high = SSTable_count - 1;
//    int mid;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        if (first_widpid[mid] == pid){
//            find = mid;
//            break;
//        }
//        else if (first_widpid[mid] > pid){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        cout<<"not find in first_widpid"<<endl;
//        string filename = "../store/SSTable_"+to_string(big_sort_id)+"-"+to_string(high);
//        cout<<filename<<endl;
//        read_sst.open(filename);                   //final place is not high+1, but high
//        assert(read_sst.is_open());
//        cout<<low<<"-"<<first_widpid[low]<<endl;
//        cout<<mid<<"-"<<first_widpid[mid]<<endl;
//        cout<<high<<"-"<<first_widpid[high]<<endl;
//
//        sst[high].kv = new key_value[sst[high].SSTable_kv_capacity];
//        read_sst.read((char *)sst[high].kv,sizeof(key_value)*sst[high].SSTable_kv_capacity);
//        read_sst.close();
//        return sst[high].search_SSTable(pid);
//    }
//    cout<<"high level binary search finish and find"<<endl;
//
//    //for the case, there are many SSTables that first_widpid==pid
//    //find start and end
//    uint pid_start = find;
//    while(pid_start>=1){
//        pid_start--;
//        if(first_widpid[pid_start]!=pid){
//            break;
//        }
//    }
//    read_sst.open("../store/SSTable_"+to_string(big_sort_id)+"-"+to_string(pid_start));
//    assert(read_sst.is_open());
//    sst[pid_start].kv = new key_value[sst[pid_start].SSTable_kv_capacity];
//    read_sst.read((char *)sst[pid_start].kv,sizeof(key_value)*sst[pid_start].SSTable_kv_capacity);
//    sst[pid_start].search_SSTable(pid);
//    read_sst.close();
//    uint cursor = pid_start+1;
//    uint temp_pid;
//    while(true) {
//        read_sst.open("../store/SSTable_" + to_string(big_sort_id) + "-" + to_string(cursor));
//        assert(read_sst.is_open());
//        sst[cursor].kv = new key_value[sst[cursor].SSTable_kv_capacity];
//        read_sst.read((char *) sst[cursor].kv, sizeof(key_value) * sst[cursor].SSTable_kv_capacity);
//        read_sst.close();
//        if (cursor + 1 < SSTable_count) {
//            if (first_widpid[cursor + 1] != pid) {               //must shut down in this cursor
//                uint index = 0;
//                while (index <= sst[cursor].SSTable_kv_capacity - 1) {
//                    temp_pid = sst[cursor].kv[index].key >> 39;
//                    if (temp_pid == pid) {
//                        cout << sst[cursor].kv[index].key << endl;
//                    } else break;
//                    index++;
//                }
//                break;
//            }
//            if (first_widpid[cursor + 1] == pid) {               //mustn't shut down in this cursor
//                for (uint i = 0; i < sst[cursor].SSTable_kv_capacity; i++) {
//                    cout << sst[cursor].kv[i].key << endl;
//                }
//            }
//            cursor++;
//        } else {                                           // cursor is the last one, same too bg_run->first_widpid[cursor+1]!=pid
//            uint index = 0;
//            while (index <= sst[cursor].SSTable_kv_capacity - 1) {
//                temp_pid = sst[cursor].kv[index].key >> 39;
//                if (temp_pid == pid) {
//                    cout << sst[cursor].kv[index].key << endl;
//                } else break;
//                index++;
//            }
//            break;
//        }
//    }
//    return true;
//}
