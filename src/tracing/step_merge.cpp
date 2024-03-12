/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "step_merge.h"

SSTable::~SSTable(){
    delete []kv;
}

//range query
bool SSTable::search_SSTable(uint pid, bool search_multi, uint &search_multi_length, uint *search_multi_pid) {
    cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = SSTable_kv_capacity - 1;
    int mid;
    uint temp_pid;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_pid = kv[mid].key >> 39;
        cout<<"temp_pid:"<<temp_pid<<endl;
        if ( temp_pid == pid){
            find = mid;
            break;
        }
        else if (temp_pid > pid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        cout<<"cannot find"<<endl;
        return false;
    }
    cout<<"exactly find"<<endl;
    uint cursor = find;
    while(temp_pid==pid&&cursor>=1){
        cursor--;
        temp_pid = kv[cursor].key >> 39;
    }
    if(temp_pid==pid&&cursor==0){
        cout<<kv[0].key<<endl;
        if(search_multi){
            search_multi_pid[search_multi_length] = kv[0].key & ((1ULL << 25) - 1);
            search_multi_length++;
        }
    }
    while(cursor+1<SSTable_kv_capacity){
        cursor++;
        temp_pid = kv[cursor].key >> 39;
        if(temp_pid==pid){
            cout<<kv[cursor].key<<endl;
            if(search_multi){
                search_multi_pid[search_multi_length] = kv[cursor].key & ((1ULL << 25) - 1);
                search_multi_length++;
            }
        }
    }
    cout<<"find !"<<endl;
    return true;
}

sorted_run::~sorted_run(){
    delete []sst;
    delete []first_pid;
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
//        if (first_pid[mid] == pid){
//            find = mid;
//            break;
//        }
//        else if (first_pid[mid] > pid){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        cout<<"not find in first_pid"<<endl;
//        string filename = "../store/SSTable_"+to_string(big_sort_id)+"-"+to_string(high);
//        cout<<filename<<endl;
//        read_sst.open(filename);                   //final place is not high+1, but high
//        assert(read_sst.is_open());
//        cout<<low<<"-"<<first_pid[low]<<endl;
//        cout<<mid<<"-"<<first_pid[mid]<<endl;
//        cout<<high<<"-"<<first_pid[high]<<endl;
//
//        sst[high].kv = new key_value[sst[high].SSTable_kv_capacity];
//        read_sst.read((char *)sst[high].kv,sizeof(key_value)*sst[high].SSTable_kv_capacity);
//        read_sst.close();
//        return sst[high].search_SSTable(pid);
//    }
//    cout<<"high level binary search finish and find"<<endl;
//
//    //for the case, there are many SSTables that first_pid==pid
//    //find start and end
//    uint pid_start = find;
//    while(pid_start>=1){
//        pid_start--;
//        if(first_pid[pid_start]!=pid){
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
//            if (first_pid[cursor + 1] != pid) {               //must shut down in this cursor
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
//            if (first_pid[cursor + 1] == pid) {               //mustn't shut down in this cursor
//                for (uint i = 0; i < sst[cursor].SSTable_kv_capacity; i++) {
//                    cout << sst[cursor].kv[i].key << endl;
//                }
//            }
//            cursor++;
//        } else {                                           // cursor is the last one, same too bg_run->first_pid[cursor+1]!=pid
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
