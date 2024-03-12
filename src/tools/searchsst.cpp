#include "../geometry/Map.h"
#include "../tracing/trace.h"

using namespace std;

//SSTable::~SSTable(){
//    delete []kv;
//}
//
////range query
//bool SSTable::search_SSTable(uint pid, bool search_multi, uint &search_multi_length, uint *search_multi_pid) {
//    cout<<"into search_SSTable"<<endl;
//    int find = -1;
//    int low = 0;
//    int high = SSTable_kv_capacity - 1;
//    int mid;
//    uint temp_pid;
//    while (low <= high) {
//        mid = (low + high) / 2;
//        temp_pid = kv[mid].key >> 39;
//        if ( temp_pid == pid){
//            find = mid;
//            break;
//        }
//        else if (temp_pid > pid){
//            high = mid - 1;
//        }
//        else {
//            low = mid + 1;
//        }
//    }
//    if(find==-1){
//        cout<<"cannot find"<<endl;
//        return false;
//    }
//    cout<<"exactly find"<<endl;
//    uint cursor = find;
//    while(temp_pid==pid&&cursor>=1){
//        cursor--;
//        temp_pid = kv[cursor].key >> 39;
//    }
//    if(temp_pid==pid&&cursor==0){
//        cout<<kv[0].key<<endl;
//        if(search_multi){
//            search_multi_pid[search_multi_length] = kv[0].key & ((1ULL << 25) - 1);
//            search_multi_length++;
//        }
//    }
//    while(cursor+1<SSTable_kv_capacity){
//        cursor++;
//        temp_pid = kv[cursor].key >> 39;
//        if(temp_pid==pid){
//            cout<<kv[cursor].key<<endl;
//            if(search_multi){
//                search_multi_pid[search_multi_length] = kv[cursor].key & ((1ULL << 25) - 1);
//                search_multi_length++;
//            }
//        }
//    }
//    cout<<"find !"<<endl;
//    return true;
//}


int main(int argc, char **argv){
    configuration config;
    workbench *bench = new workbench(&config);

    bench->big_sorted_run_count = 1;
    bench->bg_run = new sorted_run[2];
    ifstream read_sst;
    string filename = "../store/SSTable_0-meta";
    read_sst.open(filename);
    read_sst.read((char *)&bench->bg_run[0].timestamp_min,sizeof(uint));
    read_sst.read((char *)&bench->bg_run[0].timestamp_max,sizeof(uint));
    read_sst.read((char *)&bench->bg_run[0].SSTable_count,sizeof(uint));
    bench->bg_run[0].first_pid = new uint[bench->bg_run[0].SSTable_count];
    read_sst.read((char *)bench->bg_run[0].first_pid, sizeof(uint)*bench->bg_run[0].SSTable_count);
    for(int i = 0;i<100; i++){
        bench->search_in_disk(get_rand_number(9000000),100);
    }
    return 0;
}
