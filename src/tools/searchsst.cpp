#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

//range query
bool search_SSTable(SSTable *ST, uint pid) {
    cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = ST->SSTable_kv_capacity - 1;
    int mid;
    uint temp_pid;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_pid = ST->kv[mid].key/100000000 / 100000000 / 100000000;
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
        temp_pid = ST->kv[cursor].key/100000000 / 100000000 / 100000000;
    }
    if(temp_pid==pid&&cursor==0){
        print_128(ST->kv[0].key);
        cout<<endl;
    }
    while(cursor+1<ST->SSTable_kv_capacity){
        cursor++;
        temp_pid = ST->kv[cursor].key/100000000 / 100000000 / 100000000;
        if(temp_pid==pid){
            print_128(ST->kv[cursor].key);
            cout<<endl;
        }
    }
    cout<<"find !"<<endl;
    return true;
}

bool search_in_disk(sorted_run *bg_run, uint pid){
    cout<<"into disk"<<endl;
    ifstream read_sst;

    //high level binary search
    int find = -1;
    int low = 0;
    int high = bg_run->SSTable_count - 1;
    int mid;
    while (low <= high) {
        mid = (low + high) / 2;
        if (bg_run->first_pid[mid] == pid){
            find = mid;
            break;
        }
        else if (bg_run->first_pid[mid] > pid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        read_sst.open("../store/SSTable"+ to_string(high));                   //final place is not high+1, but high
        cout<<low<<"-"<<bg_run->first_pid[low]<<endl;
        cout<<mid<<"-"<<bg_run->first_pid[mid]<<endl;
        cout<<high<<"-"<<bg_run->first_pid[high]<<endl;
        read_sst.read((char *)bg_run->sst[high].kv,sizeof(key_value)*bg_run->sst[high].SSTable_kv_capacity);
        read_sst.close();
        return search_SSTable(&bg_run->sst[high], pid);
    }
    cout<<"high level binary search finish"<<endl;

    //for the case, there are many SSTables that first_pid==pid
    //find start and end
    uint pid_start = find;
    while(pid_start>=1){
        pid_start--;
        if(bg_run->first_pid[pid_start]!=pid){
            break;
        }
    }
    read_sst.open("../store/SSTable"+ to_string(pid_start));
    read_sst.read((char *)bg_run->sst[pid_start].kv,sizeof(key_value)*bg_run->sst[pid_start].SSTable_kv_capacity);
    search_SSTable(&bg_run->sst[pid_start], pid);
    read_sst.close();
    uint cursor = pid_start+1;
    uint temp_pid;
    while(true){
        read_sst.open("../store/SSTable"+ to_string(cursor));
        read_sst.read((char *)bg_run->sst[cursor].kv,sizeof(key_value)*bg_run->sst[cursor].SSTable_kv_capacity);
        read_sst.close();
        //search_SSTable(&bg_run->sst[cursor], pid);
        if(cursor+1<bg_run->SSTable_count){
            if(bg_run->first_pid[cursor+1]!=pid){               //must shut down in this cursor
                uint index = 0;
                while(index<=bg_run->sst[cursor].SSTable_kv_capacity - 1){
                    temp_pid = bg_run->sst[cursor].kv[index].key/100000000 / 100000000 / 100000000;
                    if(temp_pid==pid){
                        print_128(bg_run->sst[cursor].kv[index].key);
                        cout<<endl;
                    }
                    else break;
                    index++;
                }
                break;
            }
            if(bg_run->first_pid[cursor+1]==pid){               //mustn't shut down in this cursor
                for(uint i=0;i<bg_run->sst[cursor].SSTable_kv_capacity;i++){
                    print_128(bg_run->sst[cursor].kv[i].key);
                    cout<<endl;
                }
            }
            cursor++;
        }
        else {                                           // cursor is the last one, same too bg_run->first_pid[cursor+1]!=pid
            uint index = 0;
            while(index<=bg_run->sst[cursor].SSTable_kv_capacity - 1){
                temp_pid = bg_run->sst[cursor].kv[index].key/100000000 / 100000000 / 100000000;
                if(temp_pid==pid){
                    print_128(bg_run->sst[cursor].kv[index].key);
                    cout<<endl;
                }
                else break;
                index++;
            }
            break;
        }

//        bool check = false;
//        if(cursor+1<bg_run->SSTable_count){
//            if(bg_run->first_pid[cursor+1]!=pid){
//                check = true;
//            }
//        }
//        if(cursor+1<bg_run->SSTable_count){
//            check = true;
//        }
//        if(check){
//            ...
//            break;
//        }
//        else{
//
//        }
    }
    return true;
}

int main(int argc, char **argv){
    sorted_run *bg_run = new sorted_run;
    bg_run->first_pid = new uint[bg_run->SSTable_count];
    bg_run->sst = new SSTable[bg_run->SSTable_count];
    for(int i=0;i<bg_run->SSTable_count;i++){
        bg_run->sst[i].kv = new key_value[bg_run->sst->SSTable_kv_capacity];
    }
    ifstream read_meta;
    read_meta.open("../store/first_pid_meta");
    for(int i=0;i<bg_run->SSTable_count;i++){
        read_meta.read((char *)&bg_run->first_pid[i], sizeof(uint));
        cout<<bg_run->first_pid[i]<<" ";
    }
    cout<<endl;
    read_meta.close();
    cout<<"before func"<<endl;
    uint pid = 0;
    if(search_in_disk(bg_run,pid)){
        cout<<"finally find"<<endl;
    }
    return 0;
}


//#include "../geometry/Map.h"
//#include "../tracing/generator.h"
//#include "../tracing/trace.h"
//
//using namespace std;
//
//int main(int argc, char **argv){
//
//    generator_configuration config = get_generator_parameters(argc, argv);
//    Map *m = new Map(config.map_path);
//    //m->print_region();
//    trace_generator *gen = new trace_generator(&config, m);
//    Point *traces = new Point[config.num_objects*100];
//    tracer *t = new tracer(&config, *m->getMBR(), traces, gen);
//    cout<<"before process"<<endl;
//    t->searchsst_process();
//
//    delete []traces;
//    delete gen;
//    delete t;
//    delete m;
//    return 0;
//}
