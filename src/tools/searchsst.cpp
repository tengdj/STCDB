#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

//range query
bool search_SSTable(SSTable *ST, uint pid) {
    cout<<"into search_SSTable"<<endl;
    int low = 0;
    int find = -1;
    int high = 218454 - 1;
    int mid;
    uint temp_pid;
    while (low <= high) {
        mid = (low + high) / 2;
        temp_pid = ST->kv[mid].key/100000000 / 100000000 / 100000000;
        if ( temp_pid == pid){
            find = mid;
        }
        else if (temp_pid > pid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        return false;
    }
    uint cursor = find;
    while(temp_pid==pid&&cursor>=1){
        cursor--;
        temp_pid = ST->kv[cursor].key/100000000 / 100000000 / 100000000;
    }
    if(temp_pid==pid&&cursor==0){
        print_128(ST->kv[0].key);
    }
    while(cursor+1<218454){
        cursor++;
        temp_pid = ST->kv[cursor].key/100000000 / 100000000 / 100000000;
        if(temp_pid==pid){
            print_128(ST->kv[cursor].key);
        }
    }
    return true;
}

bool search_in_disk(workbench *bench){
    uint pid = 100000;
    cout<<"into disk"<<endl;
    ifstream read_sst;

    //binary search
    int low = 0;
    int find = -1;
    int high = bench->bg_run->SSTable_count - 1;
    int mid;
    while (low <= high) {
        mid = (low + high) / 2;
        if (bench->bg_run->first_pid[mid] == pid){
            find = mid;
        }
        else if (bench->bg_run->first_pid[mid] > pid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        SSTable ST;
        read_sst.open("../store/SSTable"+ to_string(mid));                   //final place is not high+1, but mid; usually mid == high+1
        read_sst.read((char *)&ST,sizeof(SSTable));
        return search_SSTable(&ST, pid);
    }

    //for the case, there are many SSTables that first_pid==pid
    bench->bg_run->sst = new SSTable[bench->bg_run->SSTable_count];
    //find start and end
    uint pid_start = find,pid_end = find;           //=mid
    while(pid_start>=1){
        pid_start--;
        if(bench->bg_run->first_pid[pid_start]!=find){
            break;
        }
    }
    read_sst.open("../store/SSTable"+ to_string(pid_start));
    read_sst.read((char *)&bench->bg_run->sst[pid_start],sizeof(SSTable));
    search_SSTable(&bench->bg_run->sst[pid_start], pid);
    uint cursor = pid_start;
    uint temp_pid;
    while(true){
        if(cursor+1<bench->bg_run->SSTable_count){
            read_sst.open("../store/SSTable"+ to_string(cursor));
            read_sst.read((char *)&bench->bg_run->sst[cursor],sizeof(SSTable));
            search_SSTable(&bench->bg_run->sst[cursor], pid);
        }
        else break;
        if(bench->bg_run->first_pid[cursor+1]!=find){               //must shut down in this cursor
            uint index = 0;
            while(index<=218454 - 1){
                temp_pid = bench->bg_run->sst[cursor].kv[index].key/100000000 / 100000000 / 100000000;
                if(temp_pid==pid){
                    print_128(bench->bg_run->sst[cursor].kv[index].key);
                }
                else break;
                index++;
            }
            break;
        }
        if(bench->bg_run->first_pid[cursor+1]==find){               //mustn't shut down in this cursor
            for(uint i=0;i<218454;i++){
                print_128(bench->bg_run->sst[cursor].kv[i].key);
            }
        }
        cursor++;
    }
    return true;
}

int main(int argc, char **argv){
    generator_configuration config = get_generator_parameters(argc, argv);
    workbench bench(&config);
    bench.bg_run = new sorted_run;
    bench.bg_run->first_pid = new uint[bench.bg_run->SSTable_count];
    ifstream read_meta;
    read_meta.open("../store/first_pid_meta");
    for(int i=0;i<bench.bg_run->SSTable_count;i++){
        read_meta.read((char *)&bench.bg_run->first_pid[i], sizeof(uint));
        cout<<bench.bg_run->first_pid[i]<<" ";
    }
    cout<<endl;
    read_meta.close();
    cout<<"before func"<<endl;
    cout<<"get the address right"<<endl;
    search_in_disk(&bench);

    return 0;
}
