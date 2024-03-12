#include "../tracing/trace.h"



int main(){
    configuration config;
    workbench *bench = new workbench(&config);
    ifstream read_sst;

    uint timestamp_min,timestamp_max,SSTable_count;
    string filename = "../store/SSTable_0-meta";
    read_sst.open(filename);
    read_sst.read((char *)&timestamp_min,sizeof(uint));
    read_sst.read((char *)&timestamp_max,sizeof(uint));
    read_sst.read((char *)&SSTable_count,sizeof(uint));
    //key_value first_kv;

    //bench->bg_run = new sorted_run[5];
    key_value **kvs = new key_value *[5];
    for(int i=0;i<5;i++){
        kvs[i] = new key_value[67108864];
    }
    uint SSTable_kv_capacity = 65536;               //67108864/1024 = 65536
    key_value * temp_kvs = new key_value [5];
    for(int i=0;i<5;i++){
        for(int j=0;j<SSTable_count;j++){
            string filename = "../store/SSTable_"+to_string(i)+"-"+to_string(j);
            read_sst.open(filename);
            read_sst.read((char *)temp_kvs,sizeof(key_value)*SSTable_kv_capacity);
            //memcpy(kvs[i]+SSTable_kv_capacity*j, temp_kvs, sizeof(key_value)*SSTable_kv_capacity);
            memcpy(&kvs[i][SSTable_kv_capacity*j], temp_kvs, sizeof(key_value)*SSTable_kv_capacity);
            read_sst.close();
        }
    }



    return 0;
}