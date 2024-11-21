#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

workbench * temp_load_meta(const char *path) {
    log("loading meta from %s", path);
    string bench_path = string(path) + "workbench";
    struct timeval start_time = get_cur_time();
    ifstream in(bench_path, ios::in | ios::binary);
    if(!in.is_open()){
        log("%s cannot be opened",bench_path.c_str());
        exit(0);
    }
    generator_configuration * config = new generator_configuration();
    workbench * bench = new workbench(config);
    in.read((char *)config, sizeof(generator_configuration));               //also read meta
    in.read((char *)bench, sizeof(workbench));      //bench->config = NULL
    bench->config = config;
    bench->ctbs = new CTB[config->big_sorted_run_capacity];
    for(int i = 0; i < 20; i++){
        //CTB temp_ctb;
        string CTB_path = string(path) + "CTB" + to_string(i);
        bench->load_CTB_meta(CTB_path.c_str(), i);
    }
    logt("bench meta load from %s",start_time, bench_path.c_str());
    return bench;
}

void experiment_ctf_search_oid(new_bench *bench){

}

int main(int argc, char **argv){
    string path = "../data/meta/";
    workbench * bench = temp_load_meta(path.c_str());
    new_bench * nb = new new_bench(bench->config);
    memcpy(nb, bench, sizeof(workbench));
    cout << nb->ctb_count << endl;

    nb->ctbs[10].ctfs = new CTF[100];
    nb->load_CTF_keys(10,50);


//    std::ofstream outFile("meetings"+ to_string(10) + '-' + to_string(50) +".csv");
//    if(outFile.is_open()){
//        cout << "open" << endl;
//    }
//    outFile << "id" << ',' << "start" << ',' << "end" << ',' << "box1,box2,box3,box4" << ',' << "person2_id" << endl;
//    for (int i = 0; i < nb->ctbs[10].CTF_capacity[50]; i++) {
//        __uint128_t & temp_key = nb->ctbs[10].ctfs[50].keys[i];
//        uint id = get_key_oid(temp_key);
//        uint end = get_key_end(temp_key) - nb->ctbs[10].start_time_min;
//        uint start = end - get_key_duration(temp_key);
//        box key_box;
//        parse_mbr(temp_key, key_box, nb->ctbs[10].bitmap_mbrs[50]);
//        uint person2_id = get_key_target(temp_key);
//        outFile << id << ',' << start << ',' << end << ','
//            << key_box.low[0] << ',' << key_box.low[1] << ',' << key_box.high[0] << ',' << key_box.high[1] << ',' << person2_id << endl;
//    }
//    outFile.close();

    time_query tq;
    tq.abandon = true;
    ofstream q;
    q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
    q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << ',' << "wid_filter_count" << endl;
    struct timeval total_start = get_cur_time();
    for(int i = 0; i < 10000000; i++){
        uint64_t wp = ((uint64_t)52 << OID_BIT) + i;
        struct timeval prepare_start = get_cur_time();
        uint target_count = nb->ctbs[10].ctfs[50].search_SSTable(wp, &tq, false, nb->ctbs[10].CTF_capacity[50], nb->search_count, nb->search_multi_pid);
        double prepare_consume = get_time_elapsed(prepare_start, true);
        if(target_count > 0)
            cout << i << ',' << prepare_consume << ',' << target_count << endl;
    }
    q.close();
    double total_time = get_time_elapsed(total_start, true);
    cout << "search total " << total_time << endl;

    struct timeval dump_start = get_cur_time();
    ofstream SSTable_of;
    SSTable_of.open("test_ctf" , ios::out|ios::binary|ios::trunc);
    //cout << pargs->path << endl;
    assert(SSTable_of.is_open());
    SSTable_of.write((char *)nb->ctbs[10].ctfs[50].keys, nb->ctbs[10].CTF_capacity[50]);
    SSTable_of.flush();
    SSTable_of.close();
    double dump_time = get_time_elapsed(dump_start, true);
    cout << "dump_time" << dump_time << endl;

    return 0;
}
