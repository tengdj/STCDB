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

string formatBox(const box& key_box) {
    std::ostringstream oss;
    oss << "POLYGON(("
        << key_box.low[0] << " " << key_box.low[1] << ", "
        << key_box.low[0] << " " << key_box.high[1] << ", "
        << key_box.high[0] << " " << key_box.high[1] << ", "
        << key_box.high[0] << " " << key_box.low[1] << ", "
        << key_box.low[0] << " " << key_box.low[1]  // Close the polygon
        << "))";
    return oss.str();
}

int main(int argc, char **argv){
    string path = "../data/meta/";
    workbench * bench = temp_load_meta(path.c_str());
    new_bench * nb = new new_bench(bench->config);
    memcpy(nb, bench, sizeof(workbench));
    cout << nb->ctb_count << endl;

    for(int j = 0; j < 100; j++){
        uint size = nb->ctbs[10].CTF_capacity[j] * sizeof(__uint128_t);
        printf("%.2f MB ctf %d\n", size / 1024.0 / 1024.0, j);
    }

    nb->ctbs[10].ctfs = new CTF[100];
    std::ofstream outFile("meetings"+ to_string(10) +".csv");
    if(outFile.is_open()){
        cout << "open" << endl;
    }
    outFile << "ctfid" << ',' << "oid" << ',' << "target" << ',' << "start_time" << ',' << "end_time" << ',' << "geom" << endl;
    for(int j = 0; j < 100; j++){
        nb->load_CTF_keys(10,j);
        for (int i = 0; i < nb->ctbs[10].CTF_capacity[j]; i++) {
            __uint128_t & temp_key = nb->ctbs[10].ctfs[j].keys[i];
            uint id = get_key_oid(temp_key);
            uint end = 546;
            uint start = 235;
            box key_box;
            parse_mbr(temp_key, key_box, nb->ctbs[10].bitmap_mbrs[j]);
            uint person2_id = get_key_target(temp_key);
            outFile << j << ','
                    << id << ','
                    << person2_id << ','
                    << start << ','
                    << end << ','
                    << "\"" << formatBox(key_box) << "\""
                    << endl;
        }


//        std::ofstream outFile("meetings"+ to_string(10) + '-' + to_string(j) +".csv");
//        if(outFile.is_open()){
//            cout << "open" << endl;
//        }
//        outFile << "id" << ',' << "start" << ',' << "end" << ',' << "box1,box2,box3,box4" << ',' << "person2_id" << endl;
//        for (int i = 0; i < nb->ctbs[10].CTF_capacity[j]; i++) {
//            __uint128_t & temp_key = nb->ctbs[10].ctfs[j].keys[i];
//            uint id = get_key_oid(temp_key);
//            uint end = get_key_end(temp_key) - nb->ctbs[10].start_time_min;
//            uint start = end - get_key_duration(temp_key);
//            box key_box;
//            parse_mbr(temp_key, key_box, nb->ctbs[10].bitmap_mbrs[j]);
//            uint person2_id = get_key_target(temp_key);
//            outFile << id << ',' << start << ',' << end << ','
//                    << key_box.low[0] << ',' << key_box.low[1] << ',' << key_box.high[0] << ',' << key_box.high[1] << ',' << person2_id << endl;
//        }
//        outFile.close();
    }
    outFile.close();



//    time_query tq;
//    tq.abandon = true;
//
//    uint total_count = 0;
//    struct timeval total_start = get_cur_time();
//
//    for(int j = 0; j < 100000; j++){
//        for(int i = 0; i < 100; i++){
//            uint64_t wp = ((uint64_t)(j+2) << OID_BIT) + i;
//            //struct timeval prepare_start = get_cur_time();
//            uint target_count = nb->ctbs[10].ctfs[j].search_SSTable(wp, &tq, false, nb->ctbs[10].CTF_capacity[j], nb->search_count, nb->search_multi_pid);
//            total_count += target_count;
//            //double prepare_consume = get_time_elapsed(prepare_start, true);
////        if(target_count > 0)
////            cout << i << ',' << prepare_consume << ',' << target_count << endl;
//        }
//    }
//
//
//    double total_time = get_time_elapsed(total_start, true);
//    cout << "search total " << total_time << endl;
//    cout << " total_count " << total_count << endl;

//    struct timeval dump_start = get_cur_time();
//    ofstream SSTable_of;
//    SSTable_of.open("test_ctf" , ios::out|ios::binary|ios::trunc);
//    //cout << pargs->path << endl;
//    assert(SSTable_of.is_open());
//    SSTable_of.write((char *)nb->ctbs[10].ctfs[50].keys, nb->ctbs[10].CTF_capacity[50]);
//    SSTable_of.flush();
//    SSTable_of.close();
//    double dump_time = get_time_elapsed(dump_start, true);
//    cout << "dump_time" << dump_time << endl;

    return 0;
}
