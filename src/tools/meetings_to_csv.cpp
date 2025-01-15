#include "../geometry/Map.h"
int main(){
    return 0;
}

//#include "../geometry/Map.h"
//#include "../tracing/generator.h"
//#include "../tracing/trace.h"
//
//using namespace std;
//
//workbench * temp_load_meta(const char *path) {
//    log("loading meta from %s", path);
//    string bench_path = string(path) + "workbench";
//    struct timeval start_time = get_cur_time();
//    ifstream in(bench_path, ios::in | ios::binary);
//    if(!in.is_open()){
//        log("%s cannot be opened",bench_path.c_str());
//        exit(0);
//    }
//    generator_configuration * config = new generator_configuration();
//    workbench * bench = new workbench(config);
//    in.read((char *)config, sizeof(generator_configuration));               //also read meta
//    in.read((char *)bench, sizeof(workbench));      //bench->config = NULL
//    bench->config = config;
//    bench->ctbs = new CTB[config->big_sorted_run_capacity];
//    for(int i = 0; i < 20; i++){
//        //CTB temp_ctb;
//        string CTB_path = string(path) + "CTB" + to_string(i);
//        bench->load_CTB_meta(CTB_path.c_str(), i);
//    }
//    logt("bench meta load from %s",start_time, bench_path.c_str());
//    return bench;
//}
//
//string formatBox(const box& key_box) {
//    std::ostringstream oss;
//    oss << "POLYGON(("
//        << key_box.low[0] << " " << key_box.low[1] << ", "
//        << key_box.low[0] << " " << key_box.high[1] << ", "
//        << key_box.high[0] << " " << key_box.high[1] << ", "
//        << key_box.high[0] << " " << key_box.low[1] << ", "
//        << key_box.low[0] << " " << key_box.low[1]  // Close the polygon
//        << "))";
//    return oss.str();
//}
//
//bool areFirstFourDigitsEqual(double a, double b) {
//    return std::fabs(a - b) < 0.0001; // 判断两数差值是否小于 0.0001
//}
//
//// 检查两点是否接近（前 4 位精度）
//bool are_points_equal(double x1, double y1, double x2, double y2) {
//    const double epsilon = 0.0001; // 精度
//    return std::fabs(x1 - x2) < epsilon && std::fabs(y1 - y2) < epsilon;
//}
//
//std::string generate_wkt(box & b) {
//    int same_count = 0;
//    if(areFirstFourDigitsEqual(b.low[0], b.high[0])) same_count++;
//    if(areFirstFourDigitsEqual(b.low[1], b.high[1])) same_count++;
//    if (same_count == 2) {
//        return "POINT(" + std::to_string(b.low[0]) + " " + std::to_string(b.low[1]) + ")";
//    }
//    if (same_count == 1){
//        return "LINESTRING(" + std::to_string(b.low[0]) + " " + std::to_string(b.low[1]) + ","
//                               + std::to_string(b.high[0]) + " " + std::to_string(b.high[1]) + ")";
//    }
//    //if (same_count == 0){
//    return formatBox(b);
//}
//
//
//int main(int argc, char **argv){
//    string path = "../data/meta/";
//    workbench * nb = temp_load_meta(path.c_str());
//    //new_bench * nb = new new_bench(bench->config);
//    //memcpy(nb, bench, sizeof(workbench));
//    cout << nb->ctb_count << endl;
//    char new_raid[24] = "/data3/raid0_num";
//    memcpy(nb->config->raid_path, new_raid, sizeof(nb->config->raid_path));
//
//    for(int j = 0; j < 100; j++){
//        uint size = nb->ctbs[10].CTF_capacity[j] * sizeof(__uint128_t);
//        printf("%.2f MB ctf %d\n", size / 1024.0 / 1024.0, j);
//    }
//
//    nb->ctbs[10].ctfs = new CTF[100];
//
////    std::ofstream pg_outFile("pg_meetings"+ to_string(10) +".csv");
////    if(pg_outFile.is_open()){
////        cout << "open" << endl;
////    }
////    pg_outFile << "ctfid" << ',' << "oid" << ',' << "target" << ',' << "start_time" << ',' << "end_time" << ',' << "geom" << endl;
//
//    std::ofstream outFile("mem_meetings"+ to_string(10) + ".csv");
//    if(outFile.is_open()){
//        cout << "open" << endl;
//    }
//    outFile << "id" << ',' << "start" << ',' << "end" << ',' << "box1,box2,box3,box4" << ',' << "person2_id" << endl;
//    for(int j = 0; j < 100; j++){
//        nb->load_CTF_keys(10,j);
////        for (int i = 0; i < nb->ctbs[10].CTF_capacity[j]; i++) {
////            __uint128_t & temp_key = nb->ctbs[10].ctfs[j].keys[i];
////            uint id = cuda_get_key_oid(temp_key);
////            uint end = 546;
////            uint start = 235;
////            uint person2_id = cuda_get_key_target(temp_key);
////            pg_outFile << j << ','
////                    << id << ','
////                    << person2_id << ','
////                    << start << ','
////                    << end << ',';
////            box key_box;
////            parse_mbr(temp_key, key_box, nb->ctbs[10].bitmap_mbrs[j]);
////            pg_outFile << "\"" << generate_wkt(key_box) << "\"" << endl;
////        }
//
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
//    }
//    outFile.close();
////    pg_outFile.close();
//
//
//    return 0;
//}
