#include "../tracing/workbench.h"


using namespace std;

workbench * load_meta(const char *path, uint max_ctb) {
    log("loading meta from %s", path);
    string bench_path = string(path) + "N_workbench";
    struct timeval start_time = get_cur_time();
    ifstream in(bench_path, ios::in | ios::binary);
    if(!in.is_open()){
        log("%s cannot be opened",bench_path.c_str());
        exit(0);
    }
    generator_configuration * config = new generator_configuration();
    workbench * bench = new workbench(config);
    in.read((char *)config, sizeof(generator_configuration));               //also read meta
    cout << bench->ctb_count << "bench->ctb_count" << endl;
    in.read((char *)bench, sizeof(workbench));      //bench->config = NULL
    bench->config = config;
    bench->ctbs = new CTB[config->big_sorted_run_capacity];
#pragma omp parallel for num_threads(bench->config->num_threads)
    for(int i = 0; i < min(max_ctb, bench->ctb_count); i++){
        //N_CTB
        bench->load_CTB_meta(path, i);
    }
    logt("bench meta load from %s",start_time, bench_path.c_str());
    return bench;
}


//workbench * load_meta(const char *path) {
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
//    char new_raid[24] = "/data3/raid0_num";
//    memcpy(config->raid_path, new_raid, sizeof(config->raid_path));
//    //std::copy(new_raid, new_raid + 24, config->raid_path);
//    in.read((char *)bench, sizeof(workbench));      //bench->config = NULL
//    bench->config = config;
//    bench->ctbs = new CTB[config->big_sorted_run_capacity];
//    for(int i = 0; i < bench->ctb_count; i++){
//        //CTB temp_ctb;
//        string CTB_path = string(path) + "CTB" + to_string(i);
//        bench->load_CTB_meta(CTB_path.c_str(), i);
//    }
//    logt("bench meta load from %s",start_time, bench_path.c_str());
//    return bench;
//}
//
//workbench * C_load_meta(const char *path) {
//    log("loading compaction meta from %s", path);
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
//    bench->ctbs = new CTB[20];
//    for(int i = 0; i < 5; i++){
//        //CTB temp_ctb;
//        string CTB_path = string(path) + "C_CTB" + to_string(i);
//        bench->load_CTB_meta(CTB_path.c_str(), i);
//    }
//    logt("bench meta load from %s",start_time, bench_path.c_str());
//    return bench;
//}
//
//void * id_search_unit(void * arg){
//    query_context *ctx = (query_context *)arg;
//    new_bench *bench = (new_bench *)ctx->target[0];
////    uint * pid = (uint *)ctx->target[1];
////    uint * j = (uint *)ctx->target[2];
////    time_query * tq = (time_query *)ctx->target[3];
//    bench->id_search_in_CTB(*(uint *)ctx->target[1], *(uint *)ctx->target[2], (time_query *)ctx->target[3]);
//    return NULL;
//}
//
////void single_experiment_twice(new_bench * bench){
////    bench->search_multi_pid = new uint[1000000];
////    bench->clear_all_keys();
////    clear_cache();
////    time_query tq;
////    tq.abandon = true;
////    ofstream q;
////    q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
////    q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << ',' << "wid_filter_count" << endl;
////    for(int i = 0; i < 1; i++){
////        struct timeval prepare_start = get_cur_time();
////        bench->search_count = 0;
////        bench->wid_filter_count = 0;
////        uint pid = 10000;
////        bench->search_multi = true;
////        uint the_min = min((uint)1215, bench->ctb_count);
////        pthread_t threads[the_min];
////        query_context * tctx = new query_context[the_min];
////        for(uint j = 0; j < the_min; j++){
////            tctx[j].target[0] = (void *)bench;
////            tctx[j].target[1] = (void *)&pid;
////            tctx[j].target[2] = (void *)new int(j);
////            tctx[j].target[3] = (void *)&tq;
////            pthread_create(&threads[j], NULL, id_search_unit, (void *)&tctx[j]);
////            //bench->id_search_in_CTB(pid, j, &tq);
////        }
////        delete[] tctx;
////        tctx = nullptr;
////        double prepare_consume = get_time_elapsed(prepare_start, true);
////        bench->search_multi = false;
////        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << bench->wid_filter_count << endl;
////    }
////    uint search_length = bench->search_count;
////    for(int i = 0; i < search_length; i++){
//////        bench->clear_all_keys();
//////        clear_cache();
////        struct timeval prepare_start = get_cur_time();
////        bench->search_count = 0;
////        bench->wid_filter_count = 0;
////        uint pid = bench->search_multi_pid[i];
////        bench->search_multi = false;
////        uint the_min = min((uint)1215, bench->ctb_count);
////        pthread_t threads[the_min];
////        query_context * tctx = new query_context[the_min];
////        for(uint j = 0; j < the_min; j++){
////            tctx[j].target[0] = (void *)bench;
////            tctx[j].target[1] = (void *)&pid;
////            tctx[j].target[2] = (void *)new int(j);
////            tctx[j].target[3] = (void *)&tq;
////            pthread_create(&threads[j], NULL, id_search_unit, (void *)&tctx[j]);
////            //bench->id_search_in_CTB(pid, j, &tq);
////        }
////        for(int j = 0; j < the_min; j++ ){
////            void *status;
////            pthread_join(threads[j], &status);
////        }
////        delete[] tctx;
////        tctx = nullptr;
////        double prepare_consume = get_time_elapsed(prepare_start, true);
////        bench->search_multi = false;
////        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << bench->wid_filter_count << endl;
////    }
////    q.close();
////    delete[] bench->search_multi_pid;
////};
//
//void experiment_twice(new_bench *bench){
//    bench->search_multi_pid = new uint[1000000];
//    bench->clear_all_keys();
//    clear_cache();
//    time_query tq;
//    tq.abandon = true;
//    ofstream q;
//    q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
//    q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << ',' << "wid_filter_count" << endl;
//    for(int i = 0; i < 1; i++){
//        struct timeval prepare_start = get_cur_time();
//        bench->search_count = 0;
//        bench->wid_filter_count = 0;
//        uint pid = 10000;
//        bench->search_multi = true;
//        uint the_min = min((uint)1215, bench->ctb_count);
//        pthread_t threads[the_min];
//        query_context * tctx = new query_context[the_min];
//        for(uint j = 0; j < the_min; j++){
//            tctx[j].target[0] = (void *)bench;
//            tctx[j].target[1] = (void *)&pid;
//            tctx[j].target[2] = (void *)new int(j);
//            tctx[j].target[3] = (void *)&tq;
//            pthread_create(&threads[j], NULL, id_search_unit, (void *) &tctx[j]);
//            //bench->id_search_in_CTB(pid, j, &tq);
//        }
//        for(int j = 0; j < the_min; j++ ){
//            void *status;
//            pthread_join(threads[j], &status);
//        }
//        delete[] tctx;
//        tctx = nullptr;
//        double prepare_consume = get_time_elapsed(prepare_start, true);
//        bench->search_multi = false;
//        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << bench->wid_filter_count << endl;
//    }
//    uint search_length = bench->search_count;
//    for(int i = 0; i < search_length; i++){
////        bench->clear_all_keys();
////        clear_cache();
//        struct timeval prepare_start = get_cur_time();
//        bench->search_count = 0;
//        bench->wid_filter_count = 0;
//        uint pid = bench->search_multi_pid[i];
//        bench->search_multi = false;
//        uint the_min = min((uint)1215, bench->ctb_count);
//        pthread_t threads[the_min];
//        query_context * tctx = new query_context[the_min];
//        for(uint j = 0; j < the_min; j++){
//            tctx[j].target[0] = (void *)bench;
//            tctx[j].target[1] = (void *)&pid;
//            tctx[j].target[2] = (void *)new int(j);
//            tctx[j].target[3] = (void *)&tq;
//            pthread_create(&threads[j], NULL, id_search_unit, (void *) &tctx[j]);
//            //bench->id_search_in_CTB(pid, j, &tq);
//        }
//        for(int j = 0; j < the_min; j++ ){
//            void *status;
//            pthread_join(threads[j], &status);
//        }
//        delete[] tctx;
//        tctx = nullptr;
//        double prepare_consume = get_time_elapsed(prepare_start, true);
//        bench->search_multi = false;
//        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << bench->wid_filter_count << endl;
//    }
//    q.close();
//    delete[] bench->search_multi_pid;
//}
//
//void exp4_search_oid_single(new_bench *bench){
//    time_query tq;
//    tq.abandon = true;
//    ofstream q;
//    q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
//    q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << ',' << "wid_filter_count" << endl;
//    for(int i = 0; i < 100; i++){
//        bench->clear_all_keys();
//        clear_cache();
//        struct timeval prepare_start = get_cur_time();
//        bench->search_count = 0;
//        bench->wid_filter_count = 0;
//        uint pid = get_rand_number(bench->config->num_objects);
//        bench->search_multi = false;
//        uint the_min = min((uint)1215, bench->ctb_count);
//        //pthread_t threads[the_min];
//        query_context * tctx = new query_context[the_min];
//        for(uint j = 0; j < the_min; j++){
//            tctx[j].target[0] = (void *)bench;
//            tctx[j].target[1] = (void *)&pid;
//            tctx[j].target[2] = (void *)new int(j);
//            tctx[j].target[3] = (void *)&tq;
//            id_search_unit((void *) &tctx[j]);
//            //pthread_create(&threads[j], NULL, id_search_unit, (void *)&tctx[j]);
//            //bench->id_search_in_CTB(pid, j, &tq);
//        }
////        for(int j = 0; j < the_min; j++ ){
////            void *status;
////            pthread_join(threads[j], &status);
////        }
//        delete[] tctx;
//        tctx = nullptr;
//        double prepare_consume = get_time_elapsed(prepare_start, true);
//        bench->search_multi = false;
//        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << bench->wid_filter_count << endl;
//    }
//    q.close();
//}
//
void experiment_search_oid(workbench *bench, uint max_ctb){
    time_query tq;
    tq.abandon = true;
    ofstream q;
    q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
    q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
        << "hit_buffer" << ',' << "hit_ctf" << endl;
    for(int i = 0; i < 1000; i++){
        bench->clear_all_keys();
        clear_cache();
        bench->search_count = 0;
        bench->hit_buffer = 0;
        bench->hit_ctf = 0;
        //bench->wid_filter_count = 0;
        uint pid = get_rand_number(bench->config->num_objects);
        bench->search_multi = false;
        struct timeval prepare_start = get_cur_time();
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < max_ctb; j++){
            bench->id_search_in_CTB(pid, j, &tq);
        }
        double prepare_consume = get_time_elapsed(prepare_start, true);
        bench->search_multi = false;
        q << pid << ',' << prepare_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
            << ',' << bench->hit_buffer << ',' << bench->hit_ctf << endl;
    }
    q.close();
}
//
//void *box_search_unit(void *arg){
//    query_context *ctx = (query_context *)arg;
//    new_bench *bench = (new_bench *)ctx->target[0];
//    box * b = (box *)ctx->target[1];
//    uint mbr_find_count = 0;
//    while(true){
//        size_t start = 0;
//        size_t end = 0;
//        if(!ctx->next_batch(start,end)){
//            break;
//        }
//        for(int obj = start; obj < end; obj++){
//            box_search_info info = bench->box_search_queue[obj];
//            if(!bench->ctbs[info.ctb_id].ctfs[info.ctf_id].keys){
//                bench->load_CTF_keys(info.ctb_id, info.ctf_id);
//            }
//            for(uint q = 0; q < bench->ctbs[info.ctb_id].CTF_capacity[info.ctf_id]; q++){
//                __uint128_t temp_key = bench->ctbs[info.ctb_id].ctfs[info.ctf_id].keys[q];
//                if(info.tq.abandon){
//                    uint pid = get_key_oid(temp_key);
//                    box key_box;
//                    parse_mbr(temp_key, key_box, *info.bmap_mbr);
//                    if(b->intersect(key_box)){
//                        mbr_find_count++;
//                        //cout<<"box find!"<<endl;
//                        //key_box.print();
//                    }
//                }
//            }
//            delete[] bench->ctbs[info.ctb_id].ctfs[info.ctf_id].keys;
//            bench->ctbs[info.ctb_id].ctfs[info.ctf_id].keys = nullptr;
//        }
//    }
//    bench->search_count.fetch_add(mbr_find_count, std::memory_order_relaxed);
//    return NULL;
//}
//
//void exp4_search_box_single(new_bench *bench){
//    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count / 20);
//    time_query tq;
//    tq.abandon = true;
//    ofstream q;
//    q.open("exp4_search_box_single.csv", ios::out|ios::binary|ios::trunc);
//    if(!q.is_open()){
//        log("%s cannot be opened","ex_search_box.csv");
//        exit(0);
//    }
//    q << "search area" << ',' << "find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ',' << "prepare_time(ms)" << endl;
//    for(int i = 0; i < 3; i++){
//        bench->clear_all_keys();
//        clear_cache();
//        struct timeval prepare_start = get_cur_time();
//        bench->search_count = 0;
//        bench->mbr_find_count = 0;
//        bench->intersect_sst_count = 0;
//        bench->bit_find_count = 0;
//        double edge_length = 0.01;
//        Point mid;
//        mid.x = -87.9 + 0.2;
//        mid.y = 41.6 + 0.2;
//        box search_area(mid.x - edge_length/2, mid.y - edge_length/2, mid.x + edge_length/2, mid.y + edge_length/2);
//        search_area.print();
//        for(uint j = 0; j < min((uint)1215, bench->ctb_count); j++){
//            bench->mbr_search_in_disk(search_area, &tq, j);
//        }
//        cout << "before multi" << endl;
//        double prepare_consume = get_time_elapsed(prepare_start, true);
//
//        uint64_t total_MB = 0;
//        for(auto info : bench->box_search_queue){
//            total_MB += bench->ctbs[info.ctb_id].CTF_capacity[info.ctf_id];
//        }
//        cout << "total_MB" << total_MB * sizeof(__uint128_t) / 1024 / 1024 << endl;
//
//        pthread_t threads[1];
//        query_context tctx;
//        tctx.target[0] = (void *)bench;
//        tctx.target[1] = (void *)&search_area;
//        tctx.num_units = bench->box_search_queue.size();
//        tctx.report_gap = 1;
//        tctx.num_batchs = 1;
//        for(int j = 0; j < 1; j++){
//            pthread_create(&threads[j], NULL, box_search_unit, (void *)&tctx);
//        }
//        struct timeval multi_thread_start = get_cur_time();
//        for(int j = 0; j < 1; j++ ){
//            void *status;
//            pthread_join(threads[j], &status);
//        }
//        double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
//        q << edge_length*edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
//          << bench->intersect_sst_count <<',' << bench->bit_find_count << ',' << prepare_consume << endl;
//        bench->box_search_queue.clear();
//        bench->search_count = 0;
//    }
//    q.close();
//}
//
//void experiment_search_box(new_bench *bench){
//    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count / 20);
//    time_query tq;
//    tq.abandon = true;
//    ofstream q;
//    q.open("ex_search_box.csv", ios::out|ios::binary|ios::trunc);
//    q << "search area" << ',' << "find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ',' << "prepare_time(ms)" << endl;
//    for(int i = 0; i < 10; i++){
//        bench->clear_all_keys();
//        clear_cache();
//        struct timeval prepare_start = get_cur_time();
//        bench->search_count = 0;
//        bench->mbr_find_count = 0;
//        bench->intersect_sst_count = 0;
//        bench->bit_find_count = 0;
//        double edge_length = 0.01;
//        Point mid;
//        mid.x = -87.9 + 0.2;
//        mid.y = 41.6 + 0.2;
//        box search_area(mid.x - edge_length/2, mid.y - edge_length/2, mid.x + edge_length/2, mid.y + edge_length/2);
//        search_area.print();
//        for(uint j = 0; j < min((uint)1215, bench->ctb_count); j++){
//            bench->mbr_search_in_disk(search_area, &tq, j);
//        }
//        double prepare_consume = get_time_elapsed(prepare_start, true);
//        struct timeval multi_thread_start = get_cur_time();
//        pthread_t threads[bench->config->num_threads];
//        query_context tctx;
//        tctx.target[0] = (void *)bench;
//        tctx.target[1] = (void *)&search_area;
//        tctx.num_units = bench->box_search_queue.size();
//        tctx.report_gap = 1;
//        tctx.num_batchs = 10000;
//        for(int j = 0; j < bench->config->num_threads; j++){
//            pthread_create(&threads[j], NULL, box_search_unit, (void *)&tctx);
//        }
//        for(int j = 0; j < bench->config->num_threads; j++ ){
//            void *status;
//            pthread_join(threads[j], &status);
//        }
//        double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
//        q << edge_length*edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
//          << bench->intersect_sst_count <<',' << bench->bit_find_count << ',' << prepare_consume << endl;
//        bench->box_search_queue.clear();
//    }
//    q.close();
//}
//

void get_random_point(vector<Point> &vp){
    for(uint i = 0; i < vp.size(); i++){
        vp[i].x = -87.9 + 0.01 + (0.36 - 0.01) * get_rand_double();
        vp[i].y = 41.65 + 0.01 + (0.36 - 0.01) * get_rand_double();
    }
}

void experiment_box_openmp(workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    double edge_length = 0.001;
    vector<Point> vp(100);
    get_random_point(vp);
    for(uint selectivity = 1; selectivity <= 4; selectivity++) {
        ofstream q;
        q.open("ex_search_box_omp" + to_string(selectivity) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
            << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','  << "buffer_find" << endl;
        for (int i = 0; i < 100; i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            double margin = edge_length / 2;
            Point mid = vp[i];
            box search_area(mid.x - edge_length / 2, mid.y - edge_length / 2, mid.x + edge_length / 2,
                            mid.y + edge_length / 2);

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;
#pragma omp parallel for num_threads(bench->config->num_threads)
            for (uint j = 0; j < min((uint) 1215, bench->ctb_count); j++) {
                bench->mbr_search_in_obuffer(search_area, j, &tq);
            }
            double buffer_consume = get_time_elapsed(prepare_start, true);
            uint end_buffer = bench->search_count - before_buffer;

            uint64_t total_MB = 0;
            for (auto info: bench->box_search_queue) {
                total_MB += bench->ctbs[info.ctb_id].ctfs[info.ctf_id].CTF_kv_capacity * bench->ctbs[info.ctb_id].ctfs[info.ctf_id].key_bit / 8;
            }
            //cout << "total_MB" << total_MB * sizeof(__uint128_t) / 1024 / 1024 << endl;
            struct timeval multi_thread_start = get_cur_time();
#pragma omp parallel for num_threads(bench->config->num_threads)
            for (auto info: bench->box_search_queue) {
                CTF * ctf = &bench->ctbs[info.ctb_id].ctfs[info.ctf_id];
                uint mbr_find_count = 0;
                if (!ctf->keys) {
                    bench->load_CTF_keys(info.ctb_id, info.ctf_id);
                }
                for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                    __uint128_t temp_key = ctf->keys[q];
                    if (info.tq.abandon ) {
                        //uint pid = get_key_oid(temp_key);
                        box key_box = ctf->new_parse_mbr(temp_key);
                        if (search_area.intersect(key_box)) {
                            mbr_find_count++;
                            //cout<<"box find!"<<endl;
                            //key_box.print();
                        }
                    }
                }
                delete[] ctf->keys;
                ctf->keys = nullptr;
                bench->search_count.fetch_add(mbr_find_count, std::memory_order_relaxed);
            }

            double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
            q << edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << mid.x << ',' << mid.y << ',' << buffer_consume << ','  << end_buffer << endl;
            bench->box_search_queue.clear();
        }
        q.close();
        edge_length *= 3.16;
    }
}

void experiment_search_time(workbench *bench){
    time_query tq;
    tq.abandon = false;
    for(uint i = 36000; i < 3600 * 24; i+= 3600){
        tq.t_start = i;
        tq.t_end = i + 3600;
        bench->clear_all_keys();
        clear_cache();
        struct timeval disk_search_time = get_cur_time();
        uint find_count = bench->search_time_in_disk(&tq);
        double time_consume = get_time_elapsed(disk_search_time);
        cout << "find_count" << find_count << "time_consume" << time_consume << endl;
    }
}

//void query_search_id(workbench *bench){
////    uint question_count = 10000;
////    bench->wid_filter_count = 0;
////    bench->id_find_count = 0;
////    uint pid = 100000;
////    ofstream q;
////    q.open(to_string(bench->config->MemTable_capacity/2)+"search_id.csv", ios::out|ios::binary|ios::trunc);
////    q << "question number" << ',' << "time_consume(ms)" << endl;
////    for(int i = 0; i < question_count; i++){
////        struct timeval disk_search_time = get_cur_time();
////        bench->id_search_in_disk(pid, 15);
////        for(int j = 0; j < bench->ctb_count; j++){
////            bench->id_search_in_CTB(pid, j, );
////        }
////        pid++;
////        double time_consume = get_time_elapsed(disk_search_time);
////        //printf("disk_search_time %.2f\n", time_consume);
////        q << i << ',' << time_consume << endl;
////    }
////    q.close();
////    cout << "question_count:" << question_count << " id_find_count:" << bench->id_find_count <<" kv_restriction:"<< bench->config->kv_restriction << endl;
////    cout << "wid_filter_count:" << bench->wid_filter_count << "id_not_find_count" << bench->id_not_find_count << endl;
//
//}
//
//void query_search_box(workbench *bench){
////    //                double mid_x = -87.678503;
//////                double mid_y = 41.856803;
//////                Point the_mid(mid_x, mid_y);
//////                the_mid.print();
////    double mid_x[10] = {-87.678503, -87.81683, -87.80959,-87.81004, -87.68706,-87.68616,-87.67892, -87.63235, -87.61381, -87.58352};
////    double mid_y[10] = {41.856803, 41.97466, 41.90729, 41.76984, 41.97556, 41.89960, 41.74859, 41.87157, 41.78340, 41.70744};
////    double base_edge_length = 0.01;
//////        for(int i = 0; i < bench->ctb_count; i++){
//////            bench->load_big_sorted_run(i);
//////        }
////    ofstream p;
////    p.open(to_string(bench->config->MemTable_capacity/2)+"search_mbr.csv", ios::out|ios::binary|ios::trunc);        //config->SSTable_count/50
////    p << "search area" << ',' << "find_count" << ',' << "unique_find" << ',' << "intersect_sst_count" << ',' << "bit_find_count" << ',' << "time(ms)" << endl;
////    for(uint j = 0; j < 10; j++){
////        for(int i = 0; i < 10 ; i++){
////            //cout << fixed << setprecision(6) << mid_x - edge_length/2 <<","<<mid_y - edge_length/2 <<","<<mid_x + edge_length/2 <<","<<mid_y + edge_length/2 <<endl;
////            double edge_length = base_edge_length * (i + 1);
////            box search_area(mid_x[j] - edge_length/2, mid_y[j] - edge_length/2, mid_x[j] + edge_length/2, mid_y[j] + edge_length/2);
////            search_area.print();
////            struct timeval area_search_time = get_cur_time();
//////                    uint temp = bench->config->SSTable_count;
//////                    bench->config->SSTable_count = bench->merge_sstable_count;
////            bench->mbr_search_in_disk(search_area, 5);
//////                    bench->config->SSTable_count = temp;
////            double time_consume = get_time_elapsed(area_search_time);
////            //printf("area_search_time %.2f\n", time_consume);
////            p << edge_length*edge_length << ',' << bench->mbr_find_count << ',' << bench->mbr_unique_find << ','
////              << bench->intersect_sst_count <<',' << bench->bit_find_count << ',' << time_consume << endl;
////            bench->mbr_find_count = 0;
////            bench->mbr_unique_find = 0;
////            bench->intersect_sst_count = 0;
////            bench->bit_find_count = 0;
////        }
////        p << endl;
////    }
////    p.close();
//}
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
//    char new_raid[24] = "/data3/raid0_num";
//    memcpy(config->raid_path, new_raid, sizeof(config->raid_path));
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
////void ex_pg_search_box(new_bench *bench){
////    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count / 20);
////    time_query tq;
////    tq.abandon = true;
////    ofstream q;
////    q.open("ex_search_box.csv", ios::out|ios::binary|ios::trunc);
////    if(!q.is_open()){
////        log("%s cannot be opened","ex_search_box.csv");
////        exit(0);
////    }
////    q << "search area" << ',' << "find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ',' << "prepare_time(ms)" << endl;
////
////    bench->clear_all_keys();
////    clear_cache();
////    struct timeval prepare_start = get_cur_time();
////    bench->search_count = 0;
////    bench->mbr_find_count = 0;
////    bench->intersect_sst_count = 0;
////    bench->bit_find_count = 0;
////    double edge_length = 0.01;
////    Point mid;
////    mid.x = -87.9 + 0.2;
////    mid.y = 41.6 + 0.2;
////    box search_area(mid.x - edge_length/2, mid.y - edge_length/2, mid.x + edge_length/2, mid.y + edge_length/2);
////    search_area.print();
////    for(uint j = 0; j < min((uint)1215, bench->ctb_count); j++){
////        bench->mbr_search_in_disk(search_area, &tq, j);
////    }
////    cout << "before multi" << endl;
////    double prepare_consume = get_time_elapsed(prepare_start, true);
////    struct timeval multi_thread_start = get_cur_time();
////    pthread_t threads[1];
////    query_context tctx;
////    tctx.target[0] = (void *)bench;
////    tctx.target[1] = (void *)&search_area;
////    tctx.num_units = bench->box_search_queue.size();
////    tctx.report_gap = 1;
////    tctx.num_batchs = 1;
////    for(int j = 0; j < 1; j++){
////        pthread_create(&threads[j], NULL, box_search_unit, (void *)&tctx);
////    }
////    for(int j = 0; j < 1; j++ ){
////        void *status;
////        pthread_join(threads[j], &status);
////    }
////    double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
////    q << edge_length*edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
////      << bench->intersect_sst_count <<',' << bench->bit_find_count << ',' << prepare_consume << endl;
////    bench->box_search_queue.clear();
////    bench->search_count = 0;
////
////    q.close();
////}
//
//bool temp_mbr_search_in_CTB(new_bench * nb,box b, uint CTB_id){
//    uint i = CTB_id;
//    bool ret = false, find = false;
//    box bit_b = nb->make_bit_box(b);
//    uint bit_pos = 0;
//    if(!nb->ctbs[i].ctfs){
//        nb->ctbs[i].ctfs = new CTF[nb->config->CTF_count];
//    }
//    //cout << "in bg_run" << i << endl;
//
//    uint buffer_find = 0;
////    for(uint q = 0; q < ctbs[i].o_buffer.oversize_kv_count; q++){
////        if(ctbs[i].o_buffer.boxes[q].intersect(b)){
////            //uni.insert(cuda_get_key_oid(ctbs[i].o_buffer.keys[i]));
////            buffer_find++;
////            mbr_find_count++;
////            //cout<<"box find!"<<endl;
////            //ctbs[i].o_buffer.boxes[q].print();
////        }
////    }
//
//    for (uint j = 0; j < 100; j++) {
//        uint CTF_id = j;
//        find = true;
////        for (uint p = bit_b.low[0]-1; (p <= bit_b.high[0]+1) && (!find); p++) {
////            for (uint q = bit_b.low[1]-1; (q <= bit_b.high[1]+1) && (!find); q++) {
////                bit_pos = xy2d(SID_BIT / 2, p, q);
////                if (nb->ctbs[i].bitmaps[CTF_id * (nb->bit_count / 8) + bit_pos / 8] & (1 << (bit_pos % 8))) {              //mbr intersect bitmap
////                    //cerr << "SSTable_" << CTF_id << "bit_pos" << bit_pos << endl;
////                    find = true;
////                    ret = true;
////                    break;
////                }
////            }
////        }
//        if(find){
//            nb->bit_find_count++;
//            if(!nb->ctbs[i].ctfs[CTF_id].keys){
//                nb->load_CTF_keys(i, CTF_id);
//            }
//            uint this_find = 0;
//            for(uint q = 0; q < nb->ctbs[i].CTF_capacity[CTF_id]; q++){
//                uint pid = get_key_oid(nb->ctbs[i].ctfs[CTF_id].keys[q]);
//                box key_box;
//                parse_mbr(nb->ctbs[i].ctfs[CTF_id].keys[q], key_box, nb->ctbs[i].bitmap_mbrs[j]);
//                if(b.intersect(key_box)){
//                    //uni.insert(pid);
//                    this_find++;
//                    nb->mbr_find_count++;
//                    //cout<<"box find!"<<endl;
//                    //key_box.print();
//
//                }
//            }
//            //cerr<<this_find<<"finds in sst "<<CTF_id<<endl;
//
//        }
//    }
//    return ret;
//}
//
//
////int main(int argc, char **argv){
////    clear_cache();
////    string path = "../data/meta/";
////    workbench * bench = temp_load_meta(path.c_str());
////    new_bench * nb = new new_bench(bench->config);
////    memcpy(nb, bench, sizeof(workbench));
////    cout << nb->ctb_count << endl;
////
//////    double edge_length = 0.01;
//////    Point mid;
//////    mid.x = -87.9 + 0.2;
//////    mid.y = 41.6 + 0.2;
//////    box search_area(mid.x - edge_length/2, mid.y - edge_length/2, mid.x + edge_length/2, mid.y + edge_length/2);
//////    search_area.print();
//////    struct timeval start_time = get_cur_time();
//////    //nb->old_mbr_search_in_CTB(search_area, 10);
//////    temp_mbr_search_in_CTB(nb, search_area, 10);
//////    double time_consume = get_time_elapsed(start_time, true);
//////    cout << "ctb " << 10 << "time_consume" << time_consume << endl;
//////    cout << nb->bit_find_count << " " << nb->mbr_find_count << endl;
////
////    time_query tq;
////    tq.abandon = true;
////    nb->ctbs[10].ctfs = new CTF[100];
////    struct timeval start_time = get_cur_time();
////    for(uint i = 0; i < 10; i++){
////        nb->id_search_in_CTB(i, 10, &tq);
////    }
//////    nb->id_search_in_CTB(5149112, 10, &tq);
////    cout << "search_count " << nb->search_count << endl;
////    double time_consume = get_time_elapsed(start_time, true);
////    cout << "ctb " << 10 << "time_consume" << time_consume << endl;
////    return 0;
////}
//
//
//

int main(int argc, char **argv){
    clear_cache();
    string path = "../data/meta/";
    //workbench * bench = C_load_meta(path.c_str());
    uint max_ctb = 1215;
    workbench * bench = load_meta(path.c_str(), max_ctb);
    cout << "bench->ctb_count " << bench->ctb_count << endl;
    cout << "max_ctb " << max_ctb << endl;
    bench->ctb_count = max_ctb;

    for(int i = 0; i < bench->ctb_count; i++) {
        for (int j = 0; j < bench->config->CTF_count; j++) {
            bench->ctbs[i].ctfs[j].keys = nullptr;
        }
    }

    bench->build_trees(max_ctb);


//    for(uint i = 0; i < 20; i++){
//        cout << bench->ctbs[i].ctfs[4].start_time_min << '-' << bench->ctbs[i].ctfs[4].end_time_max << endl;
//    }
//    struct timeval start_time = get_cur_time();
//    bench->build_trees(max_ctb);
//    logt("build_trees ",start_time);
//
//    Interval query(55, 10000);
//    auto result = bench->total_btree->search(query);
//    cout << "Entries intersecting with ";
//    query.print();
//    cout << ":" << endl;
//    for (const auto& entry : result) {
//        entry.first.print();
//        cout << " -> " << entry.second << endl;
//    }
//    logt("btree test finish ",start_time);



//        for(int i = 0; i < bench->ctb_count; i++){
//            bench->load_big_sorted_run(i);
//        }
//        fprintf(stderr,"\ttotal load keys:\t%.2f\n", bench->pro.load_keys_time);


    //experiment_twice(nb);
    //exp4_search_oid_single(nb);
//    struct timeval start = get_cur_time();
    //experiment_search_oid(bench, max_ctb);
//    double real_world_time = get_time_elapsed(start, true);
//    cout << "real_world_time(s) " << real_world_time / 1000 << endl;
    //exp4_search_box_single(nb);
    //experiment_search_box(nb);
    //experiment_box_openmp(bench);
    experiment_search_time(bench);
    //query_search_id(bench);
    //query_search_box(bench);

    return 0;
}

//        cerr << "output picked o bitmap" << endl;
//        Point * bit_points = new Point[bench->bit_count];
//        uint count_p;
//        cerr<<endl;
//        count_p = 0;
//        for(uint i=0;i<bench->bit_count;i++){
//            if(bench->ctbs[0].o_buffer.o_bitmaps[i/8] & (1<<(i%8))){
//                Point bit_p;
//                uint x=0,y=0;
//                x = i % DEFAULT_bitmap_edge;
//                y = i / DEFAULT_bitmap_edge;
//                bit_p.x = (double)x/DEFAULT_bitmap_edge*(bench->mbr.high[0] - bench->mbr.low[0]) + bench->mbr.low[0];
//                bit_p.y = (double)y/DEFAULT_bitmap_edge*(bench->mbr.high[1] - bench->mbr.low[1]) + bench->mbr.low[1];
//                bit_points[count_p] = bit_p;
//                count_p++;
//            }
//        }
//        cout<<"bit_points.size():"<<count_p<<endl;
//        print_points(bit_points,count_p);
//        //cerr << "process output bitmap finish" << endl;
//        delete[] bit_points;
//
//    cout << "search begin" << endl;