#include "../tracing/workbench.h"


using namespace std;

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
    q << "question number" << ',' << "prepare_consume(ms)" << ',' << "search_id_consume" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
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
        vector<uint> search_list;
        search_list.reserve(max_ctb);
        for(uint j = 0; j < max_ctb; j++){
            if(bench->ctbs->sids[pid] != 0){
                search_list.push_back(j);
            }
        }
        double prepare_consume = get_time_elapsed(prepare_start, true);
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < search_list.size(); j++){
            bench->id_search_in_CTB(pid, search_list[j], &tq);
        }
        double search_id_consume = get_time_elapsed(prepare_start, true);
        bench->search_multi = false;
        q << pid << ',' << prepare_consume << ',' << search_id_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
            << ',' << bench->hit_buffer << ',' << bench->hit_ctf << endl;
    }
    q.close();
}

void hybrid_oid(workbench *bench, uint max_ctb){
    time_query tq;
    tq.abandon = true;
    tq.t_start = 10000;
    tq.t_end = 13600;
    ofstream q;
    q.open("hybrid_oid.csv", ios::out|ios::binary|ios::trunc);
    q << "question number" << ',' << "prepare_consume(ms)" << ',' << "search_id_consume" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
      << "hit_buffer" << ',' << "hit_ctf" << endl;
    for(int i = 0; i < 100; i++){
        bench->clear_all_keys();
        clear_cache();
        bench->search_count = 0;
        bench->hit_buffer = 0;
        bench->hit_ctf = 0;
        //bench->wid_filter_count = 0;
        uint pid = get_rand_number(bench->config->num_objects);
        bench->search_multi = false;
        struct timeval prepare_start = get_cur_time();
        vector<uint> search_list;
        search_list.reserve(max_ctb);
        for(uint j = 0; j < max_ctb; j++){
            if(bench->ctbs->sids[pid] != 0){
                search_list.push_back(j);
            }
        }
        double prepare_consume = get_time_elapsed(prepare_start, true);
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < search_list.size(); j++){
            bench->id_search_in_CTB(pid, search_list[j], &tq);
        }
        double search_id_consume = get_time_elapsed(prepare_start, true);
        bench->search_multi = false;
        q << pid << ',' << prepare_consume << ',' << search_id_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
          << ',' << bench->hit_buffer << ',' << bench->hit_ctf << endl;
    }
    q.close();
}

void airport_oid(workbench *bench, uint max_ctb){
    time_query tq;
    tq.abandon = true;
    ofstream q;
    q.open("airport_oid.csv", ios::out|ios::binary|ios::trunc);
    q << "question number" << ',' << "prepare_consume(ms)" << ',' << "search_id_consume" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
      << "hit_buffer" << ',' << "hit_ctf" << endl;
    for(int i = 0; i < 100; i++){
        bench->clear_all_keys();
        clear_cache();
        bench->search_count = 0;
        bench->hit_buffer = 0;
        bench->hit_ctf = 0;
        //bench->wid_filter_count = 0;
        //uint pid = 604150;
        //uint pid = 5873134;
        uint pid = get_rand_number(bench->config->num_objects);
        bench->search_multi = false;
        struct timeval prepare_start = get_cur_time();
        vector<uint> search_list;
        search_list.reserve(max_ctb);
        for(uint j = 0; j < max_ctb; j++){
            if(bench->ctbs->sids[pid] != 0){
                search_list.push_back(j);
            }
        }


        double prepare_consume = get_time_elapsed(prepare_start, true);
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < search_list.size(); j++){
            bench->id_search_in_CTB(pid, search_list[j], &tq);
        }
        double search_id_consume = get_time_elapsed(prepare_start, true);
        bench->search_multi = false;
        q << pid << ',' << prepare_consume << ',' << search_id_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
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
        vp[i].x = -87.75 + 0.01 + (0.36 - 0.01) * get_rand_double();
        vp[i].y = 41.65 + 0.01 + (0.36 - 0.01) * get_rand_double();
    }
}

void experiment_box_openmp(workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    double edge_length = 0.104212;             //0.000104192
    vector<Point> vp(1000);
    get_random_point(vp);
    for(uint selectivity = 11; selectivity <= 11; selectivity++) {
        ofstream q;
        q.open("ex_search_box_omp" + to_string(selectivity) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
            << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
            << "buffer_find" << ',' << "buffer_hit_count" << ',' << "time_contain_count" << endl;
        for (int i = 0; i < 1000; i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            bench->time_contain_count = 0;

            vp[i].x = -87.8 + 0.01 + (0.08 - 0.01) * get_rand_double();
            vp[i].y = 41.82 + 0.01 + (0.08 - 0.01) * get_rand_double();
            Point mid = vp[i];
            box search_area(mid.x - edge_length / 2, mid.y - edge_length / 2, mid.x + edge_length / 2,
                            mid.y + edge_length / 2);

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;
            uint real_ctb_count = min((uint) 1215, bench->ctb_count);
            bool * is_buffer_hit = new bool[real_ctb_count];
#pragma omp parallel for num_threads(bench->config->num_threads)
            for (uint j = 0; j < real_ctb_count; j++) {
                is_buffer_hit[j] = bench->mbr_search_in_obuffer(search_area, j, &tq);
            }
            double buffer_consume = get_time_elapsed(prepare_start, true);
            uint end_buffer = bench->search_count - before_buffer;
            uint buffer_hit_count = 0;
            for(uint j = 0; j < real_ctb_count; j++){
                if(is_buffer_hit[j]){
                    buffer_hit_count++;
                }
            }

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
                uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
                for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                    key_info temp_ki;
                    __uint128_t temp_128 = 0;
                    memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
                    uint64_t value_mbr = 0;
                    ctf->parse_key(temp_128, temp_ki, value_mbr);

                    if (info.tq.abandon ) {
                        //uint pid = get_key_oid(temp_key);
                        box key_box = ctf->new_parse_mbr(value_mbr);
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
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << mid.x << ',' << mid.y << ',' << buffer_consume << ','
              << end_buffer << ',' << buffer_hit_count << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
        edge_length *= 2;       //3.162
    }
}

void chicago_u_openmp(workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    for(uint selectivity = 11; selectivity <= 11; selectivity++) {
        ofstream q;
        q.open("chicago_u.csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
          << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
                << "buffer_find" << ',' << "buffer_hit_count" << ',' << "time_contain_count" << endl;
        for (int i = 0; i < 1; i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            bench->time_contain_count = 0;

            //box search_area(-87.6818, 42.04871, -87.6688, 42.06275);  //西北
            //box search_area(-87.93905, 41.95142, -87.87725, 42.00961);  //奥黑尔
            box search_area(-87.60590, 41.78396, -87.59174, 41.79499);      //芝加哥大学
            search_area.print();
            bench->mbr.print();

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;
            uint real_ctb_count = min((uint) 1215, bench->ctb_count);
            bool * is_buffer_hit = new bool[real_ctb_count];
#pragma omp parallel for num_threads(bench->config->num_threads)
            for (uint j = 0; j < real_ctb_count; j++) {
                is_buffer_hit[j] = bench->mbr_search_in_obuffer(search_area, j, &tq);
            }
            double buffer_consume = get_time_elapsed(prepare_start, true);
            uint end_buffer = bench->search_count - before_buffer;
            uint buffer_hit_count = 0;
            for(uint j = 0; j < real_ctb_count; j++){
                if(is_buffer_hit[j]){
                    buffer_hit_count++;
                }
            }

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
                uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
                for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                    key_info temp_ki;
                    __uint128_t temp_128 = 0;
                    memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
                    uint64_t value_mbr = 0;
                    ctf->parse_key(temp_128, temp_ki, value_mbr);

                    if (info.tq.abandon ) {
                        //uint pid = get_key_oid(temp_key);
                        box key_box = ctf->new_parse_mbr(value_mbr);
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
            q << 0 << ',' << bench->search_count << ',' << multi_thread_consume << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << 0 << ',' << 0 << ',' << buffer_consume << ','
                    << end_buffer << ',' << buffer_hit_count << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
    }
}

void airport_analyze_openmp(workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    vector<uint> oids_in_airport(bench->config->num_objects, 0);
    for(uint selectivity = 11; selectivity <= 11; selectivity++) {
        ofstream q;
        q.open("airport_analyze.csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
          << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
          << "buffer_find" << ',' << "buffer_hit_count" << ',' << "time_contain_count" << endl;
        for (int i = 0; i < 1; i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            bench->time_contain_count = 0;

            //box search_area(-87.6818, 42.04871, -87.6688, 42.06275);  //西北
            box search_area(-87.93905, 41.95142, -87.87725, 42.00961);  //奥黑尔
            //box search_area(-87.60590, 41.78396, -87.59174, 41.79499);      //芝加哥大学
            search_area.print();
            bench->mbr.print();

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;
            uint real_ctb_count = min((uint) 1215, bench->ctb_count);
            bool * is_buffer_hit = new bool[real_ctb_count];
#pragma omp parallel for num_threads(bench->config->num_threads)
            for (uint j = 0; j < real_ctb_count; j++) {
                is_buffer_hit[j] = bench->mbr_search_in_obuffer(search_area, j, &tq);
            }
            double buffer_consume = get_time_elapsed(prepare_start, true);
            uint end_buffer = bench->search_count - before_buffer;
            uint buffer_hit_count = 0;
            for(uint j = 0; j < real_ctb_count; j++){
                if(is_buffer_hit[j]){
                    buffer_hit_count++;
                }
            }

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
                uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
                for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                    key_info temp_ki;
                    __uint128_t temp_128 = 0;
                    memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
                    uint64_t value_mbr = 0;
                    ctf->parse_key(temp_128, temp_ki, value_mbr);

                    if (info.tq.abandon ) {
                        //uint pid = get_key_oid(temp_key);
                        box key_box = ctf->new_parse_mbr(value_mbr);
                        if (search_area.intersect(key_box)) {
                            oids_in_airport[temp_ki.oid]++;
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
            q << 0 << ',' << bench->search_count << ',' << multi_thread_consume << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << 0 << ',' << 0 << ',' << buffer_consume << ','
              << end_buffer << ',' << buffer_hit_count << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
    }
    uint max_appear_count = 0;
    uint max_oid = 0;
    for(uint i = 0; i < oids_in_airport.size(); i++){
        if(max_appear_count < oids_in_airport[i]){
            max_appear_count = oids_in_airport[i];
            max_oid = i;
        }
    }
    cout << "max_appear_oid:" << max_oid << " count" << max_appear_count << endl;
}

void experiment_search_time(workbench *bench){
    uint last_search = 3600 * (24 * 7 - 5);
    uint time_pick = last_search / 10;
    ofstream q;
    q.open("search_time.csv", ios::out|ios::binary|ios::trunc);
    q << "start_second" << ',' << "duration" << ',' << "find count" << ',' << "time_consume(ms)" << ',' << "total_index_find"
        << ',' << "contained count" << ',' << "traverse_keys count" << endl;
    for(uint base_time = time_pick; base_time <= last_search; base_time += time_pick){
        time_query tq;
        tq.abandon = false;
        tq.t_start = base_time;
        for(uint i = 600; i <= 3600 * 3; i+= 600){
            tq.t_end = tq.t_start + i;
            bench->clear_all_keys();
            clear_cache();
            bench->search_count = 0;
            bench->time_find_vector_size = 0;
            bench->time_contain_count = 0;
            struct timeval disk_search_time = get_cur_time();
            uint find_count = bench->search_time_in_disk(&tq);
            double time_consume = get_time_elapsed(disk_search_time);
            cout << "find_count" << find_count << "time_consume" << time_consume << endl;
            q << tq.t_start << ',' << i << ',' << find_count << ',' << time_consume << ',' << bench->time_find_vector_size
                << ',' << bench->time_contain_count << ',' << bench->time_find_vector_size - bench->time_contain_count << endl;
        }
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



void compress_rate(workbench * bench){
    ofstream q;
    q.open("compress_rate.csv", ios::out|ios::binary|ios::trunc);
    q << "ctf" << ',' << "CTF_kv_capacity" << ',' << "code_load" << ',' << "parse_consume" << ',' << "full_load" << endl;
    struct timeval compress_time = get_cur_time();

    for(uint i = 0; i < bench->config->CTF_count; i++){
        compress_time = get_cur_time();
        bench->load_CTF_keys(0, i);
        double code_load = get_time_elapsed(compress_time, true);
        CTF * ctf = &bench->ctbs[0].ctfs[i];
        uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);

        uint * pid = new uint[ctf->CTF_kv_capacity];
        uint * target = new uint[ctf->CTF_kv_capacity];
        uint * duration = new uint[ctf->CTF_kv_capacity];
        uint * end = new uint[ctf->CTF_kv_capacity];
        f_box * b = new f_box[ctf->CTF_kv_capacity];
        compress_time = get_cur_time();

//#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < ctf->CTF_kv_capacity; j++){
            uint64_t value_mbr = 0;
            __uint128_t temp_128 = 0;
            memcpy(&temp_128, data + j * ctf->key_bit / 8, ctf->key_bit / 8);
            ctf->parse_key(ctf->keys[j], pid[j], target[j], duration[j], end[j], value_mbr);
            b[j] = ctf->new_parse_mbr_f_box(value_mbr);
        }
        double parse_consume = get_time_elapsed(compress_time, true);

        ofstream s;
        s.open("ctf0_" + to_string(i), ios::out|ios::binary|ios::trunc);
        s.write((char *)pid, ctf->CTF_kv_capacity * sizeof(uint));
        s.write((char *)target, ctf->CTF_kv_capacity * sizeof(uint));
        s.write((char *)duration, ctf->CTF_kv_capacity * sizeof(uint));
        s.write((char *)end, ctf->CTF_kv_capacity * sizeof(uint));
        s.write((char *)b, ctf->CTF_kv_capacity * sizeof(f_box));
        s.close();

        delete ctf->keys;
        clear_cache();

        ifstream in("ctf0_" + to_string(i), ios::in | ios::binary);
        if(!in.is_open()){
            cerr << " can't open " << endl;
            exit(0);
        }
        compress_time = get_cur_time();
        in.read((char *)pid, ctf->CTF_kv_capacity * sizeof(uint));
        in.read((char *)target, ctf->CTF_kv_capacity * sizeof(uint));
        in.read((char *)duration, ctf->CTF_kv_capacity * sizeof(uint));
        in.read((char *)end, ctf->CTF_kv_capacity * sizeof(uint));
        in.read((char *)b, ctf->CTF_kv_capacity * sizeof(f_box));
        in.close();
        double full_load = get_time_elapsed(compress_time, true);

        q << i << ',' << bench->ctbs[0].ctfs[i].CTF_kv_capacity << ',' << code_load << ',' << parse_consume << ',' << full_load << endl;
    }
}


void one_batch_box_search(workbench * bench, uint ctb_id){
    uint ctb_MB = 0;
    for(uint j = 0; j < bench->config->CTF_count; j++){
        CTF * ctf = &bench->ctbs[ctb_id].ctfs[j];
        ctb_MB += ctf->CTF_kv_capacity * ctf->key_bit / 8;
    }
    ctb_MB = ctb_MB / 1024 / 1024;
    cout << "ctb_MB" << ctb_MB << endl;         //1822
    struct timeval build_start = get_cur_time();
    bench->total_rtree = new RTree<int *, double, 2, double>();
    for(uint j = 0; j < bench->config->CTF_count; j++){
        f_box & m = bench->ctbs[ctb_id].ctfs[j].ctf_mbr;
        box b;
        b.low[0] = m.low[0];
        b.low[1] = m.low[1];
        b.high[0] = m.high[0];
        b.high[1] = m.high[1];
        bench->total_rtree->Insert(b.low, b.high, new int(ctb_id * bench->config->CTF_count +j));
    }
    double build_tree_time = get_time_elapsed(build_start, true);
    ofstream q;
    q.open("one_batch_box_search.csv", ios::out|ios::binary|ios::trunc);
    q << "search edge_length" << ',' << "total_find_count" << ',' << "total_time_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
      << "total_MB" << ',' << "prepare_time(ms)" << ',' << "build_tree_time" << ',' << "load_time" << ',' << "buffer_time" << ','  << "buffer_find" << endl;


    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    double edge_length = 0.000104192;             //0.000329512
    for(uint selectivity = 1; selectivity <= 7; selectivity++) {
        for (int i = 0; i < 10; i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            Point mid = {-87.717895, 41.856374};
            box search_area(mid.x - edge_length / 2, mid.y - edge_length / 2, mid.x + edge_length / 2,
                            mid.y + edge_length / 2);

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;

            bench->mbr_search_in_obuffer(search_area, ctb_id, &tq);

            double buffer_consume = get_time_elapsed(prepare_start, true);
            uint end_buffer = bench->search_count - before_buffer;

            uint64_t total_MB = 0;
            for (auto info: bench->box_search_queue) {
                total_MB += bench->ctbs[info.ctb_id].ctfs[info.ctf_id].CTF_kv_capacity * bench->ctbs[info.ctb_id].ctfs[info.ctf_id].key_bit / 8;
            }
            //cout << "total_MB" << total_MB * sizeof(__uint128_t) / 1024 / 1024 << endl;
            struct timeval multi_thread_start = get_cur_time();

            double load_time = 0;
            for (auto info: bench->box_search_queue) {
                CTF * ctf = &bench->ctbs[info.ctb_id].ctfs[info.ctf_id];
                uint mbr_find_count = 0;
                struct timeval load_start = get_cur_time();
                if (!ctf->keys) {
                    bench->load_CTF_keys(info.ctb_id, info.ctf_id);
                }
                load_time += get_time_elapsed(load_start, true);
                uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
                for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                    key_info temp_ki;
                    __uint128_t temp_128 = 0;
                    memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
                    uint64_t value_mbr = 0;
                    ctf->parse_key(temp_128, temp_ki, value_mbr);

                    if (info.tq.abandon ) {
                        //uint pid = get_key_oid(temp_key);
                        box key_box = ctf->new_parse_mbr(value_mbr);
                        if (search_area.intersect(key_box)) {
                            mbr_find_count++;
                            //cout<<"box find!"<<endl;
                            //key_box.print();
                        }
                    }
                }
//                delete[] ctf->keys;
//                ctf->keys = nullptr;
                bench->search_count.fetch_add(mbr_find_count, std::memory_order_relaxed);
            }

            double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
            q << edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << build_tree_time << ',' << load_time << ',' << buffer_consume << ','  << end_buffer << endl;
            bench->box_search_queue.clear();
        }
        edge_length *= 3.162;       //3.162
    }
    q.close();

};

void one_batch_oid_search(workbench *bench, uint max_ctb){
    time_query tq;
    tq.abandon = true;
    ofstream q;
    q.open("one_batch_oid_search.csv", ios::out|ios::binary|ios::trunc);
    q << "question number" << ',' << "prepare_consume(ms)" << ',' << "search_id_consume" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
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
        vector<uint> search_list;
        search_list.reserve(max_ctb);
        for(uint j = 0; j < max_ctb; j++){
            if(bench->ctbs->sids[pid] != 0){
                search_list.push_back(j);
            }
        }
        double prepare_consume = get_time_elapsed(prepare_start, true);
//#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < search_list.size(); j++){
            bench->id_search_in_CTB(pid, search_list[j], &tq);
        }
        double search_id_consume = get_time_elapsed(prepare_start, true);
        bench->search_multi = false;
        q << pid << ',' << prepare_consume << ',' << search_id_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
          << ',' << bench->hit_buffer << ',' << bench->hit_ctf << endl;
    }
    q.close();
}

int main(int argc, char **argv){
    clear_cache();
    string path = "../data/meta/N";
    //workbench * bench = C_load_meta(path.c_str());
    uint max_ctb = 174;
    workbench * bench = load_meta(path.c_str(), max_ctb);
    cout << "bench->ctb_count " << bench->ctb_count << endl;
    cout << "max_ctb " << max_ctb << endl;
    bench->ctb_count = max_ctb;
    for(int i = 0; i < bench->ctb_count; i++) {
        for (int j = 0; j < bench->config->CTF_count; j++) {
            bench->ctbs[i].ctfs[j].keys = nullptr;
        }
    }

//    uint64_t all_meetings_count = 0;
//    for(uint i = 0; i < max_ctb; i++){
//        for(uint j = 0; j < bench->config->CTF_count; j++){
//            all_meetings_count += bench->ctbs[i].ctfs[j].CTF_kv_capacity;
//        }
//    }
//    cout << "all_meetings_count " << all_meetings_count << " GB " << all_meetings_count / 1024 / 1024/ 1024 * 16 << endl;
//    return 0;

//    for(int i = 0; i < bench->ctb_count; i++) {
//        bench->load_CTF_keys(i, 0);
//        CTF *ctf = &bench->ctbs[i].ctfs[0];
//        uint8_t *data = reinterpret_cast<uint8_t *>(ctf->keys);
//        for (uint64_t q = 0; q < ctf->CTF_kv_capacity; q++) {
//            key_info temp_ki;
//            __uint128_t temp_128 = 0;
//            memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
//            uint64_t value_mbr = 0;
//            ctf->parse_key(temp_128, temp_ki, value_mbr);
//            cout << temp_ki.oid << " temp_ki.oid " << endl;
//        }
//    }
//    one_batch_oid_search(bench, 1);
//    one_batch_box_search(bench, 10);
//    return 0;


    //compress_rate(bench);

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
//    experiment_search_oid(bench, max_ctb);
//    double real_world_time = get_time_elapsed(start, true);
//    cout << "real_world_time(s) " << real_world_time / 1000 << endl;
    //exp4_search_box_single(nb);
    //experiment_search_box(nb);
    //experiment_box_openmp(bench);
    //experiment_search_time(bench);
    //query_search_id(bench);
    //query_search_box(bench);

    //chicago_u_openmp(bench);
    //hybrid_oid(bench, max_ctb);
    //airport_analyze_openmp(bench);
    airport_oid(bench, max_ctb);

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