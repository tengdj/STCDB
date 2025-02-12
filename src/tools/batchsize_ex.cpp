#include "../tracing/workbench.h"

using namespace std;

void batch_search_oid(workbench *bench, uint max_ctb){
    time_query tq;
    tq.abandon = true;
    ofstream q;
    q.open("ex_search_id" + to_string(bench->config->CTF_count) + ".csv", ios::out|ios::binary|ios::trunc);
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

void batch_search_box(double edge_length, vector<Point> vp, workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    for(uint selectivity = 1; selectivity <= 1; selectivity++) {
        ofstream q;
        //"bitmap_grid_search" + to_string(bench->bitmap_grid)
        //"ctf_num_search" + to_string(bench->config->CTF_count)
        q.open("batchsize_search" + to_string(bench->config->CTF_count) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
          << "total_MB" << ',' << "prepare_time(ms)" << ',' << "load_ctf_time" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
          << "buffer_find" << ',' << "buffer_bitmap_rate" << ',' << "time_contain_count" << endl;
        for (uint i = 0; i < min(100, (int)vp.size()); i++) {
            bench->clear_all_keys();
            clear_cache();
            struct timeval prepare_start = get_cur_time();
            bench->search_count = 0;
            bench->mbr_find_count = 0;
            bench->intersect_sst_count = 0;
            bench->bit_find_count = 0;
            bench->time_contain_count = 0;

            Point mid = vp[i];
            box search_area(mid.x - edge_length / 2, mid.y - edge_length / 2, mid.x + edge_length / 2,
                            mid.y + edge_length / 2);

            bench->mbr_search_in_disk(search_area, &tq);
            double prepare_consume = get_time_elapsed(prepare_start, true);

            //search_area.print();
            uint before_buffer = bench->search_count;
            float buffer_bitmap_hit = 0;
            for (uint j = 0; j < min((uint) 1215, bench->ctb_count); j++) {
                if( bench->mbr_search_in_obuffer(search_area, j, &tq) ){
                    buffer_bitmap_hit += 1;
                }
            }
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
                for (uint64_t q = 0; q < ctf->CTF_kv_capacity; q++) {
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
                // delete[] ctf->keys;
                // ctf->keys = nullptr;
                bench->search_count.fetch_add(mbr_find_count, std::memory_order_relaxed);
            }

            double multi_thread_consume = get_time_elapsed(multi_thread_start, true);
            q << edge_length << ',' << bench->search_count << ',' << multi_thread_consume << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << total_MB / 1024 / 1024 << ',' << prepare_consume << ',' << load_time << ',' << mid.x << ',' << mid.y << ',' << buffer_consume << ','
              << end_buffer << ',' << buffer_bitmap_hit / bench->ctb_count << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
        edge_length *= 2;       //3.162
    }
}

void batch_search_time(workbench *bench){
    ofstream q;
    q.open("search_time" + to_string(bench->config->CTF_count) + ".csv", ios::out|ios::binary|ios::trunc);
    q << "start_second" << ',' << "duration" << ',' << "find count" << ',' << "time_consume(ms)" << ',' << "total_index_find"
      << ',' << "contained count" << ',' << "traverse_keys count" << endl;

    time_query tq;
    tq.abandon = false;
    tq.t_start = 5000;
    for(uint i = 600; i <= 3600; i+= 600){
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

int main(int argc, char **argv){
    clear_cache();
    double edge_length = 0.000329512;             //0.000329512
    Point vp1[1000];
    ifstream IS_query_points("random_points1000", ios::in | ios::binary);
    if (!IS_query_points.is_open()) {
        std::cerr << "Error opening random_points1000" << std::endl;
    }
    IS_query_points.read((char *)&vp1, 1000*sizeof(Point));
    vector<Point> vp(vp1, vp1 + 1000);
    vector<uint> ctf_nums = {100, 196, 400, 784, 1600, 3136};
    uint GB = 2;
    uint max_ctb = 0;
    for(uint i = 0; i < 6; i++) {
        clear_cache();
        uint new_CTF_count = ctf_nums[i];
        string prefix = to_string(GB) + "G";
        string path = "../data/meta/" + prefix;
        max_ctb = 64 / GB;
        workbench * bench = load_meta(path.c_str(), max_ctb, new_CTF_count);
        strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);
        bench->ctb_count = max_ctb;
        cout << "bench->ctb_count " << bench->ctb_count << "bench->config->CTF_count " << bench->config->CTF_count << endl;
        for(int i = 0; i < max_ctb; i++) {
            for (int j = 0; j < bench->config->CTF_count; j++) {
                bench->ctbs[i].ctfs[j].keys = nullptr;
            }
        }

//        for(int i = 0; i < bench->ctb_count; i++) {
//            bench->load_CTF_keys(i, 0);
//            CTF *ctf = &bench->ctbs[i].ctfs[0];
//            uint8_t *data = reinterpret_cast<uint8_t *>(ctf->keys);
//            for (uint64_t q = 0; q < ctf->CTF_kv_capacity; q++) {
//                key_info temp_ki;
//                __uint128_t temp_128 = 0;
//                memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
//                uint64_t value_mbr = 0;
//                ctf->parse_key(temp_128, temp_ki, value_mbr);
//                cout << temp_ki.oid << " - " << temp_ki.target << endl;
//            }
//        }
        bench->build_trees(max_ctb);
        batch_search_oid(bench, max_ctb);
        batch_search_box(edge_length, vp, bench);
        batch_search_time(bench);
        GB *= 2;
    }
    return 0;
}

