#include "../tracing/workbench.h"


using namespace std;

// void experiment_search_oid(workbench *bench, uint max_ctb){
//     time_query tq;
//     tq.abandon = true;
//     ofstream q;
//     q.open("ex_search_id.csv", ios::out|ios::binary|ios::trunc);
//     q << "question number" << ',' << "prepare_consume(ms)" << ',' << "search_id_consume" << ',' << "find_id_count" << ',' << "sid_zero_count" << ','
//         << "hit_buffer" << ',' << "hit_ctf" << endl;
//     for(int i = 0; i < 1000; i++){
//         bench->clear_all_keys();
//         clear_cache();
//         bench->search_count = 0;
//         bench->hit_buffer = 0;
//         bench->hit_ctf = 0;
//         //bench->wid_filter_count = 0;
//         uint pid = get_rand_number(bench->config->num_objects);
//         bench->search_multi = false;
//         struct timeval prepare_start = get_cur_time();
//         vector<uint> search_list;
//         search_list.reserve(max_ctb);
//         for(uint j = 0; j < max_ctb; j++){
//             if(bench->ctbs->sids[pid] != 0){
//                 search_list.push_back(j);
//             }
//         }
//         double prepare_consume = get_time_elapsed(prepare_start, true);

//         for(uint j = 0; j < search_list.size(); j++){
//             bench->id_search_in_CTB(pid, search_list[j], &tq);
//         }
//         double search_id_consume = get_time_elapsed(prepare_start, true);
//         bench->search_multi = false;
//         q << pid << ',' << prepare_consume << ',' << search_id_consume << ',' << bench->search_count << ',' << max_ctb - bench->hit_buffer - bench->hit_ctf
//             << ',' << bench->hit_buffer << ',' << bench->hit_ctf << endl;
//     }
//     q.close();
// }

void ex_oversize_cut(uint k, double edge_length, vector<Point> vp, workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    for(uint selectivity = 1; selectivity <= 1; selectivity++) {
        ofstream q;
        q.open("ex_oversize_cut" + to_string(k) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
            << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
            << "buffer_find" << ',' << "time_contain_count" << endl;
        for (int i = 0; i < 1000; i++) {
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
              << end_buffer << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
        edge_length *= 2;       //3.162
    }
}

bool temp_mbr_search_in_CTB(workbench * bench, box b, uint CTB_id){
    uint i = CTB_id;
    bool ret = false, find = false;
    box bit_b = bench->make_bit_box(b);
    uint bit_pos = 0;
    if(!bench->ctbs[i].ctfs){
        bench->ctbs[i].ctfs = new CTF[bench->config->CTF_count];
    }

    uint buffer_find = 0;
    uint mbr_find_count = 0;
    for(uint q = 0; q < bench->ctbs[i].o_buffer.oversize_kv_count; q++){
        if(bench->ctbs[i].o_buffer.boxes[q].intersect(b)){
            buffer_find++;
            mbr_find_count++;
        }
    }

    for (uint j = 0; j < bench->config->CTF_count; j++) {
        uint CTF_id = j;
        find = false;
        CTF *ctf = &bench->ctbs[CTB_id].ctfs[CTF_id];
        if(ctf->ctf_mbr.intersect(b)){
            bench->intersect_sst_count++;
            for(uint bid=0; bid < ctf->ctf_bitmap_size * 8; bid++){
                if(ctf->bitmap[bid / 8] & (1 << (bid % 8))){
                    Point bit_p;
                    uint x=0,y=0;
                    x = bid % ctf->x_grid;
                    y = bid / ctf->x_grid;
                    bit_p.x = (double) x / ctf->x_grid * (ctf->ctf_mbr.high[0] - ctf->ctf_mbr.low[0]) +
                            ctf->ctf_mbr.low[0];
                    bit_p.y = (double) y / ctf->y_grid * (ctf->ctf_mbr.high[1] - ctf->ctf_mbr.low[1]) +
                            ctf->ctf_mbr.low[1];
                    if(b.contain(bit_p)){
                        find = true;
                        ret = true;
                        break;
                    }
                }
            }
        }
        if(find){
            bench->bit_find_count++;
            if(!ctf->keys){
                bench->load_CTF_keys(i, CTF_id);
            }
            uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
            for (uint q = 0; q < ctf->CTF_kv_capacity; q++) {
                key_info temp_ki;
                __uint128_t temp_128 = 0;
                memcpy(&temp_128, data + q * ctf->key_bit / 8, ctf->key_bit / 8);
                uint64_t value_mbr = 0;
                ctf->parse_key(temp_128, temp_ki, value_mbr);

                //uint pid = get_key_oid(temp_key);
                box key_box = ctf->new_parse_mbr(value_mbr);
                if (b.intersect(key_box)) {
                    mbr_find_count++;
                    //cout<<"box find!"<<endl;
                    //key_box.print();
                }
            }
        }
        bench->search_count.fetch_add(mbr_find_count, std::memory_order_relaxed);
    }
    return ret;
}

void old_search_one_CTB(double edge_length, vector<Point> vp, workbench *bench){
    time_query tq;
    tq.abandon = true;
    for(uint selectivity = 1; selectivity <= 1; selectivity++) {
        ofstream q;
        q.open("ex_oversize_cut_old.csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
            << "total_MB" << ',' << "prepare_time(ms)" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
            << "buffer_find" << ',' << "time_contain_count" << endl;
        for (int i = 0; i < 1000; i++) {
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

            
            temp_mbr_search_in_CTB(bench, search_area, 0);

            double prepare_consume = get_time_elapsed(prepare_start, true);
            q << edge_length << ',' << bench->search_count << ',' << 0 << ','
              << bench->intersect_sst_count << ',' << bench->bit_find_count << ','
              << 0 / 1024 / 1024 << ',' << prepare_consume << ',' << mid.x << ',' << mid.y << ',' << 0 << ','
              << 0 << ',' << bench->time_contain_count << endl;
        }
        q.close();
        edge_length *= 2;       //3.162
    }
}

// void experiment_search_time(workbench *bench){
//     uint last_search = 3600 * (24 * 7 - 5);
//     uint time_pick = last_search / 10;
//     ofstream q;
//     q.open("search_time.csv", ios::out|ios::binary|ios::trunc);
//     q << "start_second" << ',' << "duration" << ',' << "find count" << ',' << "time_consume(ms)" << ',' << "total_index_find"
//         << ',' << "contained count" << ',' << "traverse_keys count" << endl;
//     for(uint base_time = time_pick; base_time <= last_search; base_time += time_pick){
//         time_query tq;
//         tq.abandon = false;
//         tq.t_start = base_time;
//         for(uint i = 600; i <= 3600 * 3; i+= 600){
//             tq.t_end = tq.t_start + i;
//             bench->clear_all_keys();
//             clear_cache();
//             bench->search_count = 0;
//             bench->time_find_vector_size = 0;
//             bench->time_contain_count = 0;
//             struct timeval disk_search_time = get_cur_time();
//             uint find_count = bench->search_time_in_disk(&tq);
//             double time_consume = get_time_elapsed(disk_search_time);
//             cout << "find_count" << find_count << "time_consume" << time_consume << endl;
//             q << tq.t_start << ',' << i << ',' << find_count << ',' << time_consume << ',' << bench->time_find_vector_size
//                 << ',' << bench->time_contain_count << ',' << bench->time_find_vector_size - bench->time_contain_count << endl;
//         }
//     }
// }

int main(int argc, char **argv){
    clear_cache();
    double edge_length = 0.000329512;             //0.000329512
    vector<Point> vp(1000);
    for(uint i = 0; i < vp.size(); i++){
        vp[i].x = -87.9 + 0.01 + (0.36 - 0.01) * get_rand_double();
        vp[i].y = 41.65 + 0.01 + (0.36 - 0.01) * get_rand_double();
    }
    for(uint k = 0; k <=10; k++){
        string path = "../data/meta/6oversize_" + to_string(k);
        //workbench * bench = C_load_meta(path.c_str());
        uint max_ctb = 1;
        workbench * bench = load_meta(path.c_str(), max_ctb, 256);
        string prefix = "6oversize_" + to_string(k);
        strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);
        // for(uint i = 0;i < bench->config->CTF_count; i++){
        //     bench->ctbs[0].ctfs[i].print_bitmap();
        //     //bench->ctbs[0].ctfs[i].ctf_mbr.print(); 
        // }
        // return 0;
        
        cout << "bench->ctb_count " << bench->ctb_count << endl;
        cout << "max_ctb " << max_ctb << endl;
        bench->ctb_count = max_ctb;
        for(int i = 0; i < bench->ctb_count; i++) {
            for (int j = 0; j < bench->config->CTF_count; j++) {
                bench->ctbs[i].ctfs[j].keys = nullptr;
            }
        }
        bench->build_trees(max_ctb);    

        cout << bench->ctbs[0].o_buffer.oversize_kv_count << " bench->ctbs[0].o_buffer.oversize_kv_count" << endl;
        ex_oversize_cut(k, edge_length, vp, bench);
        //old_search_one_CTB(edge_length, vp, bench);       //error result
    }


    return 0;
}

