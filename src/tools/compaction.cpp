#include "../tracing/workbench.h"
#include <thread>
#include <mutex>

using namespace std;

bool cmp_ki(pair<key_info, f_box> a, pair<key_info, f_box> b){
    if(a.first.oid == b.first.oid){
        return a.first.target < b.first.target;
    }
    return a.first.oid < b.first.oid;
}

void cpu_sort_by_key(key_info* keys, f_box* values, uint n) {
    vector<pair<key_info, f_box>> key_value_pairs(n);

    // Pack keys and values
#pragma omp parallel for num_threads(128)
    for (size_t i = 0; i < n; ++i) {
        key_value_pairs[i] = {keys[i], values[i]};
    }

    // Sort by keys
    sort(key_value_pairs.begin(), key_value_pairs.end(), cmp_ki);

    // Unpack sorted keys and values
    for (size_t i = 0; i < n; ++i) {
        keys[i] = key_value_pairs[i].first;
        values[i] = key_value_pairs[i].second;
    }
}

void cpu_sort_by_key(vector<key_info> & keys, vector<f_box> & values) {
    uint n = keys.size();
    vector<pair<key_info, f_box>> key_value_pairs(n);

    // Pack keys and values
#pragma omp parallel for num_threads(128)
    for (size_t i = 0; i < n; ++i) {
        key_value_pairs[i] = {keys[i], values[i]};
    }

    // Sort by keys
    sort(key_value_pairs.begin(), key_value_pairs.end(), cmp_ki);

    // Unpack sorted keys and values
    for (size_t i = 0; i < n; ++i) {
        keys[i] = key_value_pairs[i].first;
        values[i] = key_value_pairs[i].second;
    }
}

// Compare function to sort by sid and ave_loc.x
bool compare_two_level(const object_info& a, const object_info& b, const unsigned short* sids) {
    bool a_is_last = (sids[a.oid] <= 1); // Check if a should be placed last
    bool b_is_last = (sids[b.oid] <= 1); // Check if b should be placed last

    // Partition logic: place elements with sids <= 1 at the end
    if (a_is_last != b_is_last) {           // Partition
        return b_is_last; // If a should be placed first and b should be placed last, return true
    }

    // Sorting logic: for elements with sids > 1, sort by ave_loc.x in ascending order
    if (!a_is_last) { // Only elements with sids > 1 need to be sorted
        return a.ave_loc.x < b.ave_loc.x;
    }

    // For elements with sids <= 1, maintain the original order (stable sort)
    return false;
}

bool compare_y(const object_info& a, const object_info& b) {
    return a.ave_loc.y < b.ave_loc.y;
}


void ctf_num_search(double edge_length, vector<Point> vp, workbench *bench){
    bench->box_search_queue.reserve(bench->ctb_count * bench->config->CTF_count);
    time_query tq;
    tq.abandon = true;
    for(uint selectivity = 1; selectivity <= 1; selectivity++) {
        ofstream q;
        //"bitmap_grid_search" + to_string(bench->bitmap_grid)
        //"ctf_num_search" + to_string(bench->config->CTF_count)
        q.open("bitmap_grid_search" + to_string(bench->bitmap_grid) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "search edge_length" << ',' << "total_find_count" << ',' << "multi_thread_consume" << ',' << "intersect_mbr_count" << ',' << "bitmap_find_count" << ','
          << "total_MB" << ',' << "prepare_time(ms)" << ',' << "load_ctf_time" << ',' << "mid x" << ',' << "mid y" << ',' << "buffer_time" << ','
          << "buffer_find" << ',' << "buffer_bitmap_hit" << ',' << "time_contain_count" << endl;
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
            short buffer_bitmap_hit = 0;
            for (uint j = 0; j < min((uint) 1215, bench->ctb_count); j++) {
                buffer_bitmap_hit = bench->mbr_search_in_obuffer(search_area, j, &tq);
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
              << end_buffer << ',' << buffer_bitmap_hit << ',' << bench->time_contain_count << endl;
            bench->box_search_queue.clear();
        }
        q.close();
        edge_length *= 2;       //3.162
    }
}

//6G ctf_size test
//void * ex_ctf_size(double edge_length, vector<Point> vp, workbench * bench, uint start_ctb, uint merge_ctb_count,
//                   vector< vector<key_info> > &keys_with_sid, vector< vector<f_box> > &mbrs_with_sid, vector< vector<uint> > &invert_index,
//                   vector< pair< vector<key_info>, vector<f_box> > > &objects_map, vector<object_info> &str_list ){
//
//    cout<<"step into the sst_dump"<<endl;
//    CTB C_ctb;
//    struct timeval bg_start = get_cur_time();
//    C_ctb.sids = new unsigned short[bench->config->num_objects];
//    memset(C_ctb.sids, 0, sizeof(unsigned short) * bench->config->num_objects);
//    //copy(bench->ctbs[start_ctb].sids, bench->ctbs[start_ctb].sids + bench->config->num_objects, C_ctb.sids);
//    for(int i = 0; i < merge_ctb_count; i++){
//#pragma omp parallel for num_threads(bench->config->num_threads)
//        for(int j = 0; j < bench->config->num_objects; j++){
//            if(bench->ctbs[start_ctb + i].sids[j] == 1){
//                C_ctb.sids[j] = 1;
//            }
//        }
//    }
//    double init_sids_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t init_sids_time:\t%.2f\n",init_sids_time);
//
////find all oversize sid
////uint sid = x_index * bench->config->split_num + y_index + 2
////    for(int i = 1; i < merge_ctb_count; i++){
////        for(int j = 0; j < bench->config->num_objects; j++){
////            if(bench->ctbs[start_ctb + i].sids[j] == 0){
////                continue;
////            }
////            if(bench->ctbs[start_ctb + i].sids[j] == 1 || C_ctb.sids[j] == 1){
////                C_ctb.sids[j] = 1;
////            }
////            else{
////                uint new_ctf = bench->ctbs[start_ctb + i].sids[j] - 2;
////                uint new_x = new_ctf / bench->config->split_num;
////                uint new_y = new_ctf % bench->config->split_num;
////                uint old_ctf = bench->ctbs[start_ctb + i].sids[j] - 2;
////                uint old_x = old_ctf / bench->config->split_num;
////                uint old_y = old_ctf % bench->config->split_num;
////                C_ctb.sids[j] = (old_x + new_x) / 2 * bench->config->split_num + (old_y + new_y) / 2 + 2;
////            }
////        }
////    }
//
//    key_info * another_o_buffer_k = new key_info[1024*1024*1024/16];        //1G
//    f_box * another_o_buffer_b = new f_box[1024*1024*1024/16];
//    atomic<int> another_o_count = 0;
//
//    uint new_CTF_count = bench->config->split_num * bench->config->split_num;
//    C_ctb.ctfs = new CTF[new_CTF_count];
//    uint old_CTF_count = bench->config->CTF_count;
//    //bench->config->CTF_count = new_CTF_count;
//    bench->merge_kv_capacity = bench->config->kv_restriction * merge_ctb_count;         //less than that
//
//    double load_keys_time = 0;
//    double invert_index_time = 0;
//    struct timeval load_start;
//    for(int i = 0; i < merge_ctb_count; i++){
//        load_start = get_cur_time();
//#pragma omp parallel for num_threads(old_CTF_count)
//        for (int j = 0; j < old_CTF_count; j++) {
//            bench->load_CTF_keys(start_ctb + i, j);
//        }
//        load_keys_time += get_time_elapsed(load_start,true);
//
//        bg_start = get_cur_time();
//#pragma omp parallel for num_threads(old_CTF_count)
//        for (int j = 0; j < old_CTF_count; j++) {
//            CTF * ctf = &bench->ctbs[start_ctb + i].ctfs[j];
//            uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
//            for (int k = 0; k < bench->ctbs[start_ctb + i].ctfs[j].CTF_kv_capacity; k++) {
//                key_info temp_ki;
//                __uint128_t temp_128 = 0;
//                memcpy(&temp_128, data + k * ctf->key_bit / 8, ctf->key_bit / 8);
//                uint64_t value_mbr = 0;
//                ctf->parse_key(temp_128, temp_ki, value_mbr);
//                temp_ki.end += ctf->end_time_min;                                           //real end
//                f_box temp_mbr = ctf->new_parse_mbr_f_box(value_mbr);
//                if (bench->ctbs[start_ctb + i].sids[temp_ki.oid] == 1) {                    //get all oversized keys
//                    int pos = another_o_count.fetch_add(1, memory_order_seq_cst);
//                    another_o_buffer_k[pos] = temp_ki;
//                    another_o_buffer_b[pos] = temp_mbr;
//                } else {
//                    //str_list[temp_ki.oid].oid = temp_ki.oid;
//                    C_ctb.sids[temp_ki.oid] = 2;
//                    str_list[temp_ki.oid].object_mbr.update(temp_mbr);
//                    //str_list[temp_ki.oid].key_count++;                  //atomic ??
//                    objects_map[temp_ki.oid].first.push_back(temp_ki);
//                    objects_map[temp_ki.oid].second.push_back(temp_mbr);
//                }
//            }
//            delete []ctf->keys;
//            ctf->keys = nullptr;
//        }
//        invert_index_time += get_time_elapsed(bg_start,true);
//    }
//    fprintf(stdout,"\t load_keys_time:\t%.2f\n",load_keys_time);
//    fprintf(stdout,"\t invert_index:\t%.2f\n",invert_index_time);
//
//#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0; i < bench->config->num_objects; i++){
//        if(!objects_map.empty()){
//            cpu_sort_by_key(objects_map[i].first, objects_map[i].second);
//        }
//    }
//    double target_sort_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t target_sort_time:\t%.2f\n",target_sort_time);
//
////cpu str
//#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0 ; i < bench->config->num_objects; i++){
//        str_list[i].oid = i;
//        if(C_ctb.sids[i] == 2){
//            str_list[i].ave_loc.x = str_list[i].object_mbr.low[0] / 2 + str_list[i].object_mbr.high[0] / 2;
//            str_list[i].ave_loc.y = str_list[i].object_mbr.low[1]/ 2 + str_list[i].object_mbr.high[1] / 2;
//        }
//
//    }
//    double init_str_list = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tinit_str_list:\t%.2f\n",init_str_list);
//
//
//    // uint zero_count = 0, one_count = 0;
//    // for(uint i = 0; i < bench->config->num_objects; i++){
//    //     if(C_ctb.sids[i] == 0) zero_count++;
//    //     else if(C_ctb.sids[i] == 1) one_count++;
//    // }
//    // cout << "zero_count = " << zero_count << endl;
//    // cout << "one_count = " << one_count << endl;
//
//
//    auto * csids = C_ctb.sids;
//    std::sort(std::execution::par, str_list.begin(), str_list.end(), [&csids](const object_info& a, const object_info& b) {
//        return compare_two_level(a, b, csids);
//    });
//    double part_and_xsort = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t part_and_xsort:\t%.2f\n",part_and_xsort);
//
//    uint split_index = 0;
//    for(uint i = str_list.size() - 1; i >= 0; i--){
//        if(C_ctb.sids[str_list[i].oid] > 1){
//            split_index = i;
//            break;
//        }
//    }
//    cout << "split_index = " << split_index << endl;
//    double get_split_index = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t get_split_index:\t%.2f\n",get_split_index);
//
//    // for(uint i = 0; i < 10; i++){
//    //     cout << "str_list[" << i << "].x = " << str_list[i].ave_loc.x << endl;
//    // }
//    // for(uint i = split_index - 5; i < split_index + 5; i++){
//    //     cout << "str_list[" << i << "].sid = " << C_ctb.sids[ str_list[i].oid ] << endl;
//    // }
//
//
//    uint x_part_capacity = split_index / bench->config->split_num + 1;
//    //std::sort(str_list.begin(), str_list.end(), compare_y);       //single thread
//    //#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0; i < bench->config->split_num - 1; i++){
//        sort(std::execution::par, str_list.begin() + x_part_capacity * i,
//             str_list.begin() + x_part_capacity * (i + 1),
//             compare_y);
//    }
//    sort(std::execution::par, str_list.begin() + x_part_capacity * (bench->config->split_num - 1),
//         str_list.begin() + split_index,
//         compare_y);
//    double y_sort_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t y_sort_time:\t%.2f\n",y_sort_time);
//
//    uint y_part_capacity = x_part_capacity / bench->config->split_num + 1;
//    cout << "x_part_capacity = " << x_part_capacity << endl;
//    cout << "y_part_capacity = " << y_part_capacity << endl;
//
//    for(uint i = 0; i < new_CTF_count; i++){
//        invert_index[i].resize(y_part_capacity);
//    }
//    cout << "resize write" << endl;
//
//#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0; i < split_index; i++){
//        uint x_index = i / x_part_capacity;
//        uint y_index = (i % x_part_capacity) / y_part_capacity;
//        //uint y_index = (i - x_part_capacity * x_index) / y_part_capacity;
//        uint temp_sid = x_index * bench->config->split_num + y_index + 2;           //should check bitmap
//        C_ctb.sids[ str_list[i].oid ] = temp_sid;
//        uint temp_oid_index = i - x_index * x_part_capacity - y_index * y_part_capacity;
//        //if(i % 1000 == 0) cout << temp_oid_index << endl;
//        invert_index[temp_sid - 2][temp_oid_index] = str_list[i].oid;
//    }
//    double write_sid_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t write_sid_time:\t%.2f\n",write_sid_time);
//
//#pragma omp parallel for num_threads(new_CTF_count)
//    for (int j = 0; j < new_CTF_count; j++) {
//        uint v_capacity = 0;
//        for (int k = 0; k < invert_index[j].size(); k++) {
//            v_capacity += objects_map[invert_index[j][k]].first.size();
//        }
//        C_ctb.ctfs[j].CTF_kv_capacity = v_capacity;
//        keys_with_sid[j].reserve(v_capacity);
//        mbrs_with_sid[j].reserve(v_capacity);
//        for (int k = 0; k < invert_index[j].size(); k++) {
//            copy(objects_map[invert_index[j][k]].first.begin(), objects_map[invert_index[j][k]].first.end(),
//                 back_inserter(keys_with_sid[j]) );
//            copy(objects_map[invert_index[j][k]].second.begin(), objects_map[invert_index[j][k]].second.end(),
//                 back_inserter(mbrs_with_sid[j]) );
////            keys_with_wid[j].insert(keys_with_wid[j].end(),
////                                    objects_map[invert_index[j][k]].first.begin(), objects_map[invert_index[j][k]].first.end());
////            mbrs_with_wid[j].insert(mbrs_with_wid[j].end(),
////                                    objects_map[invert_index[j][k]].second.begin(), objects_map[invert_index[j][k]].second.end());
//        }
//    }
//    double expand_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\texpand_time:\t%.2f\n",expand_time);
//
//    double STR_time = init_sids_time + invert_index_time + target_sort_time + init_str_list
//                      + part_and_xsort + get_split_index + y_sort_time + write_sid_time + expand_time;
//
//    uint rest_total_count = 0;
//    for(uint i = 0; i < new_CTF_count; i++){
//        rest_total_count += C_ctb.ctfs[i].CTF_kv_capacity;
//    }
//    cout << "rest_total_count" << rest_total_count << endl;
//
//    //get limit and bits
//    C_ctb.end_time_min = 1 << 30;
//    C_ctb.end_time_max = 0;
//    for(int i = 0; i < merge_ctb_count; i++){
//        C_ctb.end_time_min = min(C_ctb.end_time_min, bench->ctbs[start_ctb + i].end_time_min);
//        C_ctb.end_time_max = max(C_ctb.end_time_max, bench->ctbs[start_ctb + i].end_time_max);
//    }
//#pragma omp parallel for num_threads(new_CTF_count)
//    for(uint i = 0; i < new_CTF_count; i++){
//        float local_low[2] = {100000.0,100000.0};
//        float local_high[2] = {-100000.0,-100000.0};
//        uint local_start_time_min = 1 << 30;
//        uint local_start_time_max = 0;
//        for(uint j = 0; j < C_ctb.ctfs[i].CTF_kv_capacity; j++){
//            local_low[0] = min(local_low[0], mbrs_with_sid[i][j].low[0]);
//            local_low[1] = min(local_low[1], mbrs_with_sid[i][j].low[1]);
//            local_high[0] = max(local_high[0], mbrs_with_sid[i][j].high[0]);
//            local_high[1] = max(local_high[1], mbrs_with_sid[i][j].high[1]);
//
//            uint this_start = keys_with_sid[i][j].end - keys_with_sid[i][j].duration;
//            local_start_time_min = min(local_start_time_min, this_start);
//            local_start_time_max = max(local_start_time_max, this_start);
//        }
//        C_ctb.ctfs[i].ctf_mbr.low[0] = local_low[0];
//        C_ctb.ctfs[i].ctf_mbr.low[1] = local_low[1];
//        C_ctb.ctfs[i].ctf_mbr.high[0] = local_high[0];
//        C_ctb.ctfs[i].ctf_mbr.high[1] = local_high[1];
//        C_ctb.ctfs[i].start_time_min = local_start_time_min;
//        C_ctb.ctfs[i].start_time_max = local_start_time_max;
//        C_ctb.ctfs[i].end_time_min = C_ctb.end_time_min;
//        C_ctb.ctfs[i].end_time_max = C_ctb.end_time_max;
//        C_ctb.ctfs[i].get_ctf_bits(bench->mbr, bench->config, bench->bitmap_grid);
//        //C_ctb.ctfs[i].get_ctf_bits(bench->mbr, bench->config);
//    }
//    double get_limit = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tget_limit:\t%.2f\n",get_limit);
//
//    // for(uint i = 0; i < new_CTF_count; i++){
//    //     cout << "ctf_bitmap_size" << C_ctb.ctfs[i].ctf_bitmap_size << endl;
//    // }
//
//    // {
//    //     CTF * ctf = &C_ctb.ctfs[0];
//    //     for(int j = 0; j < ctf->CTF_kv_capacity; j+=10000) {
//    //         mbrs_with_sid[0][j].print();
//    //     }
//    // }
//
//
//#pragma omp parallel for num_threads(new_CTF_count)
//    for(uint i = 0; i < new_CTF_count; i++){
//        CTF * ctf = &C_ctb.ctfs[i];
//        ctf->bitmap = new unsigned char[ctf->ctf_bitmap_size];
//        memset(ctf->bitmap, 0, ctf->ctf_bitmap_size);
//        for(int j = 0; j < ctf->CTF_kv_capacity; j++) {
////            f_box key_box = mbrs_with_sid[i][j];
////            if(key_box.low[0] > key_box.high[0] || key_box.low[1] > key_box.high[1]){
////                cout << " error box " << endl;
////            }
//            uint low0 = (mbrs_with_sid[i][j].low[0] - ctf->ctf_mbr.low[0])/(ctf->ctf_mbr.high[0] - ctf->ctf_mbr.low[0]) * ctf->x_grid;
//            uint low1 = (mbrs_with_sid[i][j].low[1] - ctf->ctf_mbr.low[1])/(ctf->ctf_mbr.high[1] - ctf->ctf_mbr.low[1]) * ctf->y_grid;
//            uint high0 = (mbrs_with_sid[i][j].high[0] - ctf->ctf_mbr.low[0])/(ctf->ctf_mbr.high[0] - ctf->ctf_mbr.low[0]) * ctf->x_grid;
//            uint high1 = (mbrs_with_sid[i][j].high[1] - ctf->ctf_mbr.low[1])/(ctf->ctf_mbr.high[1] - ctf->ctf_mbr.low[1]) * ctf->y_grid;
//            uint bit_pos = 0;
//            for (uint m = low0; m <= high0 && m < ctf->x_grid; m++) {
//                for (uint n = low1; n <= high1 && n < ctf->y_grid; n++) {
//                    bit_pos = m + n * ctf->x_grid;
//                    if(bit_pos > ctf->ctf_bitmap_size * 8){
//                        cout << "bit_pos "  << bit_pos << " m-n " << m << '-' << n << " x_grid " << ctf->x_grid << "y_grid" << ctf->y_grid << endl;
//                    }
//                    ctf->bitmap[bit_pos / 8] |= (1 << (bit_pos % 8));
//                }
//            }
//        }
//    }
//    double write_bitmap_time = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\twrite_bitmap_time:\t%.2f\n",write_bitmap_time);
//
////    cerr << new_CTF_count << "-bitmap print" << endl;
////    for(uint i = 0; i < new_CTF_count; i++) {
////        C_ctb.ctfs[i].print_bitmap();
////    }
//
//    for(uint i = 0; i < new_CTF_count; i++) {
//        CTF * ctf = &C_ctb.ctfs[i];
//        uint key_Bytes = ctf->key_bit / 8;
//        ctf->keys = new __uint128_t[ctf->CTF_kv_capacity * key_Bytes / sizeof(__uint128_t) + 1];
//        uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
//#pragma omp parallel for num_threads(bench->config->num_threads)
//        for (int j = 0; j < C_ctb.ctfs[i].CTF_kv_capacity; j++) {
//            __uint128_t value_mbr = serialize_mbr(&mbrs_with_sid[i][j], &ctf->ctf_mbr, ctf);
//            keys_with_sid[i][j].end += C_ctb.ctfs[i].end_time_min;
//            // key =  oid target duration end box
//            __uint128_t temp_key =
//                    ((__uint128_t) keys_with_sid[i][j].oid << (ctf->id_bit + ctf->duration_bit + ctf->end_bit + ctf->mbr_bit)) +
//                    ((__uint128_t) keys_with_sid[i][j].target << (ctf->duration_bit + ctf->end_bit + ctf->mbr_bit)) +
//                    ((__uint128_t) keys_with_sid[i][j].duration << (ctf->end_bit + ctf->mbr_bit)) +
//                    ((__uint128_t) keys_with_sid[i][j].end << (ctf->mbr_bit)) + value_mbr;
//            //uint ave_ctf_size = bench->config->kv_restriction / bench->config->CTF_count * sizeof(__uint128_t);
//            memcpy(&data[j * key_Bytes], &temp_key, key_Bytes);
//        }
//    }
//    double key_info_to_new_key = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tkey_info_to_new_key:\t%.2f\n",key_info_to_new_key);
//
//
//    cpu_sort_by_key(another_o_buffer_k, another_o_buffer_b, another_o_count);
//    double new_buffer_sort = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tnew_buffer_sort:\t%.2f\n",new_buffer_sort);
//
//    uint o_count = merge_ctb_count + 1;
//    oversize_buffer * ob = new oversize_buffer[o_count];
//    for(uint i = 0; i < merge_ctb_count; i++){
//        ob[i] = bench->ctbs[start_ctb + i].o_buffer;
//    }
//    ob[o_count - 1].oversize_kv_count = another_o_count;
//    ob[o_count - 1].keys = new __uint128_t[another_o_count];
//    ob[o_count - 1].boxes = another_o_buffer_b;
//#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0; i < another_o_count; i++){
//        ob[o_count - 1].keys[i] = ((__uint128_t)another_o_buffer_k[i].oid << (OID_BIT + DURATION_BIT + END_BIT)) +
//                                  ((__uint128_t)another_o_buffer_k[i].target << (DURATION_BIT + END_BIT)) +
//                                  ((__uint128_t)another_o_buffer_k[i].duration << END_BIT) +
//                                  (__uint128_t)(another_o_buffer_k[i].end);
//        //ob[o_count - 1].write_o_buffer(ob[o_count - 1].boxes[i], bench->bit_count);       //useless
//    }
//    double key_info_to_buffer_key = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tkey_info_to_buffer_key:\t%.2f\n",key_info_to_buffer_key);
//
//    oversize_buffer merged_buffer;
//    merged_buffer.oversize_kv_count += another_o_count;
//    merged_buffer.o_bitmaps = new uint8_t[bench->bitmaps_size];
//    memset(merged_buffer.o_bitmaps, 0, sizeof(uint8_t) * bench->bitmaps_size);
//
//    for(int i = 0; i < merge_ctb_count; i++){
//        merged_buffer.oversize_kv_count += ob[i].oversize_kv_count;
//        merged_buffer.start_time_min = min(merged_buffer.start_time_min, ob[i].start_time_min);
//        merged_buffer.start_time_max = max(merged_buffer.start_time_max, ob[i].start_time_max);
//        merged_buffer.end_time_min = min(merged_buffer.end_time_min, ob[i].end_time_min);
//        merged_buffer.end_time_max = max(merged_buffer.end_time_max, ob[i].end_time_max);
//#pragma omp parallel for num_threads(bench->config->num_threads)
//        for(uint j = 0; j < bench->bitmaps_size; j++) {
//            merged_buffer.o_bitmaps[j] |= ob[i].o_bitmaps[j];
//        }
//    }
//    double merged_buffer_bitmap = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\tmerged_buffer_bitmap:\t%.2f\n",merged_buffer_bitmap);
//
//    merged_buffer.keys = new __uint128_t[merged_buffer.oversize_kv_count];
//    merged_buffer.boxes = new f_box[merged_buffer.oversize_kv_count];
//    uint *key_index = new uint[o_count]{0};        //0
//    uint taken_id = 0;
//    uint kv_count = 0;
//    while(kv_count < merged_buffer.oversize_kv_count) {
//        __uint128_t temp_key = (__uint128_t) 1 << 126;
//        taken_id = 0;
//        for (int i = 0; i < o_count; i++) {
//            if (key_index[i] < ob[i].oversize_kv_count && temp_key > ob[i].keys[key_index[i]]){
//                taken_id = i;
//                temp_key = ob[taken_id].keys[key_index[taken_id]];
//            }
//        }
//        merged_buffer.keys[kv_count] = temp_key;
//        merged_buffer.boxes[kv_count] = ob[taken_id].boxes[key_index[taken_id]];
//        key_index[taken_id]++;
//        kv_count++;
//    }
//    double ordered_Multiway_Merge = get_time_elapsed(bg_start,true);
//    fprintf(stdout,"\t ordered_Multiway_Merge:\t%.2f\n",ordered_Multiway_Merge);
//    C_ctb.o_buffer = merged_buffer;
//    double encoding_time = get_limit + write_bitmap_time + write_bitmap_time + key_info_to_new_key
//                           + new_buffer_sort + key_info_to_buffer_key + merged_buffer_bitmap + ordered_Multiway_Merge;
//
//    //different
//#pragma omp parallel for num_threads(bench->config->num_threads)
//    for(uint i = 0; i < new_CTF_count; i++){
//        string sst_path = bench->config->raid_path + to_string(i%2) + "/C_SSTable_"+to_string(0)+"-"+to_string(i);
//        C_ctb.ctfs[i].dump_keys(sst_path.c_str());
//    }
//    vector<f_box> ctf_mbrs(new_CTF_count);
//    float area_sum = 0;
//    uint bitmap_kb = 0;
//    for(uint i = 0; i < new_CTF_count; i++){
//        bitmap_kb += C_ctb.ctfs[i].ctf_bitmap_size;
//        ctf_mbrs[i] = C_ctb.ctfs[i].ctf_mbr;
//        area_sum += C_ctb.ctfs[i].ctf_mbr.area();
//    }
//    bitmap_kb /= 1024;
//    cout << "new_CTF_count " << new_CTF_count << " bitmap_grid " << bench->bitmap_grid << " bitmap_kb " << bitmap_kb << endl;
//    float overlap_area = calculate_total_area(ctf_mbrs);
//    cout << "overlap_area " << overlap_area << " area_sum " << area_sum << "overlap rate " << overlap_area / area_sum << endl;
//
//    string prefix = "C";
//    strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);
//    CTB * old_ctbs = bench->ctbs;
//    bench->ctbs = &C_ctb;
//    uint old_ctb_count = bench->ctb_count;
//    bench->ctb_count = 1;
//    bench->config->CTF_count = new_CTF_count;
//    bench->build_trees(1);
//    ctf_num_search(edge_length, vp, bench);
//    bench->config->CTF_count = old_CTF_count;           //back trace
//    bench->ctb_count = old_ctb_count;
//    bench->ctbs = old_ctbs;
//    prefix = "N";
//    strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);
//    return NULL;
//}


//for compaction

void * merge_dump(workbench * bench, uint start_ctb, uint merge_ctb_count, vector< vector<key_info> > &keys_with_sid, vector< vector<f_box> > &mbrs_with_sid,
                  vector< vector<uint> > &invert_index, vector< pair< vector<key_info>, vector<f_box> > > &objects_map, vector<object_info> &str_list){
    ofstream q;
    q.open("compaction" + to_string(merge_ctb_count) + ".csv", ios::out|ios::binary|ios::trunc);
    q << "GB" << ',' << "STR(ms)" << ',' << "encoding" << ',' << "load_keys" << ',' << "dump_all" << endl;

    uint c_ctb_id = start_ctb / merge_ctb_count;
    cout<<"step into the sst_dump"<<endl;
    CTB C_ctb;
    struct timeval bg_start = get_cur_time();
    C_ctb.sids = new unsigned short[bench->config->num_objects];
    memset(C_ctb.sids, 0, sizeof(unsigned short) * bench->config->num_objects);
    //copy(bench->ctbs[start_ctb].sids, bench->ctbs[start_ctb].sids + bench->config->num_objects, C_ctb.sids);
    for(int i = 0; i < merge_ctb_count; i++){
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(int j = 0; j < bench->config->num_objects; j++){
            if(bench->ctbs[start_ctb + i].sids[j] == 1){
                C_ctb.sids[j] = 1;
            }
        }
    }
    double init_sids_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t init_sids_time:\t%.2f\n",init_sids_time);

    //find all oversize sid
    //uint sid = x_index * bench->config->split_num + y_index + 2
    //    for(int i = 1; i < merge_ctb_count; i++){
    //        for(int j = 0; j < bench->config->num_objects; j++){
    //            if(bench->ctbs[start_ctb + i].sids[j] == 0){
    //                continue;
    //            }
    //            if(bench->ctbs[start_ctb + i].sids[j] == 1 || C_ctb.sids[j] == 1){
    //                C_ctb.sids[j] = 1;
    //            }
    //            else{
    //                uint new_ctf = bench->ctbs[start_ctb + i].sids[j] - 2;
    //                uint new_x = new_ctf / bench->config->split_num;
    //                uint new_y = new_ctf % bench->config->split_num;
    //                uint old_ctf = bench->ctbs[start_ctb + i].sids[j] - 2;
    //                uint old_x = old_ctf / bench->config->split_num;
    //                uint old_y = old_ctf % bench->config->split_num;
    //                C_ctb.sids[j] = (old_x + new_x) / 2 * bench->config->split_num + (old_y + new_y) / 2 + 2;
    //            }
    //        }
    //    }

    key_info * another_o_buffer_k = new key_info[1024*1024*1024/16];        //1G
    f_box * another_o_buffer_b = new f_box[1024*1024*1024/16];
    atomic<int> another_o_count = 0;

    uint new_CTF_count = bench->config->split_num * bench->config->split_num;
    C_ctb.ctfs = new CTF[new_CTF_count];
    uint old_CTF_count = bench->config->CTF_count;
    //bench->config->CTF_count = new_CTF_count;
    //bench->merge_kv_capacity = bench->config->kv_restriction * merge_ctb_count;         //less than that  //useless

    double load_keys_time = 0;
    double invert_index_time = 0;
    struct timeval load_start;
    for(int i = 0; i < merge_ctb_count; i++){
        load_start = get_cur_time();
#pragma omp parallel for num_threads(old_CTF_count)
        for (int j = 0; j < old_CTF_count; j++) {
            bench->load_CTF_keys(start_ctb + i, j);
        }
        load_keys_time += get_time_elapsed(load_start,true);

        bg_start = get_cur_time();
#pragma omp parallel for num_threads(old_CTF_count)
        for (int j = 0; j < old_CTF_count; j++) {
            CTF * ctf = &bench->ctbs[start_ctb + i].ctfs[j];
            uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
            for (int k = 0; k < ctf->CTF_kv_capacity; k++) {
                key_info temp_ki;
                __uint128_t temp_128 = 0;
                memcpy(&temp_128, data + k * ctf->key_bit / 8, ctf->key_bit / 8);
                uint64_t value_mbr = 0;
                ctf->parse_key(temp_128, temp_ki, value_mbr);
                temp_ki.end += ctf->end_time_min;                                           //real end
                f_box temp_mbr = ctf->new_parse_mbr_f_box(value_mbr);
                if (bench->ctbs[start_ctb + i].sids[temp_ki.oid] == 1) {                    //get all oversized keys
                    int pos = another_o_count.fetch_add(1, memory_order_seq_cst);
                    another_o_buffer_k[pos] = temp_ki;
                    another_o_buffer_b[pos] = temp_mbr;
                } else {
                    //str_list[temp_ki.oid].oid = temp_ki.oid;
                    C_ctb.sids[temp_ki.oid] = 2;
                    str_list[temp_ki.oid].object_mbr.update(temp_mbr);
                    //str_list[temp_ki.oid].key_count++;                  //atomic ??
                    objects_map[temp_ki.oid].first.push_back(temp_ki);
                    objects_map[temp_ki.oid].second.push_back(temp_mbr);
                }
            }
            delete []ctf->keys;
            ctf->keys = nullptr;
        }
        invert_index_time += get_time_elapsed(bg_start,true);
    }
    fprintf(stdout,"\t load_keys_time:\t%.2f\n",load_keys_time);
    fprintf(stdout,"\t invert_index:\t%.2f\n",invert_index_time);

#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0; i < bench->config->num_objects; i++){
        if(!objects_map[i].first.empty()){
            cpu_sort_by_key(objects_map[i].first, objects_map[i].second);
        }
    }
    double target_sort_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t target_sort_time:\t%.2f\n",target_sort_time);

    //cpu str
#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0 ; i < bench->config->num_objects; i++){
        str_list[i].oid = i;
        if(C_ctb.sids[i] == 2){
            str_list[i].ave_loc.x = str_list[i].object_mbr.low[0] / 2 + str_list[i].object_mbr.high[0] / 2;
            str_list[i].ave_loc.y = str_list[i].object_mbr.low[1] / 2 + str_list[i].object_mbr.high[1] / 2;
        }
    }
    double init_str_list = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tinit_str_list:\t%.2f\n",init_str_list);


    // uint zero_count = 0, one_count = 0;
    // for(uint i = 0; i < bench->config->num_objects; i++){
    //     if(C_ctb.sids[i] == 0) zero_count++;
    //     else if(C_ctb.sids[i] == 1) one_count++;
    // }
    // cout << "zero_count = " << zero_count << endl;
    // cout << "one_count = " << one_count << endl;


    auto * csids = C_ctb.sids;
    std::sort(std::execution::par, str_list.begin(), str_list.end(), [&csids](const object_info& a, const object_info& b) {
        return compare_two_level(a, b, csids);
    });
    double part_and_xsort = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t part_and_xsort:\t%.2f\n",part_and_xsort);

    uint split_index = 0;
    for(uint i = str_list.size() - 1; i >= 0; i--){
        if(C_ctb.sids[str_list[i].oid] > 1){
            split_index = i;
            break;
        }
    }
    cout << "split_index = " << split_index << endl;
    double get_split_index = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t get_split_index:\t%.2f\n",get_split_index);

    // for(uint i = 0; i < 10; i++){
    //     cout << "str_list[" << i << "].x = " << str_list[i].ave_loc.x << endl;
    // }
    // for(uint i = split_index - 5; i < split_index + 5; i++){
    //     cout << "str_list[" << i << "].sid = " << C_ctb.sids[ str_list[i].oid ] << endl;
    // }


    uint x_part_capacity = split_index / bench->config->split_num + 1;
    //std::sort(str_list.begin(), str_list.end(), compare_y);       //single thread
    //#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0; i < bench->config->split_num - 1; i++){
        sort(std::execution::par, str_list.begin() + x_part_capacity * i,
             str_list.begin() + x_part_capacity * (i + 1),
             compare_y);
    }
    sort(std::execution::par, str_list.begin() + x_part_capacity * (bench->config->split_num - 1),
         str_list.begin() + split_index,
         compare_y);
    double y_sort_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t y_sort_time:\t%.2f\n",y_sort_time);

    uint y_part_capacity = x_part_capacity / bench->config->split_num + 1;
    cout << "x_part_capacity = " << x_part_capacity << endl;
    cout << "y_part_capacity = " << y_part_capacity << endl;

    for(uint i = 0; i < new_CTF_count; i++){
        invert_index[i].resize(y_part_capacity);
    }
    invert_index[new_CTF_count - 1].resize(split_index - x_part_capacity * (bench->config->split_num - 1) - y_part_capacity * (bench->config->split_num - 1));
    cout << "resize write" << endl;

#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0; i < split_index; i++){
        uint x_index = i / x_part_capacity;
        uint y_index = (i % x_part_capacity) / y_part_capacity;
        //uint y_index = (i - x_part_capacity * x_index) / y_part_capacity;
        uint temp_sid = x_index * bench->config->split_num + y_index + 2;           //should check bitmap
        C_ctb.sids[ str_list[i].oid ] = temp_sid;
        uint temp_oid_index = i - x_index * x_part_capacity - y_index * y_part_capacity;
        //if(i % 1000 == 0) cout << temp_oid_index << endl;
        invert_index[temp_sid - 2][temp_oid_index] = str_list[i].oid;
    }
    double write_sid_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t write_sid_time:\t%.2f\n",write_sid_time);

#pragma omp parallel for num_threads(new_CTF_count)
    for (int j = 0; j < new_CTF_count; j++) {
        std::sort(invert_index[j].begin(), invert_index[j].end());      //same sid, sort oid
        uint v_capacity = 0;
        for (int k = 0; k < invert_index[j].size(); k++) {
            v_capacity += objects_map[invert_index[j][k]].first.size();
        }
        C_ctb.ctfs[j].CTF_kv_capacity = v_capacity;
        keys_with_sid[j].reserve(v_capacity);
        mbrs_with_sid[j].reserve(v_capacity);
        for (int k = 0; k < invert_index[j].size(); k++) {
            copy(objects_map[invert_index[j][k]].first.begin(), objects_map[invert_index[j][k]].first.end(),
                 back_inserter(keys_with_sid[j]) );
            copy(objects_map[invert_index[j][k]].second.begin(), objects_map[invert_index[j][k]].second.end(),
                 back_inserter(mbrs_with_sid[j]) );
            //            keys_with_wid[j].insert(keys_with_wid[j].end(),
            //                                    objects_map[invert_index[j][k]].first.begin(), objects_map[invert_index[j][k]].first.end());
            //            mbrs_with_wid[j].insert(mbrs_with_wid[j].end(),
            //                                    objects_map[invert_index[j][k]].second.begin(), objects_map[invert_index[j][k]].second.end());
        }
    }
    double expand_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\texpand_time:\t%.2f\n",expand_time);
    for(uint i = 90; i < new_CTF_count; i++){
        cout << "CTF[" << i << "] kv_capacity = " << C_ctb.ctfs[i].CTF_kv_capacity << endl;
    }
    for(uint i = 90; i < new_CTF_count; i++){
        cout << "keys_with_sid.size" << keys_with_sid[i].size() << endl;
    }

    double STR_time = init_sids_time + invert_index_time + target_sort_time + init_str_list
                      + part_and_xsort + get_split_index + y_sort_time + write_sid_time + expand_time;


    uint rest_total_count = 0;
    for(uint i = 0; i < new_CTF_count; i++){
        rest_total_count += C_ctb.ctfs[i].CTF_kv_capacity;
    }
    cout << "rest_total_count" << rest_total_count << endl;

    //get limit and bits
    C_ctb.start_time_min = 1 << 30;
    C_ctb.start_time_max = 0;
    C_ctb.end_time_min = 1 << 30;
    C_ctb.end_time_max = 0;
    for(int i = 0; i < merge_ctb_count; i++){
        C_ctb.end_time_min = min(C_ctb.end_time_min, bench->ctbs[start_ctb + i].end_time_min);
        C_ctb.end_time_max = max(C_ctb.end_time_max, bench->ctbs[start_ctb + i].end_time_max);
        C_ctb.start_time_min = min(C_ctb.start_time_min, bench->ctbs[start_ctb + i].start_time_min);
        C_ctb.start_time_max = max(C_ctb.start_time_max, bench->ctbs[start_ctb + i].start_time_max);
    }
#pragma omp parallel for num_threads(new_CTF_count)
    for(uint i = 0; i < new_CTF_count; i++){
        float local_low[2] = {100000.0,100000.0};
        float local_high[2] = {-100000.0,-100000.0};
        uint local_start_time_min = 1 << 30;
        uint local_start_time_max = 1;
        for(uint j = 0; j < C_ctb.ctfs[i].CTF_kv_capacity; j++){
            local_low[0] = min(local_low[0], mbrs_with_sid[i][j].low[0]);
            local_low[1] = min(local_low[1], mbrs_with_sid[i][j].low[1]);
            local_high[0] = max(local_high[0], mbrs_with_sid[i][j].high[0]);
            local_high[1] = max(local_high[1], mbrs_with_sid[i][j].high[1]);

            uint this_start = keys_with_sid[i][j].end - keys_with_sid[i][j].duration;
            if(keys_with_sid[i][j].end < keys_with_sid[i][j].duration){
                cout << " end offset error " << keys_with_sid[i][j].end << '-' << keys_with_sid[i][j].duration << endl;
            }
            local_start_time_min = min(local_start_time_min, this_start);
            local_start_time_max = max(local_start_time_max, this_start);
        }
        C_ctb.ctfs[i].ctf_mbr.low[0] = local_low[0];
        C_ctb.ctfs[i].ctf_mbr.low[1] = local_low[1];
        C_ctb.ctfs[i].ctf_mbr.high[0] = local_high[0];
        C_ctb.ctfs[i].ctf_mbr.high[1] = local_high[1];
        C_ctb.ctfs[i].start_time_min = local_start_time_min;
        C_ctb.ctfs[i].start_time_max = local_start_time_max;
        C_ctb.ctfs[i].end_time_min = C_ctb.end_time_min;
        C_ctb.ctfs[i].end_time_max = C_ctb.end_time_max;
        C_ctb.ctfs[i].get_ctf_bits(bench->mbr, bench->config);
    }
    double get_limit = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tget_limit:\t%.2f\n",get_limit);

    // {
    //     CTF * ctf = &C_ctb.ctfs[0];
    //     for(int j = 0; j < ctf->CTF_kv_capacity; j+=10000) {
    //         mbrs_with_sid[0][j].print();
    //     }
    // }

#pragma omp parallel for num_threads(new_CTF_count)
    for(uint i = 0; i < new_CTF_count; i++){
        CTF * ctf = &C_ctb.ctfs[i];
        ctf->bitmap = new unsigned char[ctf->ctf_bitmap_size];
        memset(ctf->bitmap, 0, ctf->ctf_bitmap_size);
        for(int j = 0; j < ctf->CTF_kv_capacity; j++) {
            //            f_box key_box = mbrs_with_sid[i][j];
            //            if(key_box.low[0] > key_box.high[0] || key_box.low[1] > key_box.high[1]){
            //                cout << " error box " << endl;
            //            }
            uint low0 = (mbrs_with_sid[i][j].low[0] - ctf->ctf_mbr.low[0])/(ctf->ctf_mbr.high[0] - ctf->ctf_mbr.low[0]) * ctf->x_grid;
            uint low1 = (mbrs_with_sid[i][j].low[1] - ctf->ctf_mbr.low[1])/(ctf->ctf_mbr.high[1] - ctf->ctf_mbr.low[1]) * ctf->y_grid;
            uint high0 = (mbrs_with_sid[i][j].high[0] - ctf->ctf_mbr.low[0])/(ctf->ctf_mbr.high[0] - ctf->ctf_mbr.low[0]) * ctf->x_grid;
            uint high1 = (mbrs_with_sid[i][j].high[1] - ctf->ctf_mbr.low[1])/(ctf->ctf_mbr.high[1] - ctf->ctf_mbr.low[1]) * ctf->y_grid;
            uint bit_pos = 0;
            for (uint m = low0; m <= high0 && m < ctf->x_grid; m++) {
                for (uint n = low1; n <= high1 && n < ctf->y_grid; n++) {
                    bit_pos = m + n * ctf->x_grid;
                    if(bit_pos > ctf->ctf_bitmap_size * 8){
                        cout << "bit_pos "  << bit_pos << " m-n " << m << '-' << n << " x_grid " << ctf->x_grid << "y_grid" << ctf->y_grid << endl;
                    }
                    ctf->bitmap[bit_pos / 8] |= (1 << (bit_pos % 8));
                }
            }
        }
    }
    double write_bitmap_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\twrite_bitmap_time:\t%.2f\n",write_bitmap_time);

//    cerr << new_CTF_count << "-bitmap print" << endl;
    // for(uint i = 0; i < new_CTF_count; i++) {
    //     C_ctb.ctfs[i].print_bitmap();
    // }

    for(uint i = 0; i < new_CTF_count; i++) {
        CTF * ctf = &C_ctb.ctfs[i];
        uint key_Bytes = ctf->key_bit / 8;
        ctf->keys = new __uint128_t[ctf->CTF_kv_capacity * key_Bytes / sizeof(__uint128_t) + 1];
        uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
#pragma omp parallel for num_threads(bench->config->num_threads)
        for (int j = 0; j < C_ctb.ctfs[i].CTF_kv_capacity; j++) {
            __uint128_t value_mbr = serialize_mbr(&mbrs_with_sid[i][j], &ctf->ctf_mbr, ctf);
            keys_with_sid[i][j].end += C_ctb.ctfs[i].end_time_min;
            // key =  oid target duration end box
            __uint128_t temp_key =
                    ((__uint128_t) keys_with_sid[i][j].oid << (ctf->id_bit + ctf->duration_bit + ctf->end_bit + ctf->mbr_bit)) +
                    ((__uint128_t) keys_with_sid[i][j].target << (ctf->duration_bit + ctf->end_bit + ctf->mbr_bit)) +
                    ((__uint128_t) keys_with_sid[i][j].duration << (ctf->end_bit + ctf->mbr_bit)) +
                    ((__uint128_t) keys_with_sid[i][j].end << (ctf->mbr_bit)) + value_mbr;
            //uint ave_ctf_size = bench->config->kv_restriction / bench->config->CTF_count * sizeof(__uint128_t);
            memcpy(&data[j * key_Bytes], &temp_key, key_Bytes);
        }
    }
    double key_info_to_new_key = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tkey_info_to_new_key:\t%.2f\n",key_info_to_new_key);


    cpu_sort_by_key(another_o_buffer_k, another_o_buffer_b, another_o_count);
    double new_buffer_sort = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tnew_buffer_sort:\t%.2f\n",new_buffer_sort);

    uint o_count = merge_ctb_count + 1;
    oversize_buffer * ob = new oversize_buffer[o_count];
    for(uint i = 0; i < merge_ctb_count; i++){
        ob[i] = bench->ctbs[start_ctb + i].o_buffer;
    }
    ob[o_count - 1].oversize_kv_count = another_o_count;
    ob[o_count - 1].keys = new __uint128_t[another_o_count];
    ob[o_count - 1].boxes = another_o_buffer_b;
#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0; i < another_o_count; i++){
        ob[o_count - 1].keys[i] = ((__uint128_t)another_o_buffer_k[i].oid << (OID_BIT + DURATION_BIT + END_BIT)) +
                                  ((__uint128_t)another_o_buffer_k[i].target << (DURATION_BIT + END_BIT)) +
                                  ((__uint128_t)another_o_buffer_k[i].duration << END_BIT) +
                                  (__uint128_t)(another_o_buffer_k[i].end);
        //ob[o_count - 1].write_o_buffer(ob[o_count - 1].boxes[i], bench->bit_count);       //useless
    }
    double key_info_to_buffer_key = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tkey_info_to_buffer_key:\t%.2f\n",key_info_to_buffer_key);

    oversize_buffer merged_buffer;
    merged_buffer.oversize_kv_count += another_o_count;
    merged_buffer.o_bitmaps = new uint8_t[bench->bitmaps_size];
    memset(merged_buffer.o_bitmaps, 0, sizeof(uint8_t) * bench->bitmaps_size);

    for(int i = 0; i < merge_ctb_count; i++){
        merged_buffer.oversize_kv_count += ob[i].oversize_kv_count;
        merged_buffer.start_time_min = min(merged_buffer.start_time_min, ob[i].start_time_min);
        merged_buffer.start_time_max = max(merged_buffer.start_time_max, ob[i].start_time_max);
        merged_buffer.end_time_min = min(merged_buffer.end_time_min, ob[i].end_time_min);
        merged_buffer.end_time_max = max(merged_buffer.end_time_max, ob[i].end_time_max);
#pragma omp parallel for num_threads(bench->config->num_threads)
        for(uint j = 0; j < bench->bitmaps_size; j++) {
            merged_buffer.o_bitmaps[j] |= ob[i].o_bitmaps[j];
        }
    }
    double merged_buffer_bitmap = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\tmerged_buffer_bitmap:\t%.2f\n",merged_buffer_bitmap);

    merged_buffer.keys = new __uint128_t[merged_buffer.oversize_kv_count];
    merged_buffer.boxes = new f_box[merged_buffer.oversize_kv_count];
    uint *key_index = new uint[o_count]{0};        //0
    uint taken_id = 0;
    uint kv_count = 0;
    while(kv_count < merged_buffer.oversize_kv_count) {
        __uint128_t temp_key = (__uint128_t) 1 << 126;
        taken_id = 0;
        for (int i = 0; i < o_count; i++) {
            if (key_index[i] < ob[i].oversize_kv_count && temp_key > ob[i].keys[key_index[i]]){
                taken_id = i;
                temp_key = ob[taken_id].keys[key_index[taken_id]];
            }
        }
        merged_buffer.keys[kv_count] = temp_key;
        merged_buffer.boxes[kv_count] = ob[taken_id].boxes[key_index[taken_id]];
        key_index[taken_id]++;
        kv_count++;
    }
    double ordered_Multiway_Merge = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t ordered_Multiway_Merge:\t%.2f\n",ordered_Multiway_Merge);
    C_ctb.o_buffer = merged_buffer;

    double encoding_time = get_limit + write_bitmap_time + write_bitmap_time + key_info_to_new_key
                           + new_buffer_sort + key_info_to_buffer_key + merged_buffer_bitmap + ordered_Multiway_Merge;


    //    for(int i=0;i<bench->MemTable_count; i++){
    //        cout<<"key_index"<<key_index[i]<<endl;
    //    }

    vector<f_box> ctf_mbrs(new_CTF_count);
    float area_sum = 0;
    uint bitmap_kb = 0;
    for(uint i = 0; i < new_CTF_count; i++){
        bitmap_kb += C_ctb.ctfs[i].ctf_bitmap_size;
        ctf_mbrs[i] = C_ctb.ctfs[i].ctf_mbr;
        area_sum += C_ctb.ctfs[i].ctf_mbr.area();
    }
    bitmap_kb /= 1024;
    cout << "new_CTF_count " << new_CTF_count << " bitmap_grid " << bench->bitmap_grid << " bitmap_kb " << bitmap_kb << endl;
    float overlap_area = calculate_total_area(ctf_mbrs);
    cout << "overlap_area " << overlap_area << " area_sum " << area_sum << "overlap rate " << overlap_area / area_sum << endl;

    string prefix = to_string(bench->config->G_bytes) + "G";
    strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);

    string CTB_path = string(bench->config->CTB_meta_path) + prefix + "_CTB" + to_string(c_ctb_id);
    bench->dump_CTB_meta(CTB_path.c_str(), &C_ctb);
    double dump_ctb_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t dump_ctb_time:\t%.2f\n",dump_ctb_time);

#pragma omp parallel for num_threads(bench->config->num_threads)
    for(uint i = 0; i < new_CTF_count; i++){
        string ctf_path = string(bench->config->CTB_meta_path) + prefix + "_STcL" + to_string(c_ctb_id)+"-"+to_string(i);
        C_ctb.ctfs[i].dump_meta(ctf_path);
        string sst_path = bench->config->raid_path + to_string(i%2) + "/" + prefix + "_SSTable_"+to_string(c_ctb_id)+"-"+to_string(i);
        C_ctb.ctfs[i].dump_keys(sst_path.c_str());
        delete[] C_ctb.ctfs[i].keys;
    }
    double dump_ctf_keys_time = get_time_elapsed(bg_start,true);
    fprintf(stdout,"\t dump_ctf_keys_time:\t%.2f\n",dump_ctf_keys_time);
    double dump_C_time = dump_ctb_time + dump_ctf_keys_time;
    C_ctb.print_meta();
    q << merge_ctb_count * 2 << ',' << STR_time << ',' << encoding_time
      << ',' << load_keys_time << ',' << dump_C_time << endl;
    q.close();

    prefix = "N";
    strncpy(bench->keys_file_prefix, prefix.c_str(), sizeof(bench->keys_file_prefix) - 1);
//    delete []another_o_buffer_k;
//    delete []another_o_buffer_b;
//    for(uint i = 0; i < o_count; i++){
//        delete[] ob[i].keys;
//        delete[] ob[i].boxes;
//    }
    delete []merged_buffer.keys;
    delete []merged_buffer.boxes;
    return NULL;
}


//file size
//int main(int argc, char **argv){
//    clear_cache();
////    Point vp0[1000];
////    for(uint i = 0; i < 1000; i++){
////        vp0[i].x = -87.9 + 0.01 + (0.36 - 0.01) * get_rand_double();
////        vp0[i].y = 41.65 + 0.01 + (0.36 - 0.01) * get_rand_double();
////    }
////    ofstream OS_query_points;
////    OS_query_points.open("random_points1000", ios::out|ios::binary|ios::trunc);
////    OS_query_points.write((char *)vp0, 1000*sizeof(Point));
//
//    Point vp1[1000];
//    ifstream IS_query_points("random_points1000", ios::in | ios::binary);
//    if (!IS_query_points.is_open()) {
//        std::cerr << "Error opening random_points1000" << std::endl;
//    }
//    IS_query_points.read((char *)&vp1, 1000*sizeof(Point));
//    vector<Point> vp(vp1, vp1 + 1000);
//
//    double edge_length = 0.00104192;             //0.000329512
//    string path = "../data/meta/N";
//    //workbench * bench = C_load_meta(path.c_str());
//    uint max_ctb = 3;
//    workbench * bench = load_meta(path.c_str(), max_ctb);
//    for(int i = 0; i < max_ctb; i++) {
//        for (int j = 0; j < bench->config->CTF_count; j++) {
//            bench->ctbs[i].ctfs[j].keys = nullptr;
//        }
//    }
//    //for(uint new_split_num = 30; new_split_num <= 30; new_split_num+=1){
//    for(uint new_split_num = 10; new_split_num <= 10; new_split_num+=10){
//        clear_cache();
//        //bench->clear_all_keys();
//
//        uint new_CTF_count = new_split_num * new_split_num;
//        bench->config->split_num = new_split_num;
//        vector<object_info> str_list(bench->config->num_objects);
//        vector< vector<key_info> > keys_with_sid(new_CTF_count);
//        vector< vector<f_box> > mbrs_with_sid(new_CTF_count);
//        vector< vector<uint> > invert_index(new_CTF_count);
//        for (auto& vec : invert_index) {
//            vec.reserve(200000);        //20000000 / 400
//        }
//        vector< pair< vector<key_info>, vector<f_box> > > objects_map(bench->config->num_objects);
//
//        uint merge_ctb_count = 3;
//        for(uint i = 0; i < max_ctb; i += merge_ctb_count){
//            // for(uint j = i; j < i + merge_ctb_count; j++){
//            //     for(uint k = 0; k < bench->config->CTF_count; k++){
//            //         bench->load_CTF_keys(j, k);
//            //     }
//            // }
//            struct timeval bg_start = get_cur_time();
//            ex_ctf_size(edge_length, vp, bench, i, merge_ctb_count, keys_with_sid, mbrs_with_sid, invert_index, objects_map, str_list);
//            double compaction_total = get_time_elapsed(bg_start,true);
//            fprintf(stdout,"\tcompaction_total:\t%.2f\n",compaction_total);
//
//            //init
//            // for(uint j = i; j < i + merge_ctb_count; j++){
//            //     for(uint k = 0; k < bench->config->CTF_count; k++){
//            //         if(bench->ctbs[j].ctfs[k].keys){
//            //             delete []bench->ctbs[j].ctfs[k].keys;
//            //             bench->ctbs[j].ctfs[k].keys = nullptr;
//            //         }
//
//            //     }
//            // }
//            for(uint j = 0; j < new_CTF_count; j++){
//                keys_with_sid[j].clear();
//                mbrs_with_sid[j].clear();
//                invert_index[j].clear();
//            }
//            for(auto p : objects_map){
//                p.first.clear();
//                p.second.clear();
//            }
//        }
//    }
//
//    return 0;
//}


//bitmap granulirity
//int main(int argc, char **argv){
//    clear_cache();
//    double edge_length = 0.000104192;             //0.000329512
//    vector<Point> vp(1000);
//    for(uint i = 0; i < vp.size(); i++){
//        vp[i].x = -87.9 + 0.01 + (0.36 - 0.01) * get_rand_double();
//        vp[i].y = 41.65 + 0.01 + (0.36 - 0.01) * get_rand_double();
//    }
//    string path = "../data/meta/N";
//    //workbench * bench = C_load_meta(path.c_str());
//    uint max_ctb = 1;
//    uint new_split_num = 16;
//    uint new_CTF_count = 256;
//    workbench * bench = load_meta(path.c_str(), max_ctb);
//    for(int i = 0; i < max_ctb; i++) {
//        for (int j = 0; j < bench->config->CTF_count; j++) {
//            bench->ctbs[i].ctfs[j].keys = nullptr;
//        }
//    }
//    for(uint grid = 16; grid <= 1024; grid*=2){
//        clear_cache();
//        //bench->clear_all_keys();
//        bench->bitmap_grid = grid;
//        bench->config->split_num = new_split_num;
//        vector<object_info> str_list(bench->config->num_objects);
//        vector< vector<key_info> > keys_with_sid(new_CTF_count);
//        vector< vector<f_box> > mbrs_with_sid(new_CTF_count);
//        vector< vector<uint> > invert_index(new_CTF_count);
//        for (auto& vec : invert_index) {
//            vec.reserve(200000);        //20000000 / 400
//        }
//        vector< pair< vector<key_info>, vector<f_box> > > objects_map(bench->config->num_objects);
//        uint merge_ctb_count = max_ctb;
//        for(uint i = 0; i < max_ctb; i += merge_ctb_count){
//            struct timeval bg_start = get_cur_time();
//            ex_ctf_size(edge_length, vp, bench, 0, merge_ctb_count, keys_with_sid, mbrs_with_sid, invert_index, objects_map, str_list);
//            double compaction_total = get_time_elapsed(bg_start,true);
//            fprintf(stdout,"\tcompaction_total:\t%.2f\n",compaction_total);
//            for(uint j = 0; j < new_CTF_count; j++){
//                keys_with_sid[j].clear();
//                mbrs_with_sid[j].clear();
//                invert_index[j].clear();
//            }
//            for(auto p : objects_map){
//                p.first.clear();
//                p.second.clear();
//            }
//        }
//    }
//    return 0;
//}




//compaction test
int main(int argc, char **argv){
    clear_cache();
    string path = "../data/meta/N";
    //workbench * bench = C_load_meta(path.c_str());
    uint max_ctb = 32;
    workbench * bench = load_meta(path.c_str(), max_ctb);
    for(int i = 0; i < max_ctb; i++) {
        for (int j = 0; j < bench->config->CTF_count; j++) {
            bench->ctbs[i].ctfs[j].keys = nullptr;
        }
    }
    vector<uint> ctf_nums = {100, 196, 400, 784, 1600, 3136};
    assert(bench->config->G_bytes == 2);
    bench->config->G_bytes = 2;
    for(uint j = 0; j < 6; j++){                                    //2 4 8 16 32
        clear_cache();
        bench->clear_all_keys();
        //cout << "bench->ctb_count " << bench->ctb_count << endl;


        //    for(uint i = 0; i < 100; i++){
        //        bench->ctbs[0].ctfs[i].print_bitmap();
        //        //bench->ctbs[1].ctfs[i].ctf_mbr.print();
        //    }
        //    return 0;
        //    CTF * ctf = &bench->ctbs[0].ctfs[0];
        //    bench->load_CTF_keys(0, 0);
        //    uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
        //    uint temp_oid = 0;
        //    for(uint i = 0; i < ctf->CTF_kv_capacity; i++){
        //        key_info temp_ki;
        //        __uint128_t temp_128 = 0;
        //        memcpy(&temp_128, data + i * ctf->key_bit / 8, ctf->key_bit / 8);
        //        uint64_t value_mbr = 0;
        //        ctf->parse_key(temp_128, temp_ki, value_mbr);
        ////        box key_box = ctf->new_parse_mbr(value_mbr);
        ////        if(key_box.high[1] - key_box.low[1] > 0.008 || key_box.high[0] - key_box.low[0] > 0.008){
        ////            cerr << "i " <<  i <<" ";
        ////            key_box.print();
        ////        }
        //
        //    }
        //
        //    return 0;

        uint new_CTF_count = ctf_nums[j];
        uint new_split_num = sqrt(new_CTF_count);
        bench->config->split_num = new_split_num;
        vector<object_info> str_list(bench->config->num_objects);
        vector< vector<key_info> > keys_with_sid(new_CTF_count);
        vector< vector<f_box> > mbrs_with_sid(new_CTF_count);
        vector< vector<uint> > invert_index(new_CTF_count);
        for (auto& vec : invert_index) {
            vec.reserve(50000);        //20000000 / 400
        }

        uint merge_ctb_count = bench->config->G_bytes / 2;
        for(uint i = 0; i < max_ctb; i += merge_ctb_count){
            // for(uint j = i; j < i + merge_ctb_count; j++){
            //     for(uint k = 0; k < bench->config->CTF_count; k++){
            //         bench->load_CTF_keys(j, k);
            //     }
            // }
            vector< pair< vector<key_info>, vector<f_box> > > objects_map(bench->config->num_objects);
            struct timeval bg_start = get_cur_time();
            merge_dump(bench, i, merge_ctb_count, keys_with_sid, mbrs_with_sid, invert_index, objects_map, str_list);
            //cout << i << "i-merge_ctb_count" << merge_ctb_count << endl;
            double compaction_total = get_time_elapsed(bg_start,true);
            fprintf(stdout,"\tcompaction_total:\t%.2f\n",compaction_total);

            //init
//            for(uint j = i; j < i + merge_ctb_count; j++){
//                for(uint k = 0; k < bench->config->CTF_count; k++){
//                    if(bench->ctbs[j].ctfs[k].keys){
//                        delete []bench->ctbs[j].ctfs[k].keys;
//                        bench->ctbs[j].ctfs[k].keys = nullptr;
//                    }
//                }
//            }

            vector<object_info> temp_swap(bench->config->num_objects);
            swap(str_list, temp_swap);                                  //init
            for(uint j = 0; j < new_CTF_count; j++){
                keys_with_sid[j].clear();
                mbrs_with_sid[j].clear();
                invert_index[j].clear();
            }
            for(auto p : objects_map){
                p.first.clear();
                p.second.clear();
            }
        }
        bench->config->G_bytes *= 2;
    }
    return 0;
}
