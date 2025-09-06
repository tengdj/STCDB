/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "step_merge.h"

old_oversize_buffer::~old_oversize_buffer(){
    if(keys)
        delete []keys;
    if(boxes)
        delete []boxes;
}

// oversize_buffer::~oversize_buffer() {
//     if(keys){
//         delete []keys;
//         keys = nullptr;
//     }
//     if(boxes){
//         delete []boxes;
//         boxes = nullptr;
//     }
//     if(o_bitmaps){
//         delete []o_bitmaps;
//         o_bitmaps = nullptr;
//     }
// }

void oversize_buffer::print_buffer(){
    cout << "print_buffer " << endl;
    cout << "  oversize_kv_count: " << oversize_kv_count << endl;
    cout << "  start_time_min: " << start_time_min << endl;
    cout << "  start_time_max: " << start_time_max << endl;
    cout << "  end_time_min: " << end_time_min << endl;
    cout << "  end_time_max: " << end_time_max << endl;
}

void oversize_buffer::write_o_buffer(box map_mbr, uint bit_count){
    if(!o_bitmaps){
        o_bitmaps = new unsigned char[bit_count / 8];
    }
    for(uint kid = 0; kid < oversize_kv_count; kid++) {
        uint duration = (uint)((keys[kid] >> END_BIT) & ((1ULL << DURATION_BIT) - 1));
        uint end = (uint)(keys[kid] & ((1ULL << END_BIT) - 1));
        end += end_time_min;
        start_time_min = min(start_time_min, end - duration);
        start_time_max = max(start_time_max, end - duration);

        uint low0 = (boxes[kid].low[0] - map_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * DEFAULT_bitmap_edge;
        uint low1 = (boxes[kid].low[1] - map_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * DEFAULT_bitmap_edge;
        uint high0 = (boxes[kid].high[0] - map_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * DEFAULT_bitmap_edge;
        uint high1 = (boxes[kid].high[1] - map_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * DEFAULT_bitmap_edge;
        uint bit_pos = 0;
        for (uint i = low0; i <= high0 && i < DEFAULT_bitmap_edge; i++) {
            for (uint j = low1; j <= high1 && j < DEFAULT_bitmap_edge; j++) {
                bit_pos = i + j * DEFAULT_bitmap_edge;
                o_bitmaps[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
        }
    }
}

//total binary search must be faster
uint oversize_buffer::search_buffer(uint32_t oid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid) {
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = -1;
    int low = 0;
    int high = oversize_kv_count - 1;
    int mid;
    uint64_t temp_oid;
    while (low <= high) {
        mid = (low + high) / 2;
        if(mid >= oversize_kv_count){
            break;
        }
        temp_oid = get_key_oid(keys[mid]);
        if (temp_oid == oid){
            find = mid;
            high = mid - 1;
        }
        else if (temp_oid > oid){
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    if(find==-1){
        //cout<<"cannot find"<<endl;
        return 0;
    }
    //cout<<"exactly find"<<endl;
    uint cursor = find;
    while(cursor < oversize_kv_count){
        temp_oid = get_key_oid(keys[cursor]);
        if(temp_oid == oid){
            if(tq->abandon) {       //check_key_time
                count++;
                //cout<<get_key_target(keys[cursor])<<endl;
                if (search_multi) {
                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
                    search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);
                }
            }
        }
        else break;
        cursor++;
    }
    //cout<<"find !"<<endl;
    return count;
}

uint oversize_buffer::o_time_search(time_query * tq){
    uint count = 0;
    uint start, end, duration;
    for(uint i = 0; i < oversize_kv_count; i++){
        end = get_key_end(keys[i]);
        end += end_time_min;
        duration = get_key_duration(keys[i]);
        start = end - duration;
        if((tq->t_start < start) && (end < tq->t_end)) {
            count++;
        }
    }
    return count;
}

void CTF::eight_parallel() {
    assert(key_bit == 0);
    uint temp_bit = id_bit * 2 + duration_bit + end_bit + mbr_bit;
    assert(temp_bit <= 128);
    key_bit = temp_bit / 8 * 8;
    if(key_bit < temp_bit) key_bit += 8;
}

void CTF::get_ctf_bits(box map_mbr, configuration * config){
//    ctf_mbr.high[0] += 0.000001;
//    ctf_mbr.high[1] += 0.000001;

    id_bit = min_bits_to_store(config->num_objects);
    duration_bit = min_bits_to_store(config->max_meet_time - 1);
    end_bit = duration_bit;     //end is always rest in the range of duration

    //m granularity, is 0.00001
    //left bottom float to 0.00001 granularity  ~=  (map_mbr.high[0] - map_mbr.low[0]) / 30000   ~=   (map_mbr.high[1] - map_mbr.low[1]) / 42000
    double map_x_grid_length =  (map_mbr.high[0] - map_mbr.low[0]) / 3000;
    double map_y_grid_length =  (map_mbr.high[1] - map_mbr.low[1]) / 4200;

    uint low_x_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / map_x_grid_length;
    uint low_y_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / map_y_grid_length;
    low_x_bit = min_bits_to_store(max(low_x_grid, (uint)1) - 1);
    low_y_bit = min_bits_to_store(max(low_y_grid, (uint)1) - 1);

    //box, 1m granularity. 0.008 is the max edge restriction. area is 0.00005
//    int x_restriction = EDGE_REISTICTION / map_x_grid_length;
//    int y_restriction = EDGE_REISTICTION / map_y_grid_length;
//    edge_bit = min_bits_to_store((uint)max(x_restriction, y_restriction) - 1);
    edge_bit = max(low_x_bit, low_y_bit);
    mbr_bit = low_x_bit + low_y_bit + 2 * edge_bit;
    if(mbr_bit >= 64){
        std::cout << "Low X Bit: " << low_x_bit
         << "Low Y Bit: " << low_y_bit
         << "Edge Bit: " << edge_bit << std::endl;
    }
    assert(mbr_bit <= 64);
    eight_parallel();

    //42000m high, 30000m width, 100m granularity
    int old_width_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * 300;
    int old_high_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * 420;
    x_grid = max(old_width_grid + 1, 32);
    y_grid = max(old_high_grid + 1, 32);
    if(x_grid >= 256 || y_grid >= 256){
        cout << "x_grid " << x_grid << " y_grid " << y_grid << endl;
        //ctf_mbr.print();
        x_grid = min(old_width_grid + 1, 255);
        y_grid = min(old_high_grid + 1, 255);
    }
    ctf_bitmap_size = (x_grid * y_grid + 7) / 8;
    assert(ctf_bitmap_size < 256 * 256 / 8);

}

void CTF::get_ctf_bits(box map_mbr, configuration * config, float map_granularity){
//    ctf_mbr.high[0] += 0.000001;
//    ctf_mbr.high[1] += 0.000001;

    id_bit = min_bits_to_store(config->num_objects);
    duration_bit = min_bits_to_store(config->max_meet_time - 1);
    end_bit = duration_bit;     //end is always rest in the range of duration

    //m granularity, is 0.00001
    //left bottom float to 0.00001 granularity  ~=  (map_mbr.high[0] - map_mbr.low[0]) / 30000   ~=   (map_mbr.high[1] - map_mbr.low[1]) / 42000
    double map_x_grid_length =  (map_mbr.high[0] - map_mbr.low[0]) / 3000;
    double map_y_grid_length =  (map_mbr.high[1] - map_mbr.low[1]) / 4200;

    uint low_x_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / map_x_grid_length;
    uint low_y_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / map_y_grid_length;
    low_x_bit = min_bits_to_store(max(low_x_grid, (uint)1) - 1);
    low_y_bit = min_bits_to_store(max(low_y_grid, (uint)1) - 1);

    //box, 1m granularity. 0.008 is the max edge restriction. area is 0.00005
//    int x_restriction = EDGE_REISTICTION / map_x_grid_length;
//    int y_restriction = EDGE_REISTICTION / map_y_grid_length;
//    edge_bit = min_bits_to_store((uint)max(x_restriction, y_restriction) - 1);
    edge_bit = max(low_x_bit, low_y_bit);
    mbr_bit = low_x_bit + low_y_bit + 2 * edge_bit;
    if(mbr_bit >= 64){
        std::cout << "Low X Bit: " << low_x_bit
                  << "Low Y Bit: " << low_y_bit
                  << "Edge Bit: " << edge_bit << std::endl;
    }
    assert(mbr_bit <= 64);
    eight_parallel();

    //42000m high, 30000m width, 100m granularity
    int old_width_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * 300 * map_granularity;
    int old_high_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * 420 * map_granularity;
    x_grid = max(old_width_grid + 1, 2);
    y_grid = max(old_high_grid + 1, 2);
    ctf_bitmap_size = (x_grid * y_grid + 7) / 8;

}

//void CTF::get_ctf_bits(box map_mbr, configuration * config, uint bitmap_grid){
////    ctf_mbr.high[0] += 0.000001;
////    ctf_mbr.high[1] += 0.000001;
//
//    id_bit = min_bits_to_store(config->num_objects);
//    duration_bit = min_bits_to_store(config->max_meet_time - 1);
//    end_bit = duration_bit;     //end is always rest in the range of duration
//
//    //m granularity, is 0.00001
//    //left bottom float to 0.00001 granularity  ~=  (map_mbr.high[0] - map_mbr.low[0]) / 30000   ~=   (map_mbr.high[1] - map_mbr.low[1]) / 42000
//    double map_x_grid_length =  (map_mbr.high[0] - map_mbr.low[0]) / 3000;
//    double map_y_grid_length =  (map_mbr.high[1] - map_mbr.low[1]) / 4200;
//
//    uint low_x_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / map_x_grid_length;
//    uint low_y_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / map_y_grid_length;
//    low_x_bit = min_bits_to_store(low_x_grid - 1);
//    low_y_bit = min_bits_to_store(low_y_grid - 1);
//
//    //box, 1m granularity. 0.008 is the max edge restriction. area is 0.00005
////    int x_restriction = EDGE_REISTICTION / map_x_grid_length;
////    int y_restriction = EDGE_REISTICTION / map_y_grid_length;
////    edge_bit = min_bits_to_store((uint)max(x_restriction, y_restriction) - 1);
//    edge_bit = max(low_x_bit, low_y_bit);
//    mbr_bit = low_x_bit + low_y_bit + 2 * edge_bit;
//    if(mbr_bit >= 64){
//        std::cout << "Low X Bit: " << low_x_bit
//         << "Low Y Bit: " << low_y_bit
//         << "Edge Bit: " << edge_bit << std::endl;
//    }
//    assert(mbr_bit <= 64);
//    eight_parallel();
//
//    //42000m high, 30000m width, 100m granularity
//    // int old_width_grid = (ctf_mbr.high[0] - ctf_mbr.low[0]) / (map_mbr.high[0] - map_mbr.low[0]) * bitmap_grid;
//    // int old_high_grid = (ctf_mbr.high[1] - ctf_mbr.low[1]) / (map_mbr.high[1] - map_mbr.low[1]) * bitmap_grid;
//    // x_grid = max(old_width_grid + 1, 64);
//    // y_grid = max(old_high_grid + 1, 64);
//    // if(x_grid >= 256 || y_grid >= 256){
//    //     cout << "x_grid " << x_grid << " y_grid " << y_grid << endl;
//    //     //ctf_mbr.print();
//    //     x_grid = min(old_width_grid + 1, 255);
//    //     y_grid = min(old_high_grid + 1, 255);
//    // }
//    x_grid = bitmap_grid;
//    y_grid = bitmap_grid;
//    ctf_bitmap_size = (x_grid * y_grid + 7) / 8;        //error uint16 range
//    //assert(ctf_bitmap_size < 256 * 256 / 8);
//
//}

// uint CTF::count_meta_size(){
//     uint total_B = sizeof(CTF) + ctf_bitmap_size;
//     return total_B;
// }

__uint128_t CTF::serial_key(uint64_t pid, uint64_t target, uint64_t duration, uint64_t end, __uint128_t value_mbr){
    __uint128_t temp_key = ((__uint128_t)pid << (id_bit + duration_bit + end_bit + mbr_bit)) + ((__uint128_t)target << (duration_bit + end_bit + mbr_bit))
                            + ((__uint128_t)duration << (end_bit + mbr_bit)) + ((__uint128_t)end << (mbr_bit)) + value_mbr;
    return temp_key;
}

uint64_t CTF::serial_mbr(f_box * b){
//    if(b->low[0] > b->high[0]){
//        swap(b->low[0], b->high[0]);                //old data error
//    }
//    if(b->low[1] > b->high[1]){
//        swap(b->low[1], b->high[1]);
//    }
    assert(b->low[0] <= b->high[0] && b->low[1] <= b->high[1]);
    float longer_edge = max(ctf_mbr.high[0] - ctf_mbr.low[0], ctf_mbr.high[1] - ctf_mbr.low[1]);
    uint64_t low0 = (b->low[0] - ctf_mbr.low[0])/(ctf_mbr.high[0] - ctf_mbr.low[0]) * ((1ULL << (low_x_bit)) - 1);
    uint64_t low1 = (b->low[1] - ctf_mbr.low[1])/(ctf_mbr.high[1] - ctf_mbr.low[1]) * ((1ULL << (low_y_bit)) - 1);
    if(b->high[0] - b->low[0] > longer_edge || b->high[1] - b->low[1] > longer_edge){
        cout << "longer_edge: " << longer_edge << " " << b->high[0]-b->low[0] << " " << b->high[1]-b->low[1] << endl;
        b->print();
    }
    uint64_t x = (b->high[0] - b->low[0])/longer_edge * ((1ULL << (edge_bit)) - 1);
    uint64_t y = (b->high[1] - b->low[1])/longer_edge * ((1ULL << (edge_bit)) - 1);
    uint64_t value_mbr = ((uint64_t)low0 << (low_y_bit + edge_bit + edge_bit)) + ((uint64_t)low1 << (edge_bit + edge_bit))
                         + ((uint64_t)x << (edge_bit)) + (uint64_t)y;
    return value_mbr;
}

void CTF::print_key(__uint128_t key){
    __uint128_t value_mbr = key & ((__uint128_t(1) << mbr_bit) - 1);
    key >>= mbr_bit;
    uint64_t end = key & ((1ULL << end_bit) - 1);
    key >>= end_bit;
    uint64_t duration = key & ((1ULL << duration_bit) - 1);
    key >>= duration_bit;
    uint64_t target = key & ((1ULL << id_bit) - 1);
    key >>= id_bit;
    uint64_t pid = key;

    std::cout << "Parsed Key: " << std::endl;
    std::cout << "  PID: " << pid << std::endl;
    std::cout << "  Target: " << target << std::endl;
    std::cout << "  Duration: " << duration << std::endl;
    std::cout << "  End: " << end << std::endl;
    //std::cout << "  Value MBR: " << value_mbr << std::endl;
}

uint CTF::ctf_get_key_oid(__uint128_t temp_128){
    return (uint)((temp_128 >> (id_bit + duration_bit + end_bit + mbr_bit)) & ((1ULL << id_bit) - 1));
}

void CTF::parse_key(__uint128_t key, uint &pid, uint &target, uint &duration, uint &end, uint64_t &value_mbr){
    value_mbr = key & ((uint64_t(1) << mbr_bit) - 1);
    key >>= mbr_bit;
    end = key & ((1ULL << end_bit) - 1);
    key >>= end_bit;
    duration = key & ((1ULL << duration_bit) - 1);
    key >>= duration_bit;
    target = key & ((1ULL << id_bit) - 1);
    key >>= id_bit;
    pid = key;
}

void CTF::parse_key(__uint128_t key, key_info &ki, uint64_t &value_mbr){
    value_mbr = key & ((uint64_t(1) << mbr_bit) - 1);
    key >>= mbr_bit;
    ki.end = key & ((1ULL << end_bit) - 1);
    key >>= end_bit;
    ki.duration = key & ((1ULL << duration_bit) - 1);
    key >>= duration_bit;
    ki.target = key & ((1ULL << id_bit) - 1);
    key >>= id_bit;
    ki.oid = key;
}

box CTF::new_parse_mbr(uint64_t value_mbr){
    uint64_t mask_edge = (1ULL << edge_bit) - 1;
    uint64_t mask_low_y = (1ULL << low_y_bit) - 1;
    uint64_t mask_low_x = (1ULL << low_x_bit) - 1;

    uint64_t y = value_mbr & mask_edge;  
    value_mbr >>= edge_bit;
    uint64_t x = value_mbr & mask_edge;  
    value_mbr >>= edge_bit;
    uint64_t low1 = value_mbr & mask_low_y; 
    value_mbr >>= low_y_bit;
    uint64_t low0 = value_mbr & mask_low_x;

    float longer_edge = max(ctf_mbr.high[0] - ctf_mbr.low[0], ctf_mbr.high[1] - ctf_mbr.low[1]);
    box b;
    b.low[0] = ctf_mbr.low[0] + (double)low0 / mask_low_x * (ctf_mbr.high[0] - ctf_mbr.low[0]);
    b.low[1] = ctf_mbr.low[1] + (double)low1 / mask_low_y * (ctf_mbr.high[1] - ctf_mbr.low[1]);
    b.high[0] = b.low[0] + (double)x / mask_edge * longer_edge;
    b.high[1] = b.low[1] + (double)y / mask_edge * longer_edge;
    return b;
}

f_box CTF::new_parse_mbr_f_box(uint64_t value_mbr){
    uint64_t mask_edge = (1ULL << edge_bit) - 1;
    uint64_t mask_low_y = (1ULL << low_y_bit) - 1;
    uint64_t mask_low_x = (1ULL << low_x_bit) - 1;

    uint64_t y = value_mbr & mask_edge;
    value_mbr >>= edge_bit;
    uint64_t x = value_mbr & mask_edge;
    value_mbr >>= edge_bit;
    uint64_t low1 = value_mbr & mask_low_y;
    value_mbr >>= low_y_bit;
    uint64_t low0 = value_mbr & mask_low_x;

    float longer_edge = max(ctf_mbr.high[0] - ctf_mbr.low[0], ctf_mbr.high[1] - ctf_mbr.low[1]);
    f_box b;
    b.low[0] = ctf_mbr.low[0] + (double)low0 / mask_low_x * (ctf_mbr.high[0] - ctf_mbr.low[0]);
    b.low[1] = ctf_mbr.low[1] + (double)low1 / mask_low_y * (ctf_mbr.high[1] - ctf_mbr.low[1]);
    b.high[0] = b.low[0] + (double)x / mask_edge * longer_edge;
    b.high[1] = b.low[1] + (double)y / mask_edge * longer_edge;
    return b;
}

uint old_get_key_oid(__uint128_t key){
    return (uint)((key >> (26 + 36 + 12 + 12)) & ((1ULL << 26) - 1));
}

uint old_get_key_target(__uint128_t key){
    return (uint)((key >> (36 + 12 + 12)) & ((1ULL << 26) - 1));
}

uint64_t old_get_key_mbr_code(__uint128_t key){
    return (uint64_t)((key >> ( 12 + 12)) & ((1ULL << 36) - 1));
}

uint old_get_key_duration(__uint128_t key){
    return (uint)((key >> 12) & ((1ULL << 12) - 1));
}

uint old_get_key_end(__uint128_t key){
    return (uint)(key & ((1ULL << 12) - 1));
}

void old_parse_mbr(__uint128_t key, f_box &b, f_box bitmap_mbr){        //MBR_BIT 36
    uint64_t mbr_code = old_get_key_mbr_code(key);
    uint64_t low0 = mbr_code >> (MBR_BIT/4*3);
    uint64_t low1 = (mbr_code >> (MBR_BIT/2)) & ((1ULL << (MBR_BIT/4)) - 1);
    uint64_t high0 = (mbr_code >> (MBR_BIT/4)) & ((1ULL << (MBR_BIT/4)) - 1);
    uint64_t high1 = mbr_code & ((1ULL << (MBR_BIT/4)) - 1);

    if(low0 > high0){
        high0 = low0;
    }
    if(low1 > high1){
        high1 = low1;
    }

    b.low[0] = (double)low0/((1ULL << (MBR_BIT/4)) - 1) * (bitmap_mbr.high[0] - bitmap_mbr.low[0]) + bitmap_mbr.low[0];
    b.low[1] = (double)low1/((1ULL << (MBR_BIT/4)) - 1) * (bitmap_mbr.high[1] - bitmap_mbr.low[1]) + bitmap_mbr.low[1];
    b.high[0] = (double)high0/((1ULL << (MBR_BIT/4)) - 1) * (bitmap_mbr.high[0] - bitmap_mbr.low[0]) + bitmap_mbr.low[0];
    b.high[1] = (double)high1/((1ULL << (MBR_BIT/4)) - 1) * (bitmap_mbr.high[1] - bitmap_mbr.low[1]) + bitmap_mbr.low[1];

    if(b.low[0] > b.high[0]){
        cerr << "x_err ";
        swap(b.low[0], b.high[0]);
    }
    if(b.low[1] > b.high[1]){
        cerr << "still err ";
        swap(b.low[1], b.high[1]);
    }
}

void CTF::transfer_all_in_one(){
    assert(key_bit % 8 == 0);
    uint8_t * shrink_keys = new uint8_t[key_bit / 8 * CTF_kv_capacity];
    unsigned char * shrink_bitmap = new uint8_t[ctf_bitmap_size];
    memset(shrink_bitmap, 0, ctf_bitmap_size); 
    uint pid, target, duration, end;
    uint64_t new_value_mbr;
    f_box real_mbr;
    for(uint kid = 0; kid < CTF_kv_capacity; kid++) {
        pid = old_get_key_oid(keys[kid]);
        target = old_get_key_target(keys[kid]);
        duration = old_get_key_duration(keys[kid]);
        end = old_get_key_end(keys[kid]);
        end += end_time_min;
        start_time_min = min(start_time_min, end - duration);
        start_time_max = max(start_time_max, end - duration);
        old_parse_mbr(keys[kid], real_mbr, ctf_mbr);
        assert(ctf_mbr.contain(real_mbr));

        uint low0 = (real_mbr.low[0] - ctf_mbr.low[0])/(ctf_mbr.high[0] - ctf_mbr.low[0]) * x_grid;
        uint low1 = (real_mbr.low[1] - ctf_mbr.low[1])/(ctf_mbr.high[1] - ctf_mbr.low[1]) * y_grid;
        uint high0 = (real_mbr.high[0] - ctf_mbr.low[0])/(ctf_mbr.high[0] - ctf_mbr.low[0]) * x_grid;
        uint high1 = (real_mbr.high[1] - ctf_mbr.low[1])/(ctf_mbr.high[1] - ctf_mbr.low[1]) * y_grid;
        assert(low0 <= x_grid);
        assert(high0 <= x_grid);
        assert(low1 <= y_grid);
        assert(high1 <= y_grid);
        uint bit_pos = 0;
        for(uint i=low0;i<=high0 && i < x_grid;i++){
            for(uint j=low1;j<=high1 && j < y_grid;j++){

                bit_pos = i + j * x_grid;
                //assert(bit_pos / 8 < ctf_bitmap_size);
                shrink_bitmap[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
        }

        new_value_mbr = serial_mbr(&real_mbr);
        __uint128_t temp_key = serial_key(pid, target, duration, end, new_value_mbr);
        memcpy(&shrink_keys[kid * key_bit / 8], &temp_key, key_bit / 8);

    }
    delete []bitmap;
    bitmap = shrink_bitmap;
    delete []keys;
    keys = reinterpret_cast<__uint128_t *>(shrink_keys);

}

int CTF::binary_search(uint oid) {
    uint key_bytes = key_bit / 8;
    uint low = 0;
    uint high = CTF_kv_capacity - 1;
    int find = -1;
    __uint128_t temp_128 = 0;
    uint8_t * data = reinterpret_cast<uint8_t *>(keys);
    while (low <= high) {
        size_t mid = low + (high - low) / 2;
        if (mid >= CTF_kv_capacity) {
            break;
        }
        memcpy(&temp_128, data + mid * key_bytes, key_bytes);
        uint temp_pid = ctf_get_key_oid(temp_128);
        if (temp_pid == oid) {
            find = mid;
            high = mid - 1;
        } else if (temp_pid < oid) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return find;
}

//range query
uint CTF::search_SSTable(uint pid, time_query * tq, bool search_multi, atomic<long long> &search_count, uint *search_multi_pid) {
    uint count = 0;
    //cout<<"into search_SSTable"<<endl;
    int find = binary_search(pid);
    if(find==-1){
        //cout<<"cannot find"<<endl;
        return 0;
    }
    //cout<<"exactly find"<<endl;
    uint cursor = find;
    __uint128_t temp_128 = 0;
    uint8_t * data = reinterpret_cast<uint8_t *>(keys);
    while(cursor < CTF_kv_capacity){
        memcpy(&temp_128, data + cursor * key_bit / 8, key_bit / 8);
        uint temp_pid = ctf_get_key_oid(temp_128);
        if(temp_pid == pid){
            if(tq->abandon) {       //check_key_time
                uint64_t value_mbr = 0;
                key_info temp_ki;
                parse_key(temp_128, temp_ki, value_mbr);
                box key_box = new_parse_mbr(value_mbr);
                box search_area(-87.93905, 41.95142, -87.87725, 42.00961);
                if (search_area.intersect(key_box)) {
                    count++;
                }
                //cout<< temp_pid << "-" << get_key_target(keys[cursor]) << endl;
                if (search_multi) {
                    long long search_multi_length = search_count.fetch_add(1, std::memory_order_relaxed);
                    //search_multi_pid[search_multi_length] = get_key_target(keys[cursor]);  !!!!!!!!!!!!
                }
            }
//            else{
//                vector<Interval> result = query_intervals(start_sorted, end_sorted, tq->t_start, tq->t_end);
//            }
        }
        else break;
        cursor++;
    }
    //cout<<"find !"<<endl;
    return count;
}

uint CTF::time_search(time_query * tq){
    uint count = 0;
    uint8_t * data = reinterpret_cast<uint8_t *>(keys);
    for(uint i = 0; i < CTF_kv_capacity; i++){
        key_info temp_ki;
        __uint128_t temp_128 = 0;
        memcpy(&temp_128, data + i * key_bit / 8, key_bit / 8);
        uint64_t value_mbr = 0;
        parse_key(temp_128, temp_ki, value_mbr);
        temp_ki.end += end_time_min;
        uint start = temp_ki.end - temp_ki.duration;
        if((tq->t_start < start) && (temp_ki.end < tq->t_end)) {
            count++;
        }
    }
    return count;
}

void CTF::print_ctf_meta() {
    std::cout << "CTF Metadata:" << std::endl;

    std::cout << "CTF MBR: ["
              << ctf_mbr.low[0] << ", " << ctf_mbr.low[1] << "] -> ["
              << ctf_mbr.high[0] << ", " << ctf_mbr.high[1] << "]" << std::endl;

    std::cout << "CTF KV Capacity: " << CTF_kv_capacity << std::endl;

    std::cout << "Start Time Min: " << start_time_min << std::endl;
    std::cout << "Start Time Max: " << start_time_max << std::endl;
    std::cout << "End Time Min: " << end_time_min << std::endl;
    std::cout << "End Time Max: " << end_time_max << std::endl;

    std::cout << "Key Bit: " << key_bit << std::endl;
    std::cout << "ID Bit: " << id_bit << std::endl;
    std::cout << "Duration Bit: " << duration_bit << std::endl;
    std::cout << "End Bit: " << end_bit << std::endl;
    std::cout << "Low X Bit: " << low_x_bit << std::endl;
    std::cout << "Low Y Bit: " << low_y_bit << std::endl;
    std::cout << "Edge Bit: " << edge_bit << std::endl;
    std::cout << "MBR Bit: " << mbr_bit << std::endl;

    std::cout << "X Grid: " << x_grid << std::endl;
    std::cout << "Y Grid: " << y_grid << std::endl;

    std::cout << "CTF Bitmap Size (in bytes): " << ctf_bitmap_size << std::endl;

}

void CTF::print_bitmap(){
//    for(uint i=0;i<ctf_bitmap_size;i++){
//        cout << (bitmap[i] & 0xff);
//    }

    cerr << endl;
    //ctf_mbr.print();
    Point * bit_points = new Point[ctf_bitmap_size * 8];
    uint count_p = 0;
    for(uint i=0;i<ctf_bitmap_size * 8;i++){
        if(bitmap[i/8] & (1<<(i%8))){
            Point bit_p;
            uint x=0,y=0;
            x = i % x_grid;
            y = i / x_grid;
            bit_p.x = (double)x/x_grid*(ctf_mbr.high[0] - ctf_mbr.low[0]) + ctf_mbr.low[0];
            bit_p.y = (double)y/y_grid*(ctf_mbr.high[1] - ctf_mbr.low[1]) + ctf_mbr.low[1];
            bit_points[count_p] = bit_p;
            count_p++;
        }
    }
    //cout<<"bit_points.size():"<<count_p<<endl;
    print_points(bit_points,count_p);
    delete[] bit_points;
}

void CTF::dump_meta(const string& path){
    struct timeval start_time = get_cur_time();
    ofstream wf(path.c_str(), ios::out|ios::binary|ios::trunc);
    wf.write((char *)this, sizeof(CTF));
    wf.write((char *)bitmap, ctf_bitmap_size);
    wf.close();
}

void CTF::dump_keys(const string& path){
    ofstream SSTable_of;
    SSTable_of.open(path , ios::out|ios::binary|ios::trunc);
    assert(SSTable_of.is_open());
    SSTable_of.write((char *)keys, key_bit / 8 * (uint64_t)CTF_kv_capacity);
    SSTable_of.close();
}

CTB::~CTB(){
    if(ctfs)
        delete []ctfs;
    if(sids)
        delete []sids;
//    if(box_rtree)
//        delete box_rtree;
}

