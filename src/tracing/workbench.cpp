/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"
#include <unordered_set>


workbench::workbench(workbench *bench):workbench(bench->config){
	grids_stack_index = bench->grids_stack_index;
	schema_stack_index = bench->schema_stack_index;
	cur_time = bench->cur_time;
	mbr = bench->mbr;

    if(config->bloom_filter) {
        dwMaxItems = bench->dwMaxItems;
        dProbFalse = bench->dProbFalse;
        dwFilterBits = bench->dwFilterBits;
        dwHashFuncs = bench->dwHashFuncs;
        dwSeed = bench->dwSeed;
        dwFilterSize = bench->dwFilterSize;
    }
    if(true){
        bit_count = bench->bit_count;
        bitmaps_size = bench->bitmaps_size;
    }
}

workbench::workbench(configuration *conf){

	config = conf;

	// setup the capacity of each container
	grid_capacity = config->grid_amplify*config->grid_capacity;
	// each grid contains averagely grid_capacity/2 objects, times 4 for enough space
	grids_stack_capacity = 4*max((uint)100000, config->num_objects/config->grid_capacity);

	// the number of all QTree Nodes
	schema_stack_capacity = 1.6*grids_stack_capacity;

	tmp_space_capacity = config->num_objects;

	filter_list_capacity = config->num_objects;

	grid_check_capacity = config->refine_size*config->num_objects;

	//meeting_capacity = config->num_objects/4;
	//meeting_capacity = 100;

	insert_lk = new pthread_mutex_t[MAX_LOCK_NUM];
	for(int i=0;i<MAX_LOCK_NUM;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
	for(int i=0;i<100;i++){
		data[i] = NULL;
		data_size[i] = 0;
	}
}

void workbench::clear(){
	for(int i=0;i<100;i++){
		if(data[i]!=NULL){
			free(data[i]);
			data[i] = NULL;
			data_size[i] = 0;
		}
		data_index = 0;
	}
	delete insert_lk;
}


void *workbench::allocate(size_t size){
	lock();
	uint cur_idx = data_index++;
	unlock();
	data[cur_idx] = malloc(size);
	data_size[cur_idx] = size;
	return data[cur_idx];
}

void workbench::claim_space(){

	size_t size = 0;

	size = grid_capacity*grids_stack_capacity*sizeof(uint);
	grids = (uint *)allocate(size);
	log("\t%.2f MB\tgrids",size/1024.0/1024.0);

	size = grids_stack_capacity*sizeof(uint);
	grid_counter = (uint *)allocate(size);
	//log("\t%.2f MB\tgrid counter",size/1024.0/1024.0);

	size = grids_stack_capacity*sizeof(uint);
	grids_stack = (uint *)allocate(size);
	//log("\t%.2f MB\tgrids stack",size/1024.0/1024.0);
	for(int i=0;i<grids_stack_capacity;i++){
		grids_stack[i] = i;
	}

	size = schema_stack_capacity*sizeof(QTSchema);
	schema = (QTSchema*)allocate(size);
	log("\t%.2f MB\tschema",size/1024.0/1024.0);

	size = schema_stack_capacity*sizeof(uint);
	schema_stack = (uint *)allocate(size);
	//log("\t%.2f MB\tschema stack",size/1024.0/1024.0);
	for(int i=0;i<schema_stack_capacity;i++){
		schema_stack[i] = i;
		schema[i].type = INVALID;
	}

	size = grid_check_capacity*sizeof(checking_unit);
	grid_check = (checking_unit *)allocate(size);
	log("\t%.2f MB\tchecking units",size/1024.0/1024.0);

	size = config->num_meeting_buckets*sizeof(meeting_unit);
	meeting_buckets = (meeting_unit *)allocate(size);
	log("\t%.2f MB\tmeeting bucket space",size/1024.0/1024.0);

//	size = meeting_capacity*sizeof(meeting_unit);
//	meetings = (meeting_unit *)allocate(size);
//	log("\t%.2f MB\tmeeting space",size/1024.0/1024.0);
//
//#pragma omp parallel for
//	for(size_t i=0;i<config->num_meeting_buckets;i++){
//		meeting_buckets[i].key = ULL_MAX;
//	}

    size = config->MemTable_capacity * sizeof(__uint128_t *);                 //sort
    h_keys = (__uint128_t **)allocate(size);
    //log("\t%.2f MB\tmeeting bucket space",size/1024.0/1024.0);

    for(int i=0; i<config->MemTable_capacity; i++){
        size = config->kv_capacity*sizeof(__uint128_t);
        h_keys[i] = (__uint128_t *)allocate(size);
        log("\t%.2f MB\ta element of h_keys",size/1024.0/1024.0);
    }

    size = config->search_single_capacity*sizeof(search_info_unit);                   //search
    search_single_list = (search_info_unit *)allocate(size);
    size = config->search_single_capacity*sizeof(uint);
    search_multi_pid = (uint *)allocate(size);
    size = config->search_multi_capacity*sizeof(search_info_unit);
    search_multi_list = (search_info_unit *)allocate(size);

    if(config->bloom_filter) {
        size = config->MemTable_capacity * sizeof(unsigned char *);                       //bloom
        pstFilter = (unsigned char **) allocate(size);

#pragma omp parallel for
        for (int i = 0; i < config->MemTable_capacity; i++) {
            size = dwFilterSize;
            pstFilter[i] = (unsigned char *) allocate(size);
            log("\t%.2f MB\t pstFilter", size / 1024.0 / 1024.0);
            memset(pstFilter[i], 0, dwFilterSize);
        }
    }

    size = config->big_sorted_run_capacity*sizeof(sorted_run);
    bg_run = (sorted_run *)allocate(size);

    if(true){
        size = config->MemTable_capacity * sizeof(unsigned char *);
        h_bitmaps = (unsigned char **)allocate(size);
        size = config->MemTable_capacity * sizeof(box *);
        h_bitmap_mbrs = (box **)allocate(size);
        size = config->MemTable_capacity * sizeof(unsigned short *);
        h_wids = (unsigned short **)allocate(size);

        for(int i=0;i<config->MemTable_capacity; i++){
            size = bitmaps_size;
            h_bitmaps[i] = (unsigned char *) allocate(size);
            log("\t%.2f MB\t h_bitmaps", size / 1024.0 / 1024.0);
            size = config->SSTable_count * sizeof(box);
            h_bitmap_mbrs[i] = (box *)allocate(size);
            log("\t%.2f MB\t h_bitmap_mbrs", size / 1024.0 / 1024.0);
            size = config->num_objects * sizeof(unsigned short);
            h_wids[i] = (unsigned short *) allocate(size);
            log("\t%.2f MB\t h_wids", size / 1024.0 / 1024.0);
        }
    }
}

box workbench::bit_box(box b){
    box new_b;
    new_b.low[0] = (b.low[0] - mbr.low[0])/(mbr.high[0] - mbr.low[0]) * (pow(2,WID_BIT/2) - 1);
    new_b.low[1] = (b.low[1] - mbr.low[1])/(mbr.high[1] - mbr.low[1]) * (pow(2,WID_BIT/2) - 1);
    new_b.high[0] = (b.high[0] - mbr.low[0])/(mbr.high[0] - mbr.low[0]) * (pow(2,WID_BIT/2) - 1);
    new_b.high[1] = (b.high[1] - mbr.low[1])/(mbr.high[1] - mbr.low[1]) * (pow(2,WID_BIT/2) - 1);
    return new_b;
}

void workbench::load_big_sorted_run(uint b){
    if(!bg_run[b].sst){
        bg_run[b].sst = new SSTable[config->SSTable_count];
    }
    ifstream read_sst;
    for(int i = 0; i < config->SSTable_count; i++){
        if(!bg_run[b].sst[i].keys){
            string filename = "../store/SSTable_"+to_string(b)+"-"+to_string(i);
            //cout<<filename<<endl;
            read_sst.open(filename);
            assert(read_sst.is_open());
            bg_run[b].sst[i].keys = new __uint128_t [SSTable_kv_capacity];
            read_sst.read((char *)bg_run[b].sst[i].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
            read_sst.close();
        }
    }

}

bool workbench::search_memtable(uint64_t pid){          //wid_pid
    cout<<"memtable search"<<pid<<endl;
    uint offset = 0;
    if(big_sorted_run_count%2==1){
        offset = config->MemTable_capacity/2;
    }
    bool ret = false;
//    for(int i=0;i<MemTable_count;i++) {
//        int find = -1;
//        int low = 0;
//        int high = config->kv_restriction - 1;
//        int mid;
//        uint64_t temp_pid;
//        //box temp_box;
//        while (low <= high) {
//            mid = (low + high) / 2;
//            temp_pid = (h_keys[offset+i][mid] >> 23) ;
//            if (temp_pid == pid) {
//                find = mid;
//                ret = true;
//                break;
//            } else if (temp_pid > pid) {
//                high = mid - 1;
//            } else {
//                low = mid + 1;
//            }
//        }
//        if (find == -1) {
//            cout << "cannot find" << endl;
//            break;
//        }
//        cout << "exactly find" << endl;
//        uint cursor = find;
//        while (temp_pid == pid && cursor >= 1) {
//            cursor--;
//            temp_pid = (h_keys[offset+i][cursor] >> 23) ;
//        }
//        if (temp_pid == pid && cursor == 0) {
//            //print_128(h_keys[offset+i][0]);
//            cout<<"duration:"<<(uint)(h_values[offset+i][0] >> 113)<<endl;
//            cout<<"target:"<<(uint)((h_values[offset+i][0] >> 88) & ((1ULL << 25) - 1))<<endl;
//            box temp_box(h_values[offset+i][0]);
//            temp_box.print();
//            if(search_multi){
//                search_multi_pid[search_multi_length] = (h_keys[offset+i][0] >> 23) ;       //real pid
//                search_multi_length++;
//            }
//        }
//        while (cursor + 1 < config->kv_restriction) {
//            cursor++;
//            temp_pid = (h_keys[offset+i][cursor] >> 23) & ((1ULL << 25) - 1);
//            if (temp_pid == pid) {
//                //cout<<h_keys[offset+i][cursor];
//                print_128(h_keys[offset+i][cursor]);
//                cout<<"duration:"<<(uint)(h_values[offset+i][cursor] >> 113)<<endl;
//                cout<<"target:"<<(uint)((h_values[offset+i][cursor] >> 88) & ((1ULL << 25) - 1))<<endl;
//                box temp_box(h_values[offset+i][cursor]);
//                temp_box.print();
//                if(search_multi) {
//                    search_multi_pid[search_multi_length] = (h_keys[offset+i][cursor] >> 23);
//                    search_multi_length++;
//                }
//            }
//        }
//    }
    return ret;
}

bool workbench::search_in_disk(uint pid, uint timestamp){
    //cout<<"disk search "<<pid<<endl;
    bool ret = false;
    for(int i=0;i<big_sorted_run_count;i++) {
        if ( (bg_run[i].start_time_min < timestamp)&&(timestamp < bg_run[i].end_time_max) ) {
            //cout<<"big_sorted_run_num:"<<i<<endl;
            uint64_t wp = pid;
            if(bg_run[i].wids[pid] == 0){
                wid_filter_count++;
                continue;
            }
            else{
                wp += ((uint64_t)bg_run[i].wids[pid] << PID_BIT);
                //cout<<"wp: "<<wp<<endl;
            }
            if(!bg_run[i].sst){
                //cout<<"new SSTables"<<endl;
                bg_run[i].sst = new SSTable[config->SSTable_count];                   //maybe useful later, should not delete after this func , if(!NULL)
            }
            ifstream read_sst;

            //high level binary search
            int find = -1;
            int low = 0;
            int high = config->SSTable_count - 1;
            int mid;
            while (low <= high) {
                mid = (low + high) / 2;
                //cout << bg_run[i].first_widpid[mid] << endl;
                if (bg_run[i].first_widpid[mid] == wp){
                    find = mid;
                    break;
                }
                else if (bg_run[i].first_widpid[mid] > wp){
                    high = mid - 1;
                }
                else {
                    low = mid + 1;
                }
            }
            if(find==-1){
//                cout<<"not find in first_widpid"<<endl;
//                cout << low << "-" << bg_run[i].first_widpid[low] << endl;
//                cout << mid << "-" << bg_run[i].first_widpid[mid] << endl;
//                cout << high << "-" << bg_run[i].first_widpid[high] << endl;
                if(high<0){
                    high = 0;
                }
                if(!bg_run[i].sst[high].keys){
                    //cout<<"new SSTables keys"<<high<<endl;
                    string filename = "../store/SSTable_"+to_string(i)+"-"+to_string(high);
                    //cout<<filename<<endl;
                    read_sst.open(filename);                   //final place is not high+1, but high
                    assert(read_sst.is_open());
                    bg_run[i].sst[high].keys = new __uint128_t [SSTable_kv_capacity];
                    read_sst.read((char *)bg_run[i].sst[high].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
                    read_sst.close();
                }
                uint target_count = bg_run[i].sst[high].search_SSTable(wp,search_multi,SSTable_kv_capacity,search_multi_length,search_multi_pid);
                disk_find_count += target_count;
                if(target_count){
                    ret = true;
                }
                else{
                    disk_not_find_count++;
                }
                continue;
            }
            //cout<<"high level binary search finish and find"<<endl;

            //for the case, there are many SSTables that first_widpid==wp
            //find start and end
            uint pid_start = find;
            while(pid_start>=1){
                pid_start--;
                if(bg_run[i].first_widpid[pid_start] != wp){
                    break;
                }
            }
            if(!bg_run[i].sst[pid_start].keys){
                read_sst.open("../store/SSTable_"+to_string(i)+"-"+to_string(pid_start));
                assert(read_sst.is_open());
                bg_run[i].sst[pid_start].keys = new __uint128_t[SSTable_kv_capacity];
                read_sst.read((char *)bg_run[i].sst[pid_start].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
                read_sst.close();
            }
            bg_run[i].sst[pid_start].search_SSTable(wp,search_multi,SSTable_kv_capacity,search_multi_length,search_multi_pid);
            uint cursor = pid_start+1;
            uint temp_pid;
            while(true) {
                read_sst.open("../store/SSTable_" + to_string(i) + "-" + to_string(cursor));
                assert(read_sst.is_open());
                if(!bg_run[i].sst[cursor].keys){
                    bg_run[i].sst[cursor].keys = new __uint128_t[SSTable_kv_capacity];
                }
                read_sst.read((char *) bg_run[i].sst[cursor].keys, sizeof(__uint128_t) * SSTable_kv_capacity);
                read_sst.close();
                if (cursor + 1 < config->SSTable_count) {
                    if (bg_run[i].first_widpid[cursor + 1] != wp) {               //must shut down in this cursor
                        cout<<"case 1"<<endl;
                        uint index = 0;
                        while (index <= SSTable_kv_capacity - 1) {
                            temp_pid = get_key_pid(bg_run[i].sst[cursor].keys[index]) ;
                            if (temp_pid == pid) {
                                disk_find_count++;
                                //cout << bg_run[i].sst[cursor].keys[index] << endl;
                                if(search_multi){
                                    search_multi_pid[search_multi_length] = get_key_pid(bg_run[i].sst[cursor].keys[index]) ;
                                    search_multi_length++;
                                }
                            } else break;
                            index++;
                        }
                        break;
                    }
                    if (bg_run[i].first_widpid[cursor + 1] == wp) {               //mustn't shut down in this cursor
                        for (uint j = 0; j < SSTable_kv_capacity; j++) {
                            //cout << bg_run[i].sst[cursor].keys[j] << endl;
                            if(search_multi){
                                search_multi_pid[search_multi_length] = get_key_pid(bg_run[i].sst[cursor].keys[j]) ;
                                search_multi_length++;
                            }
                        }
                    }
                    cursor++;
                } else {                                           // cursor is the last one, same too bg_run->first_widpid[cursor+1]!=pid
                    uint index = 0;
                    while (index <= SSTable_kv_capacity - 1) {
                        temp_pid = get_key_pid(bg_run[i].sst[cursor].keys[index]);
                        cout<<"temp_pid: "<<temp_pid<<endl;
                        if (temp_pid == pid) {
                            //cout << bg_run[i].sst[cursor].keys[index] << endl;
                            if(search_multi){
                                search_multi_pid[search_multi_length] = get_key_pid(bg_run[i].sst[cursor].keys[index]);
                                search_multi_length++;
                            }
                        } else break;
                        index++;
                    }
                    break;
                }
            }
            ret = true;
        }
    }
    //cout<<"finish disk search "<<pid<<endl;
    return ret;
}

box workbench::parse_to_real_mbr(unsigned short first_low, unsigned short first_high, uint64_t value) {
//    uint first_low0, first_low1, first_high0, first_high1;
//    d2xy(FIRST_HILBERT_BIT/2, first_low, first_low0, first_low1);
//    d2xy(FIRST_HILBERT_BIT/2, first_high, first_high0, first_high1);
//    double float_first_low0 = (double)first_low0/(pow(2,WID_BIT/2) - 1)*(mbr.high[0] - mbr.low[0]) + mbr.low[0];
//    double float_first_low1 = (double)first_low1/(pow(2,WID_BIT/2) - 1)*(mbr.high[1] - mbr.low[1]) + mbr.low[1];
//    double float_first_high0 = (double)first_high0/(pow(2,WID_BIT/2) - 1)*(mbr.high[0] - mbr.low[0]) + mbr.low[0];
//    double float_first_high1 = (double)first_high1/(pow(2,WID_BIT/2) - 1)*(mbr.high[1] - mbr.low[1]) + mbr.low[1];
//    box first(float_first_low0, float_first_low1, float_first_high0, float_first_high1);
//    cerr<<"first\n";
//    first.print();
//
//    uint second_low, second_high;
//    second_low = get_value_mbr_low(value);
//    second_high = get_value_mbr_high(value);
//    uint second_low0, second_low1, second_high0, second_high1;
//    d2xy(FIRST_HILBERT_BIT/2, second_low, second_low0, second_low1);
//    d2xy(FIRST_HILBERT_BIT/2, second_high, second_high0, second_high1);
//    box ret;
//    ret.low[0] = (double)second_low0/15*(float_first_high0 - float_first_low0) + float_first_low0;
//    ret.low[1] = (double)second_low1/15*(float_first_high1 - float_first_low1) + float_first_low1;
//    ret.high[0] = (double)second_high0/15*(float_first_high0 - float_first_low0) + float_first_low0;
//    ret.high[1] = (double)second_high1/15*(float_first_high1 - float_first_low1) + float_first_low1;
//    ret.print();
    box ret;
    return ret;
}

bool workbench::mbr_search_in_disk(box b, uint timestamp) {
    assert(mbr.contain(b));
    cout << "mbr disk search" << endl;
    uint find_count = 0;
    unordered_set<uint> uni;
    bool ret = false, find = false;
    box bit_b;
    uint bit_pos = 0;
    for (int i = 0; i < big_sorted_run_count; i++) {
        if ((bg_run[i].start_time_min < timestamp) && (timestamp < bg_run[i].end_time_max)) {
            if(!bg_run[i].sst){
                bg_run[i].sst = new SSTable[config->SSTable_count];
            }
            cout << "in bg_run" << i << endl;
            bit_b = bit_box(b);
            cout<<bit_b.low[0]<<","<<bit_b.low[1]<<","<<bit_b.high[0]<<","<<bit_b.high[1]<<endl;
            for (uint j = 0; j < config->SSTable_count; j++) {
                //bg_run[i].bitmap_mbrs[j].print();
                if(b.intersect(bg_run[i].bitmap_mbrs[j])) {     //real box intersect
                    find = false;
                    for (uint p = bit_b.low[0]-1; (p <= bit_b.high[0]) && (!find); p++) {
                        for (uint q = bit_b.low[1]-1; (q <= bit_b.high[1]) && (!find); q++) {
                            bit_pos = xy2d(WID_BIT/2, p, q);
                            if (bg_run[i].bitmaps[j * (bit_count / 8) + bit_pos / 8] & (1 << (bit_pos % 8))) {              //mbr intersect bitmap
                                //cout << "SSTable_" << j << "bit_pos" << bit_pos << endl;
                                find = true;
                                ret = true;
                                break;
                            }
                        }
                    }
                    if(find){
                        if(!bg_run[i].sst[j].keys){
                            ifstream read_sst;
                            string filename = "../store/SSTable_"+to_string(i)+"-"+to_string(j);
                            cout<<filename<<endl;
                            read_sst.open(filename);                   //final place is not high+1, but high
                            assert(read_sst.is_open());
                            bg_run[i].sst[j].keys = new __uint128_t[SSTable_kv_capacity];
                            read_sst.read((char *)bg_run[i].sst[j].keys,sizeof(__uint128_t)*SSTable_kv_capacity);
                            //cout<<"read right"<<endl;
                            read_sst.close();
                        }
                        for(uint q = 0; q < SSTable_kv_capacity; q++){
                            uint pid = get_key_pid(bg_run[i].sst[j].keys[q]);
                            //box value_box = parse_to_real_mbr(bg_run[i].wids[2 * pid], bg_run[i].wids[2 * pid + 1], bg_run[i].sst[j].kv[q].value);
                            box key_box;
                            parse_mbr(bg_run[i].sst[j].keys[q], key_box, bg_run[i].bitmap_mbrs[j]);
                            if(b.intersect(key_box)){
                                uni.insert(pid);
                                find_count++;
                                //cout<<"box find!"<<endl;
                                //key_box.print();

                            }
                        }
                    }
                }
            }
        }
    }
    cout<<"disk_find_count:"<<find_count<<"kv_restriction:"<<config->kv_restriction<<endl;
    cout<<"uni.size():"<<uni.size()<<"num_objects:"<<config->num_objects<<endl;
    uni.clear();
    return ret;
}