/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"


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

    size = config->MemTable_capacity * sizeof(uint64_t *);                 //sort
    h_keys = (uint64_t **)allocate(size);
    //log("\t%.2f MB\tmeeting bucket space",size/1024.0/1024.0);

    size = config->MemTable_capacity * sizeof(__uint128_t *);
    h_values = (__uint128_t **)allocate(size);

    for(int i=0;i<config->MemTable_capacity; i++){
        size = config->kv_capacity*sizeof(uint64_t);
        h_keys[i] = (uint64_t *)allocate(size);
        log("\t%.2f MB\ta element of h_keys",size/1024.0/1024.0);

        size = config->kv_capacity*sizeof(__uint128_t);
        h_values[i] = (__uint128_t *)allocate(size);
        log("\t%.2f MB\ta element of h_values",size/1024.0/1024.0);
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
            log("\t%.2f MB\ta pstFilter", size / 1024.0 / 1024.0);
            memset(pstFilter[i], 0, dwFilterSize);
        }
    }

    size = config->big_sorted_run_capacity*sizeof(sorted_run);
    bg_run = (sorted_run *)allocate(size);
    for(int i=0;i<config->big_sorted_run_capacity;i++){
        bg_run[i].SSTable_count = config->SSTable_count;
    }

}

bool workbench::search_memtable(uint pid){
    cout<<"memtable search"<<pid<<endl;
    uint offset = 0;
    if(big_sorted_run_count%2==1){
        offset = config->MemTable_capacity/2;
    }
    bool ret = false;
    for(int i=0;i<MemTable_count;i++) {

        int find = -1;
        int low = 0;
        int high = config->kv_restriction - 1;
        int mid;
        uint temp_pid;
        //box temp_box;
        while (low <= high) {
            mid = (low + high) / 2;
            temp_pid = h_keys[offset+i][mid] >> 39;
            if (temp_pid == pid) {
                find = mid;
                ret = true;
                break;
            } else if (temp_pid > pid) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        if (find == -1) {
            cout << "cannot find" << endl;
            break;
        }
        cout << "exactly find" << endl;
        uint cursor = find;
        while (temp_pid == pid && cursor >= 1) {
            cursor--;
            temp_pid = h_keys[offset+i][cursor] >> 39;
        }
        if (temp_pid == pid && cursor == 0) {
            cout<<h_keys[offset+i][0];
            cout<<" :"<<(uint)(h_values[offset+i][0] >> 112)<<endl;
            box temp_box(h_values[offset+i][0]);
            temp_box.print();
            if(search_multi){
                search_multi_pid[search_multi_length] = (h_keys[offset+i][0] >> 14) & ((1ULL << 25) - 1);
                search_multi_length++;
            }
        }
        while (cursor + 1 < config->kv_restriction) {
            cursor++;
            temp_pid = h_keys[offset+i][cursor] >> 39;
            if (temp_pid == pid) {
                cout<<h_keys[offset+i][cursor]<<endl;
                cout<<" :"<<(uint)(h_values[offset+i][cursor] >> 112)<<endl;
                box temp_box(h_values[offset+i][cursor]);
                temp_box.print();
                if(search_multi) {
                    search_multi_pid[search_multi_length] = (h_keys[offset + i][cursor] >> 14) & ((1ULL << 25) - 1);
                    search_multi_length++;
                }
            }
        }
    }
    return ret;
}

bool workbench::search_in_disk(uint pid, uint timestamp){
    cout<<"disk search "<<pid<<endl;
    bool ret = false;
    for(int i=0;i<big_sorted_run_count;i++) {
        if ((bg_run[i].timestamp_min < timestamp)) {
            //(bg_run[i].timestamp_min < timestamp) && (timestamp < bg_run[i].timestamp_max)
            cout<<"into disk"<<endl;
            bg_run[i].sst = new SSTable[bg_run[i].SSTable_count];                   //maybe useful later, should not delete after this func
            ifstream read_sst;

            //high level binary search
            int find = -1;
            int low = 0;
            int high = bg_run[i].SSTable_count - 1;
            int mid;
            while (low <= high) {
                mid = (low + high) / 2;
                if (bg_run[i].first_pid[mid] == pid){
                    find = mid;
                    break;
                }
                else if (bg_run[i].first_pid[mid] > pid){
                    high = mid - 1;
                }
                else {
                    low = mid + 1;
                }
            }
            if(find==-1){
                cout<<"not find in first_pid"<<endl;
                string filename = "../store/SSTable_"+to_string(i)+"-"+to_string(high);
                cout<<filename<<endl;
                read_sst.open(filename);                   //final place is not high+1, but high
                assert(read_sst.is_open());
                cout<<low<<"-"<<bg_run[i].first_pid[low]<<endl;
                cout<<mid<<"-"<<bg_run[i].first_pid[mid]<<endl;
                cout<<high<<"-"<<bg_run[i].first_pid[high]<<endl;

                bg_run[i].sst[high].kv = new key_value[bg_run[i].sst[high].SSTable_kv_capacity];
                read_sst.read((char *)bg_run[i].sst[high].kv,sizeof(key_value)*bg_run[i].sst[high].SSTable_kv_capacity);
                read_sst.close();
                ret |= bg_run[i].sst[high].search_SSTable(pid,search_multi,search_multi_length,search_multi_pid);
                continue;
            }
            cout<<"high level binary search finish and find"<<endl;

            //for the case, there are many SSTables that first_pid==pid
            //find start and end
            uint pid_start = find;
            while(pid_start>=1){
                pid_start--;
                if(bg_run[i].first_pid[pid_start]!=pid){
                    break;
                }
            }
            read_sst.open("../store/SSTable_"+to_string(i)+"-"+to_string(pid_start));
            assert(read_sst.is_open());
            bg_run[i].sst[pid_start].kv = new key_value[bg_run[i].sst[pid_start].SSTable_kv_capacity];
            read_sst.read((char *)bg_run[i].sst[pid_start].kv,sizeof(key_value)*bg_run[i].sst[pid_start].SSTable_kv_capacity);
            bg_run[i].sst[pid_start].search_SSTable(pid,search_multi,search_multi_length,search_multi_pid);
            read_sst.close();
            uint cursor = pid_start+1;
            uint temp_pid;
            while(true) {
                read_sst.open("../store/SSTable_" + to_string(i) + "-" + to_string(cursor));
                assert(read_sst.is_open());
                bg_run[i].sst[cursor].kv = new key_value[bg_run[i].sst[cursor].SSTable_kv_capacity];
                read_sst.read((char *) bg_run[i].sst[cursor].kv, sizeof(key_value) * bg_run[i].sst[cursor].SSTable_kv_capacity);
                read_sst.close();
                if (cursor + 1 < bg_run[i].SSTable_count) {
                    if (bg_run[i].first_pid[cursor + 1] != pid) {               //must shut down in this cursor
                        uint index = 0;
                        while (index <= bg_run[i].sst[cursor].SSTable_kv_capacity - 1) {
                            temp_pid = bg_run[i].sst[cursor].kv[index].key >> 39;
                            if (temp_pid == pid) {
                                cout << bg_run[i].sst[cursor].kv[index].key << endl;
                                if(search_multi){
                                    search_multi_pid[search_multi_length] = (bg_run[i].sst[cursor].kv[index].key >> 14) & ((1ULL << 25) - 1);
                                    search_multi_length++;
                                }
                            } else break;
                            index++;
                        }
                        break;
                    }
                    if (bg_run[i].first_pid[cursor + 1] == pid) {               //mustn't shut down in this cursor
                        for (uint j = 0; j < bg_run[i].sst[cursor].SSTable_kv_capacity; j++) {
                            cout << bg_run[i].sst[cursor].kv[j].key << endl;
                            if(search_multi){
                                search_multi_pid[search_multi_length] = (bg_run[i].sst[cursor].kv[j].key >> 14) & ((1ULL << 25) - 1);
                                search_multi_length++;
                            }
                        }
                    }
                    cursor++;
                } else {                                           // cursor is the last one, same too bg_run->first_pid[cursor+1]!=pid
                    uint index = 0;
                    while (index <= bg_run[i].sst[cursor].SSTable_kv_capacity - 1) {
                        temp_pid = bg_run[i].sst[cursor].kv[index].key >> 39;
                        if (temp_pid == pid) {
                            cout << bg_run[i].sst[cursor].kv[index].key << endl;
                            if(search_multi){
                                search_multi_pid[search_multi_length] = (bg_run[i].sst[cursor].kv[index].key >> 14) & ((1ULL << 25) - 1);
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
    return ret;
}