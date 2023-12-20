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

    size = config->MemTable_capacity * sizeof(__uint128_t *);                 //sort
    h_keys = (__uint128_t **)allocate(size);
    //log("\t%.2f MB\tmeeting bucket space",size/1024.0/1024.0);

    size = config->MemTable_capacity * sizeof(uint *);
    h_values = (uint **)allocate(size);

    size = config->MemTable_capacity * sizeof(box *);
    h_box_block = (box **)allocate(size);

    for(int i=0;i<config->MemTable_capacity; i++){
        size = config->kv_capacity*sizeof(__uint128_t);
        h_keys[i] = (__uint128_t *)allocate(size);
        log("\t%.2f MB\ta element of h_keys",size/1024.0/1024.0);

        size = config->kv_capacity*sizeof(uint);
        h_values[i] = (uint *)allocate(size);
        log("\t%.2f MB\ta element of h_values",size/1024.0/1024.0);

        size = config->kv_capacity*sizeof(box);
        h_box_block[i] = (box *)allocate(size);
        log("\t%.2f MB\ta element of h_values",size/1024.0/1024.0);
    }

    size = config->search_list_capacity*sizeof(search_info_unit);                   //search
    search_list = (search_info_unit *)allocate(size);

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

}

bool workbench::search_memtable(uint pid){
    cout<<"into search_memtable"<<endl;
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
        box * temp_box;
        while (low <= high) {
            mid = (low + high) / 2;
            temp_pid = h_keys[offset+i][mid]/ 100000000 / 100000000 / 100000000;
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
            temp_pid = h_keys[offset+i][cursor]/ 100000000 / 100000000 / 100000000;
        }
        if (temp_pid == pid && cursor == 0) {
            print_128(h_keys[offset+i][0]);
            temp_box = &h_box_block[offset+i][h_values[offset+i][0]];
            cout<< ": "<< temp_box->low[0] << endl;
        }
        while (cursor + 1 < config->kv_restriction) {
            cursor++;
            temp_pid = h_keys[offset+i][cursor]/ 100000000 / 100000000 / 100000000;
            if (temp_pid == pid) {
                print_128(h_keys[offset+i][cursor]);
                temp_box = &h_box_block[offset+i][h_values[offset+i][cursor]];
                cout<< ": "<< temp_box->low[0] << endl;
            }
        }
        cout << "find !" << endl;
    }
    return ret;
}
