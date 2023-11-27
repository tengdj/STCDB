/*
 * working_bench.h
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_WORKBENCH_H_
#define SRC_TRACING_WORKBENCH_H_
#include "../util/util.h"
#include "../util/config.h"
#include "../util/query_context.h"
#include "../geometry/geometry.h"
#include "../index/QTree.h"

typedef struct profiler{
	double copy_time = 0;
	double partition_time = 0;
	double filter_time = 0;
	double refine_time = 0;
	double meeting_identify_time = 0;
	double index_update_time = 0;
	uint rounds = 0;

	uint max_refine_size = 0;
	uint max_filter_size = 0;
	uint max_grid_size = 0;
	uint max_grid_num = 0;
	uint max_schema_num = 0;
	size_t max_bucket_num = 0;

	uint grid_count = 0;
	uint grid_overflow = 0;
	double grid_dev = 0.0;
	vector<double> grid_overflow_list;
	vector<double> grid_deviation_list;
	size_t num_pairs = 0;
	size_t num_meetings = 0;

    double cuda_sort_time = 0;
}profiler;

typedef struct checking_unit{
	uint pid;
	uint gid;
	unsigned short offset;
	unsigned short inside;
}checking_unit;

typedef struct meeting_unit{
    size_t key;
//    unsigned short start;       //2023.9.21
//    unsigned short end;
    unsigned start;       //2023.9.21
    unsigned end;
    //Point midpoint;            //2023.7.17
    box mbr;                     //7.24 7.26
    bool isEmpty(){
        return key == ULL_MAX;
    }
    void reset(){
        key = ULL_MAX;
    }
    uint get_pid1(){
        return ::InverseCantorPairing1(key).first;
    }
    uint get_pid2(){
        return ::InverseCantorPairing1(key).second;
    }
}meeting_unit;

typedef struct reach_unit{
	uint pid1;
	uint pid2;
}reach_unit;

// the workbench where stores the memory space
// used for processing


class workbench{
	pthread_mutex_t *insert_lk;
	void *data[100];
	size_t data_size[100];
	uint data_index = 0;
	void *allocate(size_t size);
public:
	profiler pro;
	uint test_counter = 0;
	configuration *config = NULL;
	uint cur_time = 0;
	box mbr;

	// the pool of maintaining objects assignment
	// each grid buffer: |point_id1...point_idn|
	uint *grids = NULL;
	uint grid_capacity = 0;
	uint *grid_counter = NULL;

	// the stack that keeps the available grids
	uint *grids_stack = NULL;
	uint grids_stack_capacity = 0;
	uint grids_stack_index = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	// stack that keeps the available schema nodes
	uint *schema_stack = NULL;
	uint schema_stack_capacity = 0;
	uint schema_stack_index = 0;

	uint *part_counter;
	uint *schema_assigned;

	// the space for point-unit pairs
	checking_unit *grid_check = NULL;
	uint grid_check_capacity = 0;
	uint grid_check_counter = 0;


	// the space to store the point-node pairs for filtering
	uint *filter_list = NULL;
	uint filter_list_index = 0;
	uint filter_list_capacity = 0;

	// the space for the overall meeting information maintaining now
	meeting_unit *meeting_buckets = NULL;

	size_t num_taken_buckets = 0;
	size_t num_active_meetings = 0;

//	// the space for the valid meeting information now
//	meeting_unit *meetings = NULL;
//	uint meeting_capacity = 0;
//	uint meeting_counter = 0;

    //for space for cuda sort
    __uint128_t *d_keys = NULL;
    uint *d_values = NULL;
    box *d_box_block = NULL;
    uint kv_capacity = 45000000;
    uint kv_2G = 44739243;
    uint kv_count = 0;

    //space for MemTable
    __uint128_t **h_keys = NULL;
    uint **h_values = NULL;
    box **h_box_block = NULL;
    uint MemTable_capacity = 5;
    uint MemTable_count = 0;

	// the temporary space
	uint *tmp_space = NULL;
	uint tmp_space_capacity = 0;


	uint *merge_list = NULL;
	uint merge_list_index = 0;
	uint *split_list = NULL;
	uint split_list_index = 0;

	// external source
	Point *points = NULL;

	workbench(workbench *bench);
	workbench(configuration *conf);
	~workbench(){};
	void clear();

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);

	// generate pid-gid-offset pairs for processing
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num);
	bool batch_meet(meeting_unit *m, uint num);

	void merge_node(uint cur_node);
	void split_node(uint cur_node);
	void filter();
	void update_schema();
	void reachability();

	void claim_space();
	size_t space_claimed(){
		size_t total = 0;
		for(int i=0;i<100;i++){
			total += data_size[i];
		}
		return total;
	}
	void reset(){
		// reset the number of objects in each grid
		for(int i=0;i<grids_stack_capacity;i++){
			grid_counter[i] = 0;
		}
		grid_check_counter = 0;
	}
	inline uint get_grid_size(uint gid){
		assert(gid<grids_stack_capacity);
		return min(grid_counter[gid],grid_capacity);
	}
	inline uint *get_grid(uint gid){
		assert(gid<grids_stack_capacity);
		return grids + gid*grid_capacity;
	}

	void update_meetings();

	void analyze_grids();
	void analyze_reaches();
	void print_profile();

	void lock(uint key = 0){
		pthread_mutex_lock(&insert_lk[key%MAX_LOCK_NUM]);
	}
	void unlock(uint key = 0){
		pthread_mutex_unlock(&insert_lk[key%MAX_LOCK_NUM]);
	}
};
extern void lookup_rec(QTSchema *schema, Point *p, uint curnode, vector<uint> &gids, double max_dist, bool include_owner = false);
extern void lookup_stack(QTSchema *schema, Point *p, uint curnode, vector<uint> &gids, double max_dist, bool include_owner = false);

#endif /* SRC_TRACING_WORKBENCH_H_ */
