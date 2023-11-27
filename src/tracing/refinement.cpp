/*
 * reachability.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */

#include "workbench.h"

void *reachability_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];
	Point *points = bench->points;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = bench->grid_check[pairid].pid;
			uint gid = bench->grid_check[pairid].gid;

			uint size = bench->get_grid_size(gid);
			uint *cur_pids = bench->get_grid(gid);

			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				if(pid<cur_pids[i]||!bench->grid_check[pairid].inside){
					Point *p2 = points + cur_pids[i];
					//p2->print();
					if(p1->distance(p2, true)<=ctx->config->reach_distance){
						uint pid1 = min(pid,cur_pids[i]);
						uint pid2 = max(cur_pids[i],pid);
						size_t key = cantorPairing(pid1,pid2);
						size_t slot = key%bench->config->num_meeting_buckets;
						//printf("%ld %ld %ld\n",slot,key,kHashTableCapacity);
						while (true){
							if(bench->meeting_buckets[slot].key==key){
								bench->meeting_buckets[slot].end = bench->cur_time;
								break;
							}else if(bench->meeting_buckets[slot].key==ULL_MAX){
								bench->lock(slot);
								bool inserted = (bench->meeting_buckets[slot].key==ULL_MAX);
								bench->unlock(slot);
								if(inserted){
									bench->meeting_buckets[slot].key = key;
									bench->meeting_buckets[slot].start = bench->cur_time;
									bench->meeting_buckets[slot].end = bench->cur_time;
									break;
								}
							}
							slot = (slot + 1)%bench->config->num_meeting_buckets;
						}

					}//distance
				}//
			}// grid size
		}// pairs
	}

	return NULL;
}

void workbench::reachability(){

	query_context tctx;
	tctx.config = config;
	tctx.num_units = grid_check_counter;
	tctx.target[0] = (void *)this;

	// generate a new batch of reaches
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, reachability_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	//bench->grid_check_counter = 0;
	logt("reachability compute",start);
}



