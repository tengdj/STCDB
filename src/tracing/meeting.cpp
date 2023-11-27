///*
// * meeting.cpp
// *
// *  Created on: Feb 25, 2021
// *      Author: teng
// */
//
//
//#include "workbench.h"
//
//bool workbench::batch_meet(meeting_unit *buffer, uint num){
//	if(num == 0){
//		return false;
//	}
//	uint cur_counter = 0;
//	lock();
//	cur_counter = meeting_counter;
//	meeting_counter += num;
//	unlock();
//	if(meeting_counter<meeting_capacity){
//		memcpy(meetings+cur_counter,buffer,sizeof(meeting_unit)*num);
//	}
//	return true;
//}
//
///*
// *
// * update the meetings maintained with reachability information collected in this round
// *
// * */
//
//void *update_meetings_unit(void *arg){
//
//	query_context *ctx = (query_context *)arg;
//	workbench *bench = (workbench *)ctx->target[0];
//
//	meeting_unit *meetings = new meeting_unit[200];
//	uint meeting_index = 0;
//
//	size_t start = 0;
//	size_t end = 0;
//
//	uint active_count = 0;
//	uint taken_count = 0;
//
//	while(ctx->next_batch(start,end)){
//		for(size_t bid=start;bid<end;bid++){
//			if(bench->meeting_buckets[bid].isEmpty()){
//				continue;
//			}
//			taken_count++;
//			// still active
//			if(bench->meeting_buckets[bid].end==bench->cur_time){
//				active_count++;
//				continue;
//			}
//
//			//log("%d %d",bench->cur_time,bench->meeting_buckets[bid].start);
//			if(bench->cur_time-bench->meeting_buckets[bid].start>=bench->config->min_meet_time+1){
//				meetings[meeting_index++] = bench->meeting_buckets[bid];
//				if(meeting_index==200){
//					bench->batch_meet(meetings, 200);
//					meeting_index = 0;
//				}
//			}
//			bench->meeting_buckets[bid].reset();
//		}
//	}
//	bench->batch_meet(meetings, meeting_index);
//	delete []meetings;
//	bench->lock();
//	bench->num_active_meetings += active_count;
//	bench->num_taken_buckets += taken_count;
//	bench->unlock();
//	return NULL;
//}
//
//void workbench::update_meetings(){
//	struct timeval start = get_cur_time();
//
//	query_context tctx;
//	tctx.config = config;
//	tctx.target[0] = (void *)this;
//	tctx.num_units = config->num_meeting_buckets;
//	pthread_t threads[tctx.config->num_threads];
//
//	num_active_meetings = 0;
//	num_taken_buckets = 0;
//	meeting_counter = 0;
//
//	for(int i=0;i<tctx.config->num_threads;i++){
//		pthread_create(&threads[i], NULL, update_meetings_unit, (void *)&tctx);
//	}
//	for(int i = 0; i < tctx.config->num_threads; i++ ){
//		void *status;
//		pthread_join(threads[i], &status);
//	}
//
//	logt("update meeting: %d taken, %d active, %d identified",start,
//			num_taken_buckets,num_active_meetings,meeting_counter);
//}
//
