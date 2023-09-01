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
//                        box temp;
//                        temp.update(*p1);
//                        temp.update(*p2);
//                        //temp.print();
//						while (true){
//							if(bench->meeting_buckets[slot].key==key){
//								bench->meeting_buckets[slot].end = bench->cur_time;
//                                //assert(bench->meeting_buckets[slot].mbr!=NULL);
//                                bench->meeting_buckets[slot].mbr->update(temp);
//								break;
//							}else if(bench->meeting_buckets[slot].key==ULL_MAX){
//								bench->lock(slot);
//								bool inserted = (bench->meeting_buckets[slot].key==ULL_MAX);
//								bench->unlock(slot);
//								if(inserted){
//									bench->meeting_buckets[slot].key = key;
//									bench->meeting_buckets[slot].start = bench->cur_time;
//									bench->meeting_buckets[slot].end = bench->cur_time;
//                                    //assert(bench->meeting_buckets[slot].mbr==NULL);
//                                    *(bench->meeting_buckets[slot].mbr) = temp;
//									break;
//								}
//							}
//							slot = (slot + 1)%bench->config->num_meeting_buckets;
//						}
                        box *temp = new box;
                        temp->update(*p1);
                        temp->update(*p2);
                        //temp.print();
                        while (true){
                            if(bench->meeting_buckets[slot].key==key){
                                bench->meeting_buckets[slot].end = bench->cur_time;
                                //assert(bench->meeting_buckets[slot].mbr!=NULL);
                                bench->meeting_buckets[slot].mbr.update(*temp);
                                break;
                            }else if(bench->meeting_buckets[slot].key==ULL_MAX){
                                bench->lock(slot);
                                bool inserted = (bench->meeting_buckets[slot].key==ULL_MAX);
                                bench->unlock(slot);
                                if(inserted){
                                    bench->meeting_buckets[slot].key = key;
                                    bench->meeting_buckets[slot].start = bench->cur_time;
                                    bench->meeting_buckets[slot].end = bench->cur_time;
                                    //assert(bench->meeting_buckets[slot].mbr==NULL);
                                    bench->meeting_buckets[slot].mbr = *temp;
                                    break;
                                }
                            }
                            slot = (slot + 1)%bench->config->num_meeting_buckets;
                        }
                        delete temp;


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






//#include "workbench.h"
//
//void *reachability_unit(void *arg){
//    query_context *ctx = (query_context *)arg;
//    workbench *bench = (workbench *)ctx->target[0];
//    Point *points = bench->points;
//
//    // pick one batch of point-grid pair for processing
//    size_t start = 0;
//    size_t end = 0;
//    while(ctx->next_batch(start,end)){
//        for(uint pairid=start;pairid<end;pairid++){
//            uint pid = bench->grid_check[pairid].pid;
//            uint gid = bench->grid_check[pairid].gid;
//
//            uint size = bench->get_grid_size(gid);
//            uint *cur_pids = bench->get_grid(gid);
//
//            //vector<Point *> pts;
//            Point *p1 = points + pid;
//            for(uint i=0;i<size;i++){
//                //pts.push_back(points + cur_pids[i]);
//                if(pid<cur_pids[i]||!bench->grid_check[pairid].inside){
//                    Point *p2 = points + cur_pids[i];
//                    //p2->print();
//                    if(p1->distance(p2, true)<=ctx->config->reach_distance){
//                        uint pid1 = min(pid,cur_pids[i]);
//                        uint pid2 = max(cur_pids[i],pid);
//                        size_t key = cantorPairing(pid1,pid2);
//                        size_t slot = key%bench->config->num_meeting_buckets;
//                        //Point mid = Point((p1->x+p2->x)/2 , (p1->y+p2->y)/2);
//                        //printf("%ld %ld %ld\n",slot,key,kHashTableCapacity);
//
//                        meeting_unit test;
//                        double low0 = test.mbr.low[0];
//                        fprintf(stdout, "变量meeting_unit low0: %lf\n",low0);
//                        if(low0>=(-1e-6)&&low0<=(1e-6)){
//                            fprintf(stdout, "this is the zero\n");
//                        }
//                        double low1 = test.mbr.low[1];
//                        fprintf(stdout, "变量meeting_unit low1: %lf\n",low1);
//                        if(low1>=(-1e-6)&&low1<=(1e-6)){
//                            fprintf(stdout, "this is the zero\n");
//                        }
//                        while (true){
//                            if(bench->meeting_buckets[slot].key==key){
//                                bench->meeting_buckets[slot].end = bench->cur_time;
//                                if(p1->x>=(-1e-6)&&p1->x<=(1e-6)){
//                                    fprintf(stdout, "p1->x==0\n");
//                                }
//                                if(p2->x>=(-1e-6)&&p2->x<=(1e-6)){
//                                    fprintf(stdout, "p2->x==0\n");
//                                }
//                                if(p1->y>=(-1e-6)&&p1->y<=(1e-6)){
//                                    fprintf(stdout, "p1->y==0\n");
//                                }
//                                if(p2->y>=(-1e-6)&&p2->y<=(1e-6)){
//                                    fprintf(stdout, "p2->y==0\n");
//                                }
//                                if(bench->meeting_buckets[slot].mbr.low[0]>p1->x){
//                                    bench->meeting_buckets[slot].mbr.low[0] = p1->x;
//                                }
//                                if(bench->meeting_buckets[slot].mbr.high[0]<p1->x){
//                                    bench->meeting_buckets[slot].mbr.high[0] = p1->x;
//                                }
//
//                                if(bench->meeting_buckets[slot].mbr.low[1]>p1->y){
//                                    bench->meeting_buckets[slot].mbr.low[1] = p1->y;
//                                }
//                                if(bench->meeting_buckets[slot].mbr.high[1]<p1->y){
//                                    bench->meeting_buckets[slot].mbr.high[1] = p1->y;
//                                }
//                                if(bench->meeting_buckets[slot].mbr.low[0]>p2->x){
//                                    bench->meeting_buckets[slot].mbr.low[0] = p2->x;
//                                }
//                                if(bench->meeting_buckets[slot].mbr.high[0]<p2->x){
//                                    bench->meeting_buckets[slot].mbr.high[0] = p2->x;
//                                }
//
//                                if(bench->meeting_buckets[slot].mbr.low[1]>p2->y){
//                                    bench->meeting_buckets[slot].mbr.low[1] = p2->y;
//                                }
//                                if(bench->meeting_buckets[slot].mbr.high[1]<p2->y){
//                                    bench->meeting_buckets[slot].mbr.high[1] = p2->y;
//                                }
//                                //bench->meeting_buckets[slot].mbr.update(*p1);
//                                //bench->meeting_buckets[slot].mbr.update(*p2);
//                                double low1 = bench->meeting_buckets[slot].mbr.low[1];
//                                double high0 = bench->meeting_buckets[slot].mbr.high[0];
//                                if(low1>=(-1e-6)&&low1<=(1e-6)){
//                                    //bench->meeting_buckets[slot].mbr.print();
//                                    fprintf(stdout, "low1==0\n");
//                                }
//                                if(high0>=(-1e-6)&&high0<=(1e-6)){
//                                    fprintf(stdout, "high0==0\n");
//                                }
//                                //
//                                break;
//                            }else if(bench->meeting_buckets[slot].key==ULL_MAX){
//                                bench->lock(slot);
//                                bool inserted = (bench->meeting_buckets[slot].key==ULL_MAX);
//                                bench->unlock(slot);
//                                if(inserted){
//                                    bench->meeting_buckets[slot].key = key;
//                                    bench->meeting_buckets[slot].start = bench->cur_time;
//                                    bench->meeting_buckets[slot].end = bench->cur_time;
//
//                                    double low1 = bench->meeting_buckets[slot].mbr.low[1];
//                                    fprintf(stdout, "low0: %lf\n",bench->meeting_buckets[slot].mbr.low[0]);
//                                    fprintf(stdout, "low1: %lf\n",low1);
//                                    fprintf(stdout, "high0: %lf\n",bench->meeting_buckets[slot].mbr.high[0]);
//                                    fprintf(stdout, "high1: %lf\n",bench->meeting_buckets[slot].mbr.high[1]);
//
////                                    if(low1>=(-1e-6)&&low1<=(1e-6)){
////                                        fprintf(stdout, "this is the zero\n");
////                                    }
//
//                                    if(bench->meeting_buckets[slot].mbr.low[0]>p1->x){
//                                        bench->meeting_buckets[slot].mbr.low[0] = p1->x;
//                                    }
//                                    if(bench->meeting_buckets[slot].mbr.high[0]<p1->x){
//                                        bench->meeting_buckets[slot].mbr.high[0] = p1->x;
//                                    }
//
//                                    if(bench->meeting_buckets[slot].mbr.low[1]>p1->y){
//                                        bench->meeting_buckets[slot].mbr.low[1] = p1->y;
//                                    }
//                                    if(bench->meeting_buckets[slot].mbr.high[1]<p1->y){
//                                        bench->meeting_buckets[slot].mbr.high[1] = p1->y;
//                                    }
//                                    if(bench->meeting_buckets[slot].mbr.low[0]>p2->x){
//                                        bench->meeting_buckets[slot].mbr.low[0] = p2->x;
//                                    }
//                                    if(bench->meeting_buckets[slot].mbr.high[0]<p2->x){
//                                        bench->meeting_buckets[slot].mbr.high[0] = p2->x;
//                                    }
//
//                                    if(bench->meeting_buckets[slot].mbr.low[1]>p2->y){
//                                        bench->meeting_buckets[slot].mbr.low[1] = p2->y;
//                                    }
//                                    if(bench->meeting_buckets[slot].mbr.high[1]<p2->y){
//                                        bench->meeting_buckets[slot].mbr.high[1] = p2->y;
//                                    }
//                                    //bench->meeting_buckets[slot].mbr.print();
//                                    break;
//                                }
//                            }
//                            slot = (slot + 1)%bench->config->num_meeting_buckets;
//                        }
//
//                    }//distance
//                }//
//            }// grid size
//        }// pairs
//    }
//
//    return NULL;
//}
//
//void workbench::reachability(){
//
//    query_context tctx;
//    tctx.config = config;
//    tctx.num_units = grid_check_counter;
//    tctx.target[0] = (void *)this;
//
//    // generate a new batch of reaches
//    struct timeval start = get_cur_time();
//    pthread_t threads[tctx.config->num_threads];
//
//    for(int i=0;i<tctx.config->num_threads;i++){
//        pthread_create(&threads[i], NULL, reachability_unit, (void *)&tctx);
//    }
//    for(int i = 0; i < tctx.config->num_threads; i++ ){
//        void *status;
//        pthread_join(threads[i], &status);
//    }
//    //bench->grid_check_counter = 0;
//    logt("reachability compute",start);
//}





