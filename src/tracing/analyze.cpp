/*
 * analyze.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"

#define OVERFLOW_THRESHOLD 0.997

void workbench::analyze_grids(){
	uint overflow = 0;
	uint max_one = 0;
	uint *gridc = new uint[grid_capacity];
	uint total = 0;
	uint total_schema = 0;
	for(int i=0;i<grid_capacity;i++){
		gridc[i] = 0;
	}

	double mean = config->num_objects/this->grids_stack_index;
	double dev = 0.0;

	uint level_count[30];
	for(int i=0;i<30;i++){
		level_count[i] = 0;
	}
	for(int i=0;i<schema_stack_capacity;i++){
		if(schema[i].type==LEAF){
			uint gid = schema[i].grid_id;
			uint gsize = grid_counter[gid];

			assert(schema[i].level<30);
			level_count[schema[i].level]+=gsize;
			dev += (gsize-mean)*(gsize-mean);
			// todo increase the actual capacity
			if(gsize>config->grid_capacity){
				overflow++;
			}
			gridc[gsize>=grid_capacity?(grid_capacity-1):gsize]++;
			total++;
			if(max_one<gsize){
				max_one = gsize;
			}
		}
		total_schema += (schema[i].type!=INVALID);
	}

	pro.max_schema_num = max(pro.max_schema_num, total_schema);
	pro.max_grid_num = max(pro.max_grid_num, total);

	double cum = 0;
	int capacity = 0;
	for(int i=0;i<grid_capacity;i++){
		cum += 1.0*gridc[i]/total;
		printf("%d\t%d\n",i,gridc[i]);
		if(cum>OVERFLOW_THRESHOLD){
			if(pro.max_grid_size<i){
				pro.max_grid_size = i;
			}
			capacity = i;
			break;
		}
	}
	pro.grid_count += grids_stack_index;
	pro.grid_overflow += overflow;
	pro.grid_overflow_list.push_back(100.0*overflow/grids_stack_index);
	pro.grid_deviation_list.push_back(sqrt(dev/grids_stack_index));
	pro.grid_dev += sqrt(dev/grids_stack_index);
	log("%d/%d overflow %d max",overflow,grids_stack_index,max_one);
//	uint total_compute = 0;
//	for(int i=0;i<30;i++){
//		if(level_count[i]>0){
//			total_compute += level_count[i]*i;
//			printf("%d\t%d\t%f\n",i,level_count[i],100.0*level_count[i]/config->num_objects);
//		}
//	}
//	//printf("%f\n",total_compute*100.0/(14*config->num_objects));
}

void workbench::analyze_reaches(){

	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	uint total = 0;
	for(size_t i=0;i<config->num_meeting_buckets;i++){
		if(!meeting_buckets[i].isEmpty()){
			pair<uint, uint> pids = InverseCantorPairing1(meeting_buckets[i].key);
			unit_count[pids.first]++;
			unit_count[pids.second]++;
		}
	}
	uint max_one = 0;
	for(int i=0;i<config->num_objects;i++){
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}

	uint *counter = new uint[unit_count[max_one]+1];
	memset(counter,0,(unit_count[max_one]+1)*sizeof(uint));

	for(int i=0;i<config->num_objects;i++){
		counter[unit_count[i]]++;
	}
	double cum_portion = 0;
	for(int i=0;i<=unit_count[max_one];i++){
		cum_portion += 1.0*counter[i]/config->num_objects;
		log("%d\t%d\t%.3f",i,counter[i],cum_portion);
	}
	delete counter;

	vector<Point *> max_reaches;
	for(size_t i=0;i<config->num_meeting_buckets;i++){
		if(!meeting_buckets[i].isEmpty()){
			pair<uint, uint> pids = InverseCantorPairing1(meeting_buckets[i].key);
			if(pids.second==max_one){
				max_reaches.push_back(points+pids.first);
			}
			if(pids.first==max_one){
				max_reaches.push_back(points+pids.second);
			}
		}
	}

	vector<Point *> all_points;
	vector<Point *> valid_points;
	Point *p1 = points + max_one;
	vector<uint> nodes;
	lookup_rec(schema, p1, 0, nodes, config->reach_distance, true);

	for(uint n:nodes){
		schema[n].mbr.print();
		uint gid = schema[n].grid_id;
		uint *cur_pid = get_grid(gid);
		for(uint i=0;i<get_grid_size(gid);i++){
			Point *p2 = points+cur_pid[i];
			if(p1==p2){
				continue;
			}
			all_points.push_back(p2);
			double dist = p1->distance(*p2,true);
			if(dist<=config->reach_distance){
				valid_points.push_back(p2);
			}
		}
	}

	p1->print();
	print_points(max_reaches);
	print_points(valid_points);
	print_points(all_points);

	log("point %d has %d contacts in result, %ld checked, %ld validated"
			,max_one,unit_count[max_one],all_points.size(), valid_points.size());
	max_reaches.clear();
	all_points.clear();
	valid_points.clear();
	delete []unit_count;
}




void workbench::print_profile(){

	//
	//
//		printf("overflow rate:\n");
//		for(double o:pro.grid_overflow_list){
//			printf("%f\n", o);
//		}
//
//		printf("deviation:\n");
//		for(double o:pro.grid_deviation_list){
//			printf("%f\n", o);
//		}

	fprintf(stderr,"memory space:\n");
	fprintf(stderr,"\tgrid buffer:\t%ld MB\n",pro.max_grid_num*pro.max_grid_size*sizeof(uint)/1024/1024);
	fprintf(stderr,"\tschema list:\t%ld MB\n",pro.max_schema_num*sizeof(QTSchema)/1024/1024);

	fprintf(stderr,"\tfilter list:\t%ld MB\n",pro.max_filter_size*2*sizeof(uint)/1024/1024);
	fprintf(stderr,"\trefine list:\t%ld MB\n",pro.max_refine_size*sizeof(checking_unit)/1024/1024);
	fprintf(stderr,"\tmeeting table:\t%ld MB\n",pro.max_bucket_num*sizeof(meeting_unit)/1024/1024);
	fprintf(stderr,"\tstack size:\t%ld MB\n",tmp_space_capacity*sizeof(meeting_unit)/1024/1024);

	if(pro.rounds>0){
		fprintf(stderr,"time cost:\n");
		fprintf(stderr,"\tcopy data:\t%.2f\n",pro.copy_time/pro.rounds);
		fprintf(stderr,"\tpartition:\t%.2f\n",pro.partition_time/pro.rounds);
		fprintf(stderr,"\tfiltering:\t%.2f\n",pro.filter_time/pro.rounds);
		fprintf(stderr,"\trefinement:\t%.2f\n",pro.refine_time/pro.rounds);
		fprintf(stderr,"\tupdate meets:\t%.2f\n",pro.meeting_identify_time/pro.rounds);
		fprintf(stderr,"\tupdate index:\t%.2f\n",pro.index_update_time/pro.rounds);
		fprintf(stderr,"\toverall:\t%.2f\n",(pro.copy_time+pro.partition_time+pro.filter_time+pro.refine_time+pro.meeting_identify_time+pro.index_update_time)/pro.rounds);


		fprintf(stderr,"statistics:\n");
		fprintf(stderr,"\tnum pairs:\t%.2f \n",2.0*(pro.num_pairs/pro.rounds)/config->num_objects);
		fprintf(stderr,"\tnum meetings:\t%ld \n",pro.num_meetings/pro.rounds);
		fprintf(stderr,"\tusage rate:\t%.2f%% (%ld/%ld)\n",100.0*(pro.max_bucket_num)/config->num_meeting_buckets,pro.max_bucket_num,config->num_meeting_buckets);
		fprintf(stderr,"\t80 usage:\t%ld\n",(size_t)(pro.max_bucket_num/0.8));
		fprintf(stderr,"\toverflow:\t%.4f\n",100.0*pro.grid_overflow/pro.grid_count);
		fprintf(stderr,"\tmean:\t%.2f\n",(double)config->num_objects*pro.rounds/pro.grid_count);
		fprintf(stderr,"\tdeviation:\t%.4f\n",pro.grid_dev/pro.grid_deviation_list.size());
	}

	printf("grid_buffer\tschema\tfilter_list\trefine_list\tmeeting_table\tstack_size\t");
	printf("copy\tpartition\tfiltering\trefinement\tidentify\tupdate_index\t");
	printf("num_contacts\tnum_meetings\toverflow\tmean\tdeviation\n");

	printf("%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t",pro.max_grid_num*pro.max_grid_size*sizeof(uint)/1024/1024
													   ,pro.max_schema_num*sizeof(QTSchema)/1024/1024
													   ,pro.max_filter_size*2*sizeof(uint)/1024/1024
													   ,pro.max_refine_size*sizeof(checking_unit)/1024/1024
													   ,pro.max_bucket_num*sizeof(meeting_unit)/1024/1024
													   ,tmp_space_capacity*sizeof(meeting_unit)/1024/1024);

	printf("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t",pro.copy_time/pro.rounds
										   ,pro.partition_time/pro.rounds
										   ,pro.filter_time/pro.rounds
										   ,pro.refine_time/pro.rounds
										   ,pro.meeting_identify_time/pro.rounds
										   ,pro.index_update_time/pro.rounds);

	printf("%.2f\t%ld\t%.4f\t%.2f\t%.4f\n",2.0*(pro.num_pairs/pro.rounds)/config->num_objects
							  ,pro.num_meetings/pro.rounds
							  ,100.0*pro.grid_overflow/pro.grid_count
							  ,(double)config->num_objects*pro.rounds/pro.grid_count
							  ,pro.grid_dev/pro.grid_deviation_list.size());


	// bucket number
	//printf("%ld\t%.2f\t%.4f\n",2*pro.max_bucket_size*config->num_meeting_buckets*sizeof(meeting_unit)/1024/1024,pro.meeting_update_time/pro.rounds,pro.meet_coefficient/pro.rounds);

	// grid size
	//printf("%.2f\t%.2f\t%.2f\t%ld\t%.4f\n",pro.filter_time/pro.rounds,pro.refine_time/pro.rounds,pro.index_update_time/pro.rounds,pro.
	//		max_filter_size*sizeof(checking_unit)/1024/1024,100.0*pro.grid_overflow/pro.grid_count);

	// minimum distance
//	printf("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",pro.filter_time/pro.rounds,pro.refine_time/pro.rounds,pro.index_update_time/pro.rounds,
//			pro.meeting_update_time/pro.rounds,overall/pro.rounds,2.0*(pro.num_pairs/pro.rounds)/config->num_objects);

	// minimum duration
	//printf("%.2f\t%.2f\t%ld\n",pro.copy_time/pro.rounds,pro.meeting_update_time/pro.rounds,pro.num_meetings/pro.rounds);

}


