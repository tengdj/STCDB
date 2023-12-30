/*
 * process.cpp
 *
 *  Created on: Feb 11, 2021
 *      Author: teng
 */

#include "trace.h"


/*
 * functions for tracer
 *
 * */



tracer::tracer(configuration *conf, box &b, Point *t, trace_generator *gen){
	trace = t;
	mbr = b;
	config = conf;
	part = new partitioner(mbr,config);
    generator = gen;
#ifdef USE_GPU
	if(config->gpu){
		vector<gpu_info *> gpus = get_gpus();
		if(gpus.size()==0){
			log("no GPU is found, use CPU mode");
			config->gpu = false;
		}else{
			assert(config->specific_gpu<gpus.size());
			gpu = gpus[config->specific_gpu];
			gpu->print();
			for(int i=0;i<gpus.size();i++){
				if(i!=config->specific_gpu){
					delete gpus[i];
				}
			}
			gpus.clear();
		}
	}
#endif
}
tracer::tracer(configuration *conf){
	config = conf;
	loadMeta(config->trace_path.c_str());
	part = new partitioner(mbr,config);
#ifdef USE_GPU
	if(config->gpu){
		vector<gpu_info *> gpus = get_gpus();
		if(gpus.size()==0){
			log("not GPU is found, use CPU mode");
			config->gpu = false;
		}else{
			assert(config->specific_gpu<gpus.size());
			gpu = gpus[config->specific_gpu];
			gpu->print();
			for(int i=0;i<gpus.size();i++){
				if(i!=config->specific_gpu){
					delete gpus[i];
				}
			}
			gpus.clear();
		}
	}
#endif
}
tracer::~tracer(){
	if(owned_trace){
		free(trace);
	}
	if(part){
		delete part;
	}
	if(bench){
		delete bench;
	}
#ifdef USE_GPU
	if(gpu){
		delete gpu;
	}
#endif
}
void tracer::dumpTo(const char *path) {
	struct timeval start_time = get_cur_time();
	ofstream wf(path, ios::out|ios::binary|ios::trunc);
	wf.write((char *)&config->num_objects, sizeof(config->num_objects));
	wf.write((char *)&config->duration, sizeof(config->duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = config->duration*config->num_objects;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

void tracer::loadMeta(const char *path) {

	uint true_num_objects;
	uint true_duration;
	ifstream in(path, ios::in | ios::binary);
	if(!in.is_open()){
		log("%s cannot be opened",path);
		exit(0);
	}
	in.read((char *)&true_num_objects, sizeof(true_num_objects));
	in.read((char *)&true_duration, sizeof(true_duration));
	log("%d objects last for %d seconds in file",true_num_objects,true_duration);
	in.read((char *)&mbr, sizeof(mbr));
	mbr.to_squre(true);
	mbr.print();
	assert((size_t)config->num_objects*(config->start_time+config->duration)<=(size_t)true_num_objects*true_duration);
	//assert(config->num_objects<=true_num_objects);
	assert(config->start_time+config->duration<=true_duration);
	in.close();
}

void tracer::loadData(const char *path, int st, int duration) {

	log("loading locations from %d to %d",st,st+duration);
	assert(duration<=100);
	uint true_num_objects;
	uint true_duration;
	box mbr;
	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	if(!in.is_open()){
		log("%s cannot be opened",path);
		exit(0);
	}
	in.read((char *)&true_num_objects, sizeof(true_num_objects));
	in.read((char *)&true_duration, sizeof(true_duration));
	in.read((char *)&mbr, sizeof(mbr));

	assert((size_t)config->num_objects*(st+duration)<=(size_t)true_num_objects*true_duration);
	//assert(config->num_objects<=true_num_objects);
	assert(st+duration<=true_duration);

	in.seekg(st*true_num_objects*sizeof(Point), ios_base::cur);
	if(!trace){
		trace = (Point *)malloc(min((uint)100,config->duration)*config->num_objects*sizeof(Point));
	}

	uint loaded = 0;
	while(loaded<config->num_objects){
		uint cur_num_objects = min(true_num_objects,config->num_objects-loaded);
		for(int i=0;i<duration;i++){
			in.read((char *)(trace+i*config->num_objects+loaded), cur_num_objects*sizeof(Point));
			if(true_num_objects>cur_num_objects){
				in.seekg((true_num_objects-cur_num_objects)*sizeof(Point), ios_base::cur);
			}
		}
		loaded += cur_num_objects;
	}

	in.close();
	logt("loaded %d objects last for %d seconds start from %d time from %s",start_time, config->num_objects, duration, st, path);
	owned_trace = true;
}

void tracer::print(){
	print_points(trace,config->num_objects,min(config->num_objects,(uint)10000));
}
void tracer::print_trace(int oid){
	vector<Point *> points;
	for(int i=0;i<config->cur_duration;i++){
		points.push_back(trace+i*config->num_objects+oid);
	}
	print_points(points);
	points.clear();
}
void tracer::print_traces(){
	vector<Point *> points;
	for(int oid=0;oid<config->num_objects;oid++){
		for(int i=0;i<config->duration;i++){
			points.push_back(trace+i*config->num_objects+oid);
		}
	}
	print_points(points, 10000);
	points.clear();
}

inline bool BloomFilter_Check(workbench *bench, uint sst, uint pid){
    uint pdwHashPos;
    uint64_t hash1, hash2;
    int ret;
    for (int i = 0; i < bench->dwHashFuncs; i++){
        hash1 = MurmurHash2_x64((const void *)&pid, sizeof(uint), bench->dwSeed);
        hash2 = MurmurHash2_x64((const void *)&pid, sizeof(uint), MIX_UINT64(hash1));
        pdwHashPos = (hash1 + i*hash2) % bench->dwFilterBits;
        ret = bench->pstFilter[sst][pdwHashPos/8] & (1<<(pdwHashPos%8));        //0 1 ... 128 ...
        if(ret == 0)
            return false;
    }
    return true;
}

void *sst_dump(void *arg){
    struct timeval bg_start = get_cur_time();
    cout<<"step into the sst_dump"<<endl;
    workbench *bench = (workbench *)arg;
    cout<<"cur_time: "<<bench->cur_time<<endl;
    uint offset = 0;
    assert(bench->config->MemTable_capacity%2==0);
    uint old_big = bench->big_sorted_run_count;                 //atomic add
    bench->big_sorted_run_count++;
    if(old_big%2==1){
        offset = bench->config->MemTable_capacity/2;
    }
    //bench->bg_run[old_big].SSTable_count = bench->config->SSTable_count;
    bench->bg_run[old_big].first_pid = new uint[bench->bg_run[old_big].SSTable_count];

    //merge sort
    ofstream SSTable_of;
    //bool of_open = false;
    uint kv_count = 0;
    uint sst_count = 0;
    uint sst_capacity = bench->config->kv_restriction*bench->config->MemTable_capacity/2/bench->config->SSTable_count;     //218454   10G /1024
    cout<<"sst_capacity:"<<sst_capacity<<endl;
    //uint sst_capacity = 218454;
    key_value *temp_kvs = new key_value[sst_capacity];

    uint t_min = 3600*24*14;
    uint t_max = 0;

    uint *key_index = new uint[bench->config->MemTable_capacity/2]{0};
    int finish = 0;
    clock_t time1,time2;
    time1 = clock();
    while(finish<bench->config->MemTable_capacity/2){
        if(kv_count==0){
            SSTable_of.open("../store/SSTable_"+to_string(old_big)+"-"+to_string(sst_count), ios::out | ios::trunc);
            assert(SSTable_of.is_open());
        }
        finish = 0;
        __uint128_t temp_key = (__uint128_t)1<<126;
        box  temp_box;
        uint take_id =0;
        for(int i=0;i<bench->config->MemTable_capacity/2; i++){
//            if( bench->h_keys[offset+i][key_index[i]] == 0){              //empty kv
//                finish++;
//                continue;
//            }
            if(key_index[i]>= bench->config->kv_restriction){              //empty kv
                finish++;
                continue;
            }
            if( temp_key > bench->h_keys[offset+i][key_index[i]] ){
                temp_key = bench->h_keys[offset+i][key_index[i]];
                temp_box = bench->h_box_block[offset+i][bench->h_values[offset+i][key_index[i]]];               //bench->  i find the right 2G, then in box_block[ h_values ]
                take_id = i;
            }
        }
        if(finish<bench->config->MemTable_capacity/2){
            bench->h_keys[offset+take_id][key_index[take_id]] = 0;                                     //init
            key_index[take_id]++;                                                   // one while, one kv
            if(kv_count==0){
                bench->bg_run[old_big].first_pid[sst_count] = temp_key/100000000 / 100000000 / 100000000;
            }
            uint end = temp_key % 100000000;
            if(t_min>end){
                t_min = end;
            }
            if(t_max<end){
                t_max = end;
            }
//            print_128(temp_key);
//            cout<< ": "<< temp_box->low[0] << endl;
            temp_kvs[kv_count].key = temp_key;
            temp_kvs[kv_count].value = temp_box;
            kv_count++;
        }
        if(kv_count==sst_capacity||finish==bench->config->MemTable_capacity/2){
            SSTable_of.write((char *)temp_kvs, sizeof(key_value)*sst_capacity);
            SSTable_of.flush();
            SSTable_of.close();
            sst_count++;
            kv_count = 0;
        }
    }
    cout<<"sst_count :"<<sst_count<<" less than"<<1024<<endl;
    time2 = clock();
    double this_time = (double)(time2-time1)/CLOCKS_PER_SEC;
    cout<<"merge sort t: "<<bench->cur_time<<" time: "<< this_time <<std::endl;
    for(int i=0;i<bench->config->MemTable_capacity/2; i++){
        cout<<"key_index"<<key_index[i]<<endl;
    }
    delete[] key_index;
    bench->bg_run[old_big].timestamp_min = t_min;
    bench->bg_run[old_big].timestamp_max = t_max;
    bench->bg_run[old_big].print_meta();
    cout<<"now SSTable_count:"<<bench->bg_run[old_big].SSTable_count<<endl;

    bench->pro.bg_merge_flush_time += get_time_elapsed(bg_start,false);
    logt("merge sort and flush", bg_start);
    return NULL;
}

#ifdef USE_GPU
workbench *cuda_create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu);
#endif

void tracer::process(){
	struct timeval start = get_cur_time();
	for(int st=config->start_time;st<config->start_time+config->duration;st+=100){
        config->cur_duration = min((config->start_time+config->duration-st),(uint)100);
        if(config->load_data){
            loadData(config->trace_path.c_str(),st,config->cur_duration);;
        }
        else{
            generator->generate_trace(trace);
        }
		start = get_cur_time();
		if(!bench){
			bench = part->build_schema(trace, config->num_objects);
			bench->mbr = mbr;

#ifdef USE_GPU
			if(config->gpu){
				d_bench = cuda_create_device_bench(bench, gpu);
			}
#endif
		}
		for(int t=0;t<config->cur_duration;t++){
			log("");
			bench->reset();
			bench->points = trace+t*config->num_objects;
			bench->cur_time = st + t;
            if(bench->cur_time==config->start_time+config->duration-1){         //finish and all dump
                bench->crash_consistency = true;
            }
			// process the coordinate in this time point

            if(bench->cur_time==1900){
                bench->config->search_kv = true;                            //cuda search
                if(bench->config->search_kv){
                    bench->search_count = config->search_list_capacity;
                    for(int i=0;i<bench->search_count;i++){
                        bench->search_list[i].pid = 500000;                      //range query
                        //bench->search_list[i].pid = bench->h_keys[0][500]/100000000/100000000/100000000;
                        //bench->search_list[i].pid = bench->bg_run[1].first_pid[300];
                        bench->search_list[i].target = 0;
                        bench->search_list[i].start = 0;
                        bench->search_list[i].end = 0;
                    }
                }
            }

            if(bench->cur_time==3200){
                bench->config->search_kv = true;                            //cuda search
                if(bench->config->search_kv){
                    bench->search_count = config->search_list_capacity;
                    for(int i=0;i<bench->search_count;i++){
                        bench->search_list[i].pid = 3333333;                      //range query
                        //bench->search_list[i].pid = bench->h_keys[0][500]/100000000/100000000/100000000;
                        //bench->search_list[i].pid = bench->bg_run[1].first_pid[300];
                        bench->search_list[i].target = 0;
                        bench->search_list[i].start = 0;
                        bench->search_list[i].end = 0;
                    }
                }
            }

			if(!config->gpu){
				struct timeval ct = get_cur_time();
				bench->filter();
				bench->pro.filter_time += get_time_elapsed(ct,true);
				bench->reachability();
				bench->pro.refine_time += get_time_elapsed(ct,true);
				//bench->update_meetings();
				//bench->pro.meeting_identify_time += get_time_elapsed(ct,true);
			}else{
#ifdef USE_GPU
                process_with_gpu(bench,d_bench,gpu);
#endif
			}
            if(bench->cur_time==1900){                                   //total search
                cout<<"cuda search"<<endl;
                uint pid = 500000;
                //uint pid = bench->h_keys[0][500]/100000000/100000000/100000000;
                //uint pid = bench->bg_run[1].first_pid[300];
                cout<<"pid: "<<pid<<endl;
                for(int i=0;i<bench->search_count;i++){
                    //set<key_value> *range_result = new set<key_value>;
                    if(bench->search_list[i].target>0) {
                        cout << bench->search_list[i].pid << "-" << bench->search_list[i].target << "-"
                             << bench->search_list[i].start << "-" << bench->search_list[i].end << endl;
                    }

    //                if(bench->MemTable_count>0){
    //                    if(BloomFilter_Check(bench, 0 ,bench->search_list[i].pid)){
    //                        cout<< bench->search_list[i].pid <<"BloomFilter_Check :"<<endl;
    //                    }
    //                }
                }

                //search memtable
                struct timeval newstart = get_cur_time();
                bench->search_memtable(pid);
                bench->pro.search_memtable_time += get_time_elapsed(newstart,false);
                logt("search memtable",newstart);

                bench->search_count = 0;                //init
                bench->find_count = 0;
                uint valid_timestamp = 400;
                for(int i=0;i<bench->big_sorted_run_count;i++){
                    if((bench->bg_run[i].timestamp_min<valid_timestamp)&&(valid_timestamp<bench->bg_run[i].timestamp_max)){
                        bench->bg_run[i].search_in_disk(i,pid);
                    }
                }
                bench->pro.search_in_disk_time += get_time_elapsed(newstart,false);
                logt("search in disk",newstart);
            }
            if(bench->cur_time==3200){                                   //total search
                cout<<"cuda search"<<endl;
                uint pid = 3333333;
                cout<<"pid: "<<pid<<endl;
                for(int i=0;i<bench->search_count;i++){
                    if(bench->search_list[i].target>0) {
                        cout << bench->search_list[i].pid << "-" << bench->search_list[i].target << "-"
                             << bench->search_list[i].start << "-" << bench->search_list[i].end << endl;
                    }
                }

                //search memtable
                bench->search_memtable(pid);

                bench->search_count = 0;                //init
                bench->find_count = 0;
                uint valid_timestamp = 2000;
                for(int i=0;i<bench->big_sorted_run_count;i++){
                    if((bench->bg_run[i].timestamp_min<valid_timestamp)&&(valid_timestamp<bench->bg_run[i].timestamp_max)){
                        bench->bg_run[i].search_in_disk(i,pid);
                    }
                }
            }

//            if(bench->MemTable_count>0){
//                bool *check_2G = new bool[10000000];
//                int count = 0;
//                for(int i=0;i<bench->config->kv_restriction;i++){
//                    uint pid = bench->h_keys[0][i]/100000000 / 100000000 / 100000000;
//                    if(!check_2G[pid]){
//                        check_2G[pid] = true;
//                        count++;
//                    }
//                }
//                cout<<count<<" in "<<10000000<<endl;
//                assert(0);
//            }

//            if(bench->MemTable_count==bench->config->MemTable_capacity){              //check 10G
//                bool *check_2G = new bool[10000000];
//                int unique_count = 0;
//                for(int j=0;j<bench->MemTable_count;j++){
//                    for(int i=0;i<bench->config->kv_restriction;i++){
//                        uint pid = bench->h_keys[j][i]/100000000 / 100000000 / 100000000;
//                        if(!check_2G[pid]){
//                            check_2G[pid] = true;
//                            unique_count++;
//                        }
//                    }
//                }
//                cout<<unique_count<<" in "<<10000000<<endl;
//                assert(0);
//            }


            if(bench->MemTable_count==bench->config->MemTable_capacity/2) {
                bench->MemTable_count = 0;                                  //0<MemTable_count<MemTable_capacity/2
                uint offset = 0;
                if(bench->big_sorted_run_count%2==1){
                    offset = bench->config->MemTable_capacity/2;
                }
                for(int i=0;i<bench->config->MemTable_capacity/2; i++){
                    for(int j=0;j<10;j++){
                        print_128(bench->h_keys[offset+i][j]);
                        cout<<endl;
                    }
                    cout<<endl;
                }
                cout << "dump begin time: " << bench->cur_time << endl;
                pthread_t bg_thread;
                int ret;
                if ((ret = pthread_create(&bg_thread, NULL, sst_dump, (void *) bench)) != 0) {
                    fprintf(stderr, "pthread_create:%s\n", strerror(ret));
                }
                pthread_detach(bg_thread);
                //bool findit = searchkv_in_all_place(bench, 2);
            }

			if(config->analyze_grid||config->profile){
				bench->analyze_grids();
			}
			if(config->analyze_reach){
				bench->analyze_reaches();
			}
			if(config->dynamic_schema&&!config->gpu){
				struct timeval ct = get_cur_time();
				bench->update_schema();
				bench->pro.index_update_time += get_time_elapsed(ct,true);
			}
			logt("round %d",start,st+t+1);
			bench->pro.rounds++;
			bench->pro.max_refine_size = max(bench->pro.max_refine_size, bench->grid_check_counter);
			bench->pro.max_filter_size = max(bench->pro.max_filter_size, bench->filter_list_index);
			bench->pro.max_bucket_num = max(bench->pro.max_bucket_num, bench->num_taken_buckets);
			bench->pro.num_pairs += bench->num_active_meetings;

//			bench->pro.num_meetings += bench->meeting_counter;
//            if (t != 0 && bench->meeting_counter > 0) {
//                fprintf(stdout,"time=%d meeting_counter=%d\n",st + t,bench->meeting_counter);           // st+t+1
//                for (int i = 0; i < bench->meeting_counter; i++) {
//                    //fprintf(stdout,"(%d,%d) %d-%d (%f,%f); ",bench->meetings[i].get_pid1(),bench->meetings[i].get_pid2(),bench->meetings[i].start,bench->meetings[i].end,bench->meetings[i].midpoint.x,bench->meetings[i].midpoint.y);
//                    fprintf(stdout, "%zu (%f,%f)(%f,%f)|%d-%d;", bench->meetings[i].key,
//                            bench->meetings[i].mbr.ow[0], bench->meetings[i].mbr.low[1], bench->meetings[i].mbr.high[0], bench->meetings[i].mbr.high[1],
//                            bench->meetings[i].start, benlch->meetings[i].end);
//                }
//                fprintf(stdout, "\n");
//            }

		}
	}
	bench->print_profile();
}










//void tracer::trace_process(){
//    struct timeval start = get_cur_time();
//    for(int st=config->start_time;st<config->start_time+config->duration;st+=100){
//        int cur_duration = min((config->start_time+config->duration-st),(uint)100);
//        loadData(config->trace_path.c_str(),st,cur_duration);
//        start = get_cur_time();
//        if(!bench){
//            bench = part->build_schema(trace, config->num_objects);
//            bench->mbr = mbr;
//
//#ifdef USE_GPU
//            if(config->gpu){
//				d_bench = cuda_create_device_bench(bench, gpu);
//			}
//#endif
//        }
//        for(int t=0;t<cur_duration;t++) {
//            log("");
//            bench->reset();
//            bench->points = trace + t * config->num_objects;
//            bench->cur_time = st + t;
//
//            bench->config->search_kv = true;                            //cuda search
//            if(bench->config->search_kv){
//                bench->search_count = 100;
//                for(int i=0;i<bench->search_count;i++){
//                    bench->search_list[i].pid = i;
//                    bench->search_list[i].target = 0;
//                    bench->search_list[i].start = 0;
//                    bench->search_list[i].end = 0;
//                }
//            }
//
//            // process the coordinate in this time point
//            if (!config->gpu) {
//                struct timeval ct = get_cur_time();
//                bench->filter();
//                bench->pro.filter_time += get_time_elapsed(ct, true);
//                bench->reachability();
//                bench->pro.refine_time += get_time_elapsed(ct, true);
////                bench->update_meetings();
////                bench->pro.meeting_identify_time += get_time_elapsed(ct, true);
//            } else {
//#ifdef USE_GPU
//                process_with_gpu(bench,d_bench,gpu);
//#endif
//            }
//
////            for(int i=0;i<bench->search_count;i++){
////                if(bench->search_list[i].target>0)
////                    cout<< bench->search_list[i].pid<<"-"<<bench->search_list[i].target<<"-"<<bench->search_list[i].start<<"-"<<bench->search_list[i].end<<endl;
////                if(bench->MemTable_count>0){
////                    if(BloomFilter_Check(bench, 0 ,bench->search_list[i].pid)){
////                        cout<< bench->search_list[i].pid <<"BloomFilter_Check :"<<endl;
////                    }
////                }
////            }
//
////            if(bench->MemTable_count==bench->config->MemTable_capacity){
////                cout<<"dump begin time: "<<bench->cur_time<<endl;
////                pthread_t bg_thread;
////                int ret;
////                if ((ret=pthread_create(&bg_thread,NULL,sst_dump,(void*)bench)) != 0){
////                    fprintf(stderr,"pthread_create:%s\n",strerror(ret));
////                    exit(1);
////                }
////            }
//
//
////            if(bench->MemTable_count==bench->config->MemTable_capacity){
////                //merge sort can be optimized, since they are always kv_restriction now.
////
////                ofstream SSTable_of;
////                SSTable_of.open("../store/SSTable_of" + to_string(t), ios::out | ios::trunc);           //config.DBPath
////                uint *key_index = new uint[bench->config->MemTable_capacity]{0};
////                int finish = 0;
////                clock_t time1,time2;
////                time1 = clock();
////                while(finish<bench->config->MemTable_capacity){
////                    finish = 0;
////                    __uint128_t temp_key = (__uint128_t)1<<126;
////                    box * temp_box;
////                    uint take_id =0;
////                    for(int i=0;i<bench->config->MemTable_capacity;i++){
////                        if( bench->h_keys[i][key_index[i]] == 0){              //empty kv
////                            finish++;
////                            continue;
////                        }
////                        if( temp_key > bench->h_keys[i][key_index[i]] ){
////                            temp_key = bench->h_keys[i][key_index[i]];
////                            temp_box = &bench->h_box_block[i][bench->h_values[i][key_index[i]]];               //bench->  i find the right 2G, then in box_block[ h_values ]
////                            take_id = i;
////                            bench->h_keys[i][key_index[i]] = 0;                 //init
////                        }
////                    }
////                    if(finish<bench->config->MemTable_capacity){
////                        key_index[take_id]++;
////                        print_128(temp_key);
////                        cout<< ": "<< temp_box->low[0] << endl;
////                        SSTable_of.write((char *)&temp_key, sizeof(__uint128_t));
////                        SSTable_of.write((char *) temp_box, sizeof(box));
////                    }
////                }
////
////                time2 = clock();
////                double this_time = (double)(time2-time1)/CLOCKS_PER_SEC;
////                cout<<"merge sort t: "<<bench->cur_time<<" time: "<< this_time <<std::endl;
////                for(int i=0;i<bench->config->MemTable_capacity;i++){
////                    cout<<"key_index"<<key_index[i]<<endl;
////                }
////                delete[] key_index;
////
////                SSTable_of.flush();
////                SSTable_of.close();
////
////                //init
////                bench->MemTable_count = 0;
////                for(uint i=0;i<bench->config->MemTable_capacity;i++){                   //useless
////                    for(uint j=bench->config->kv_restriction; j < bench->config->kv_capacity ; j++){
////                        bench->h_keys[i][j] = 0;
////                    }
////                }
////
//////                ifstream read_f;
//////                read_f.open("SSTable377");
//////                for(int i=0;i<100;i++){
//////                    __uint128_t first_key;
//////                    box first_box;
//////                    read_f.read((char *)&first_key, sizeof(__uint128_t));
//////                    read_f.read((char *)&first_box, sizeof(box));
//////                    print_128(first_key);
//////                    cout<< ": "<< first_box.low[0] << endl;
//////                }
////            }
//
//
//            if (config->analyze_grid || config->profile) {
//                bench->analyze_grids();
//            }
//            if (config->analyze_reach) {
//                bench->analyze_reaches();
//            }
//            if (config->dynamic_schema && !config->gpu) {
//                struct timeval ct = get_cur_time();
//                bench->update_schema();
//                bench->pro.index_update_time += get_time_elapsed(ct, true);
//            }
//            logt("round %d", start, st + t + 1);
//            bench->pro.rounds++;
//            bench->pro.max_refine_size = max(bench->pro.max_refine_size, bench->grid_check_counter);
//            bench->pro.max_filter_size = max(bench->pro.max_filter_size, bench->filter_list_index);
//            bench->pro.max_bucket_num = max(bench->pro.max_bucket_num, bench->num_taken_buckets);
//            bench->pro.num_pairs += bench->num_active_meetings;
//
//        }
//    }
//    //bench->print_profile();
//}


