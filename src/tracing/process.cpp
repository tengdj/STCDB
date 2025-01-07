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
			//gpu = gpus[config->specific_gpu];
            gpu = gpus[gpus.size()-1];
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
	//loadMeta(config->trace_path.c_str());
	part = new partitioner(mbr,config);
#ifdef USE_GPU
	if(config->gpu){
		vector<gpu_info *> gpus = get_gpus();
		if(gpus.size()==0){
			log("not GPU is found, use CPU mode");
			config->gpu = false;
		}else{
			assert(config->specific_gpu<gpus.size());
			//gpu = gpus[config->specific_gpu];
            gpu = gpus[gpus.size()-1];
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
	if(part){
		delete part;
	}
	if(bench){
        //bench->clear();
        delete bench;
	}
    if(trace){
        delete[] trace;
    }
    if(generator){
        delete generator;
    }
//    if(config){
//        delete config;
//    }
#ifdef USE_GPU
//	if(gpu){
//		delete gpu;
//	}
#endif
}

void tracer::dumpTo(const char *path) {
	struct timeval start_time = get_cur_time();
	ofstream wf(path, ios::out|ios::binary|ios::trunc);
	wf.write((char *)&config->num_objects, sizeof(config->num_objects));
	wf.write((char *)&config->duration, sizeof(config->duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = config->duration*config->num_objects;
    cout<<"config->duration:"<<config->duration<<endl;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

//void tracer::loadMeta(const char *path) {
//
//	uint true_num_objects;
//	uint true_duration;
//	ifstream in(path, ios::in | ios::binary);
//	if(!in.is_open()){
//		log("%s cannot be opened",path);
//		exit(0);
//	}
//	in.read((char *)&true_num_objects, sizeof(true_num_objects));
//	in.read((char *)&true_duration, sizeof(true_duration));
//	log("%d objects last for %d seconds in file",true_num_objects,true_duration);
//	in.read((char *)&mbr, sizeof(mbr));
//	mbr.to_squre(true);
//	mbr.print();
//	assert((size_t)config->num_objects*(config->start_time+config->duration)<=(size_t)true_num_objects*true_duration);
//	//assert(config->num_objects<=true_num_objects);
//	assert(config->start_time+config->duration<=true_duration);
//	in.close();
//}

void tracer::loadData(const char *path, int st) {
    log("loading locations from %d to %d",st, st + 100);
    struct timeval start_time = get_cur_time();
    string filename = path + to_string(st) + "_" + to_string(config->num_objects) + ".tr";
    ifstream in(filename, ios::in | ios::binary);
    if(!in.is_open()){
        log("%s cannot be opened",filename.c_str());
        exit(0);
    }
    if(!trace){
        //trace = (Point *)malloc(min((uint)100,config->duration)*config->num_objects*sizeof(Point));
        trace = new Point[config->num_objects*100];
    }
    in.read((char *)trace, config->num_objects * 100 * sizeof(Point));
    in.close();
    logt("loaded %d objects last for 100 seconds start from %d time from %s",start_time, config->num_objects, st, filename.c_str());
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
	in.read((char *)&true_num_objects, sizeof(true_num_objects));               //also read meta
	in.read((char *)&true_duration, sizeof(true_duration));
	in.read((char *)&mbr, sizeof(mbr));

	assert((size_t)config->num_objects*(st+duration)<=(size_t)true_num_objects*true_duration);
	//assert(config->num_objects<=true_num_objects);
	assert(st+duration<=true_duration);

	in.seekg(st*true_num_objects*sizeof(Point), ios_base::cur);
	if(!trace){
        //trace = (Point *)malloc(min((uint)100,config->duration)*config->num_objects*sizeof(Point));
        trace = new Point[config->num_objects*100];
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

void *straight_dump(void *arg){
    struct timeval bg_start = get_cur_time();
    cout<<"step into the sst_dump"<<endl;
    workbench *bench = (workbench *)arg;
    cout<<"cur_time: "<<bench->cur_time<<endl;
    uint offset = 0;
    assert(bench->config->MemTable_capacity%2==0);
    uint old_big = bench->ctb_count;                 //atomic add
    cout<<"old big_sorted_run_count: "<<old_big<<endl;
    //bench->big_sorted_run_count++;
    bench->dumping = true;
    if(old_big%2==1){
        offset = bench->config->MemTable_capacity/2;
    }
    //CTB temp_ctb;
//    bench->ctbs.push_back(CTB());
    bench->ctbs[old_big].ctfs = NULL;
    bench->end_time_min = bench->end_time_max;              //new min = old max
    bench->ctbs[old_big].sids = new unsigned short[bench->config->num_objects];
    copy(bench->h_sids[offset], bench->h_sids[offset] + bench->config->num_objects, bench->ctbs[old_big].sids);
    bench->ctbs[old_big].o_buffer.oversize_kv_count = bench->h_oversize_buffers[offset].oversize_kv_count;
    bench->ctbs[old_big].o_buffer.keys = new __uint128_t[bench->ctbs[old_big].o_buffer.oversize_kv_count];
    copy(bench->h_oversize_buffers[offset].keys, bench->h_oversize_buffers[offset].keys + bench->ctbs[old_big].o_buffer.oversize_kv_count, bench->ctbs[old_big].o_buffer.keys);
    bench->ctbs[old_big].o_buffer.boxes = new f_box[bench->ctbs[old_big].o_buffer.oversize_kv_count];
    copy(bench->h_oversize_buffers[offset].boxes, bench->h_oversize_buffers[offset].boxes + bench->ctbs[old_big].o_buffer.oversize_kv_count, bench->ctbs[old_big].o_buffer.boxes);

//    cout << "test oversize" << endl;
//    for(uint i = 0; i < min((uint)100, bench->ctbs[old_big].o_buffer.oversize_kv_count); i++){
//        print_parse_key(bench->ctbs[old_big].o_buffer.keys[i]);
//            cout << bench->ctbs[old_big].o_buffer.boxes[i].low[0] << endl;
//        cout << "sid" << bench->ctbs[old_big].sids[get_key_oid(bench->ctbs[old_big].o_buffer.keys[i]) ] << endl;
//    }

    bench->ctbs[old_big].box_rtree = new RTree<short *, double, 2, double>();
    for(uint i = 0; i < bench->config->CTF_count; ++i){
        bench->ctbs[old_big].box_rtree->Insert(bench->h_bitmap_mbrs[offset][i].low, bench->h_bitmap_mbrs[offset][i].high, new short(i));
    }

//    cout << "CTF_capacitys:" << endl;
//    for(uint i = 0; i < bench->config->CTF_count; i++){
//        cout << bench->ctbs[old_big].CTF_capacity[i] << endl;
//    }
//    cerr << "all bitmap_mbrs" << endl;
//    for(uint i = 0; i < bench->config->SSTable_count; i++){
//        bench->bg_run[old_big].bitmap_mbrs[i].print();
//    }

    logt("CPU organization time",bg_start);

    uint * key_index = new uint[bench->config->CTF_count];          //prefix
    uint * bytes_index = new uint[bench->config->CTF_count];          //prefix
    key_index[0] = 0;
    bytes_index[0] = 0;
    for(uint i = 1; i < bench->config->CTF_count; i++){
        key_index[i] = key_index[i - 1] + bench->h_CTF_capacity[offset][i - 1];
        bytes_index[i] = bytes_index[i - 1] + bench->h_CTF_capacity[offset][i - 1] * bench->h_ctfs[offset][i - 1].key_bit / 8;
    }

    uint8_t * h_ctf_keys = reinterpret_cast<uint8_t *>(bench->h_keys[offset]);
#pragma omp parallel for num_threads(bench->config->CTF_count)          //there is little improvement when 100->128 threads
    for(uint sst_count=0; sst_count < bench->config->CTF_count; sst_count++){
        string sst_path = bench->config->raid_path + to_string(sst_count%2) + "/N_SSTable_"+to_string(old_big)+"-"+to_string(sst_count);
        ofstream SSTable_of;
        SSTable_of.open(sst_path , ios::out|ios::binary|ios::trunc);
        assert(SSTable_of.is_open());
        SSTable_of.write((char *)(h_ctf_keys + bytes_index[sst_count]), bench->h_ctfs[offset][sst_count].key_bit / 8 * bench->h_CTF_capacity[offset][sst_count]);
        SSTable_of.flush();
        SSTable_of.close();
    }
    bench->pro.bg_merge_time += get_time_elapsed(bg_start,false);
    logt("dumped keys for CTB %d",bg_start, old_big);

    //bench->ctbs[old_big].print_meta();
    //logt("merge sort and flush", bg_start);
    //delete[] bit_points;
    bench->dumping = false;

    string CTB_path = string(bench->config->CTB_meta_path) + "N_CTB" + to_string(old_big);
    bench->dump_CTB_meta(CTB_path.c_str(), old_big);
    logt("dumped meta for CTB %d",bg_start, old_big);

#pragma omp parallel for num_threads(bench->config->CTF_count)
    for(uint i = 0; i < bench->config->CTF_count; i++){
        string ctf_path = string(bench->config->CTB_meta_path) + "STcL" + to_string(old_big)+"-"+to_string(i);
        bench->h_ctfs[offset][i].bitmap = &bench->h_bitmaps[offset][bench->bitmaps_size * i];
        bench->h_ctfs[offset][i].dump(ctf_path);
    }
    logt("dumped meta for CTF",bg_start);
    return NULL;
}

void* commandThreadFunction(void* arg) {
    workbench *bench = (workbench *)arg;
    cout << "bench->config->SSTable_count: " << bench->config->CTF_count << endl;
    pthread_mutex_init(&bench->mutex_i,NULL);
    while (true) {
        pthread_mutex_lock(&bench->mutex_i);
        std::string command;
        std::cout << "Enter command: ";
        std::cin >> command;
        if (command == "interrupt") {
            cout<<"will interrupt in next round"<<endl;
            bench->interrupted = true;
        } else if (command == "exit") {
            cout<<"will exit"<<endl;
            break;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
            pthread_mutex_unlock(&bench->mutex_i);
        }
    }
    return NULL;
}

#ifdef USE_GPU
workbench *cuda_create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu);
#endif

void tracer::process(){
    //config->reach_distance /= sqrt(config->num_objects / 10000000);
    //config->num_threads = 1;
    struct timeval start = get_cur_time();
    std::cout << "Running main program..." << std::endl;
	for(int st=config->start_time;st<config->start_time+config->duration;st+=100){
        config->cur_duration = min((config->start_time+config->duration-st),(uint)100);
        if(config->load_data){
            //loadData(config->trace_path.c_str(),st,config->cur_duration);
            loadData(config->trace_path.c_str(),st);
        }
        else if(!config->load_meetings_pers){
//            cout << "config.cur_duration : "<< config->cur_duration <<endl;
//            generator->map->print_region();
//            sleep(2);
            if(true){
                cout << st << endl;
//                generator->map->check_Streets();
//                generator->map->check_Nodes();
            }
            generator->generate_trace(trace);
        }
		start = get_cur_time();
        if(!bench){
            bench = part->build_schema(trace, config->num_objects);
            bench->mbr = mbr;
            bench->end_time_min = config->start_time + config->min_meet_time;           //first min time
            bench->start_time_min = (1ULL<<32) -1;
            bench->start_time_max = 0;

//            //command
//            int ret1;
//            if ((ret1 = pthread_create(&bench->command_thread, NULL, commandThreadFunction, (void *) bench)) != 0) {
//                fprintf(stderr, "pthread_create:%s\n", strerror(ret1));
//            }
//            pthread_detach(bench->command_thread);
#ifdef USE_GPU
            if(config->gpu){
				d_bench = cuda_create_device_bench(bench, gpu);
			}
#endif
        }
        if(config->load_meetings_pers){
            bench->load_meetings(st);
        }

		for(int t=0;t<config->cur_duration;t++){
			log("");
			bench->reset();
			bench->points = trace+t*config->num_objects;
			bench->cur_time = st + t;
            if(bench->cur_time==config->start_time+config->duration-1){         //finish and all dump
                //pthread_cancel(bench->command_thread);
                //bench->crash_consistency = true;
                //bench->clear();
            }
			// process the coordinate in this time point

            if(bench->interrupted){
                bench->search_single = true;
                cout<<"search pid: ";
                cin>>bench->search_single_pid;
                cout<<"valid_timestamp: ";
                cin>>bench->valid_timestamp;
                cout<<endl;
            }

			if(!config->gpu){
				struct timeval ct = get_cur_time();
				bench->filter();
				bench->pro.filter_time += get_time_elapsed(ct,true);
				bench->reachability();
				bench->pro.refine_time += get_time_elapsed(ct,true);
				bench->update_meetings();
				bench->pro.meeting_identify_time += get_time_elapsed(ct,true);
			}else{
#ifdef USE_GPU
                process_with_gpu(bench,d_bench,gpu);
#endif
			}
            if(bench->search_multi){
                bench->search_multi = false;
                cout<<"cuda multi search"<<endl;
                cout<<"cuda multi_find_count: "<<bench->multi_find_count<<endl;
                for(int i=0;i<bench->multi_find_count;i++){
                    cout << bench->search_multi_list[i].pid << "-" << bench->search_multi_list[i].target << "-"
                         << bench->search_multi_list[i].start << "-" << bench->search_multi_list[i].end << "-"
                         << bench->search_multi_list[i].low0 << "-" << bench->search_multi_list[i].low1 << "-"
                         << bench->search_multi_list[i].high0 << "-" << bench->search_multi_list[i].high1 << endl;
                }
            }
            if(bench->search_single){                                   //search_single
                bench->interrupted = false;                             //reset
                bench->search_single = false;
                bench->search_multi = true;
                cout<<"cuda single search"<<endl;
                cout<<"single_find_count: "<<bench->single_find_count<<endl;
                bench->search_multi_length = bench->single_find_count;
                for(int i=0;i<bench->single_find_count;i++){
                    cout << bench->search_single_pid << "-" << bench->search_single_list[i].target << "-"
                        << bench->search_single_list[i].start << "-" << bench->search_single_list[i].end << "-"
                        << bench->search_single_list[i].low0 << "-" << bench->search_single_list[i].low1 << "-"
                        << bench->search_single_list[i].high0 << "-" << bench->search_single_list[i].high1 << endl;
                    bench->search_multi_pid[i] = bench->search_single_list[i].target;
                }

                struct timeval newstart = get_cur_time();
                bench->pro.search_in_disk_time += get_time_elapsed(newstart,false);
                logt("search in disk",newstart);
                pthread_mutex_unlock(&bench->mutex_i);
            }
            if(bench->crash_consistency){
                uint offset = 0;
                if(bench->ctb_count % 2 == 1){
                    offset = bench->config->MemTable_capacity/2;
                }
                for(int i=0;i<bench->MemTable_count; i++){
                    for(int j=0;j<10;j++){
                        print_128(bench->h_keys[offset+i][j]);
                        cout<<(uint)(bench->h_keys[offset+i][j] >> 39)<<endl;
                        //print_128(bench->h_values[offset+i][j]);
                        cout<<endl;
                    }
                    cout<<endl;
                }
                cout<<"crash_consistency, 2 merge sort and dump"<<endl;
                cout << "crash dump begin time: " << bench->cur_time << endl;
                //crash_sst_dump((void *)bench);
                bench->clear();
            }
            else if(bench->MemTable_count==bench->config->MemTable_capacity/2) {    //0 <= MemTable_count <= MemTable_capacity/2
                uint offset = 0;
                if(bench->ctb_count % 2 == 1){
                    offset = bench->config->MemTable_capacity/2;
                }

//                Point * bit_points = new Point[bench->bit_count];
//                uint count_p;
//                for(uint j = 0;j<bench->config->CTF_count; j++){
//                    //cerr<<"bitmap"<<j<<endl;
//                    cerr<<end
//                    count_p = 0;
//                    bool is_print = false;
//                    for(uint i=0;i<bench->bit_count;i++){
//                        if(bench->h_bitmaps[offset][j*(bench->bit_count/8) + i/8] & (1<<(i%8))){
//                            if(!is_print){
//                                cout<<i<<"in SST"<<j<<endl;
//                                is_print = true;
//                            }
//                            Point bit_p;
//                            uint x=0,y=0;
//                            d2xy(SID_BIT/2,i,x,y);
//                            bit_p.x = (double)x/(1ULL << (SID_BIT/2))*(bench->mbr.high[0] - bench->mbr.low[0]) + bench->mbr.low[0];           //int low0 = (f_low0 - bench->mbr.low[0])/(bench->mbr.high[0] - bench->mbr.low[0]) * (pow(2,WID_BIT/2) - 1);
//                            bit_p.y = (double)y/(1ULL << (SID_BIT/2))*(bench->mbr.high[1] - bench->mbr.low[1]) + bench->mbr.low[1];               //int low1 = (f_low1 - bench->mbr.low[1])/(bench->mbr.high[1] - bench->mbr.low[1]) * (pow(2,WID_BIT/2) - 1);
//                            bit_points[count_p] = bit_p;
//                            count_p++;
//                        }
//                    }
//                    cout<<"bit_points.size():"<<count_p<<endl;
//                    print_points(bit_points,count_p);
//                    //cerr << "process output bitmap finish" << endl;
//                }
//                delete[] bit_points;

                bench->end_time_max = bench->cur_time;              //old max
                cout<<"meeting_cut_count:"<<bench->meeting_cut_count<<endl;
                cout<<"start_time_min:"<<bench->start_time_min<<"start_time_max:"<<bench->start_time_max<<"bench->end_time_min:"<<bench->end_time_min<<"bench->end_time_max:"<<bench->end_time_max<<endl;

                cout << "dump begin time: " << bench->cur_time << endl;
//                pthread_t bg_thread;
//                int ret;
//                if ((ret = pthread_create(&bg_thread, NULL, merge_dump, (void *) bench)) != 0) {
//                    fprintf(stderr, "pthread_create:%s\n", strerror(ret));
//                }
//                pthread_detach(bg_thread);



//                ofstream p;
//                string filename = "longer_edges" + to_string(bench->ctb_count) + ".csv";
//                cout << filename << endl;
//                p.open(filename, ios::out|ios::binary|ios::trunc);
//                p << "percent(%)" << ',' << "edge_length" << endl;
//                int this_count = 0;
//                for(int i = 0 ; i < bench->config->num_objects; i += 20000){
//                    p << i/20000 << ',' << bench->h_longer_edges[i] << endl;
////                    if(i%1000000==0){
////                        cout << i/1000000 << " " << bench->h_longer_edges[i] << endl;
////                    }
//                }
//                p.close();


                if(config->MemTable_capacity==2){
                    straight_dump((void *)bench);
                }
//                else{
//                    merge_dump((void *)bench);
//                }

                //f((void *)bench);

                //init
                bench->ctb_count++;
                bench->MemTable_count = 0;
                bench->do_some_search = false;

                //bool findit = searchkv_in_all_place(bench, 2);
            }


//			if(config->analyze_grid||config->profile){
//				bench->analyze_grids();
//			}
//			if(config->analyze_reach){
//				bench->analyze_reaches();
//			}
			if(config->dynamic_schema&&!config->gpu){
				struct timeval ct = get_cur_time();
				bench->update_schema();
				bench->pro.index_update_time += get_time_elapsed(ct,true);
			}
            bench->pro.sum_round_time += get_time_elapsed(start,false);
            fprintf(stdout,"\tbench->pro.sum_round_time :\t%.2f\n",bench->pro.sum_round_time);
            //cout << "sum_round_time: "<< bench->pro.sum_round_time << endl;
			logt("round %d",start,st+t+1);
			bench->pro.rounds++;
			bench->pro.max_refine_size = max(bench->pro.max_refine_size, bench->grid_check_counter);
			bench->pro.max_filter_size = max(bench->pro.max_filter_size, bench->filter_list_index);
			bench->pro.max_bucket_num = max(bench->pro.max_bucket_num, bench->num_taken_buckets);
			bench->pro.num_pairs += bench->num_active_meetings;

            if(config->save_meetings_pers && t == 99){
                bench->dump_meetings(st);
            }

            if(st+t+1 == config->start_time+config->duration - 5  || (st+t+1) % 100000 == 0){
                bench->dump_meta(config->CTB_meta_path);
            }
		}
	}
    //bench->clear();
	bench->print_profile();
}








