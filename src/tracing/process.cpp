/*
 * process.cpp
 *
 *  Created on: Feb 11, 2021
 *      Author: teng
 */

#include "trace.h"
#include "../index/QTree.h"
#include <set>


/*
 * functions for tracer
 *
 * */



tracer::tracer(configuration *conf, box &b, Point *t){
    trace = t;
    mbr = b;
    config = conf;
    part = new partitioner(mbr,config);
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
};
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

//void tracer::print_point(int oid,int t){            //print single point
//    Point *p= trace+t*config->num_objects+oid;
//    p->print();
//}

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
        for(int i=0;i<config->cur_duration;i++){
            points.push_back(trace+i*config->num_objects+oid);
        }
    }
    print_points(points, 10000);
    points.clear();
}

//template <typename T>
//std::string Pack(const T* data)
//{
//    std::string d(sizeof(T), L'\0');
//    memcpy(&d[0], data, d.size());
//    return d;
//}
//
//template <typename T>
//std::unique_ptr<T> Unpack(const std::string& data)
//{
//    if (data.size() != sizeof(T))
//        return nullptr;
//
//    auto d = std::make_unique<T>();
//    memcpy(d.get(), data.data(), data.size());
//    return d;
//}

#ifdef USE_GPU
workbench *cuda_create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu, int t);
#endif


void tracer::process(int st){
    struct timeval start = get_cur_time();

    //loadData(config->trace_path.c_str(),st,cur_duration);
    start = get_cur_time();
    meeting_unit * meeting_temp = new meeting_unit[config->num_meeting_buckets];
    //fprintf(stderr,"%ld\n",config->num_meeting_buckets);
    if(st != config->start_time){               // if bench
        for(int i=0;i<config->num_meeting_buckets;i++){
            meeting_temp[i] = bench->meeting_buckets[i];
//            if(!meeting_temp[i].isEmpty()){
//                fprintf(stderr,"copy out right\n");
//            }
        }
        bench->clear();
        delete bench;
#ifdef USE_GPU
        if(config->gpu){
            gpu->clear();
        }
#endif
    }
    bench = part->build_schema(trace, config->num_objects);
    if(st != config->start_time){
        for(int i=0;i<config->num_meeting_buckets;i++){
            bench->meeting_buckets[i] = meeting_temp[i];
//            if(!meeting_temp[i].isEmpty()){
//                fprintf(stderr,"copy back right\n");
//            }
        }
    }
    delete []meeting_temp;
    bench->mbr = mbr;
#ifdef USE_GPU
    if(config->gpu){
        d_bench = cuda_create_device_bench(bench, gpu);
    }
#endif
    int t = 0;
    for(t=0;t<config->cur_duration;t++) {
        log("");
        bench->reset();
        bench->points = trace + t * config->num_objects;
        bench->cur_time = st + t;
        // process the coordinate in this time point
        if (!config->gpu) {
            struct timeval ct = get_cur_time();
            bench->filter();
            bench->pro.filter_time += get_time_elapsed(ct, true);
            bench->reachability();
            bench->pro.refine_time += get_time_elapsed(ct, true);
            bench->update_meetings();
            bench->pro.meeting_identify_time += get_time_elapsed(ct, true);
        } else {
#ifdef USE_GPU
            process_with_gpu(bench,d_bench,gpu,t);
#endif
        }


        if (config->analyze_grid || config->profile) {
            bench->analyze_grids();
        }
        if (config->analyze_reach) {
            bench->analyze_reaches();
        }
        if (config->dynamic_schema && !config->gpu) {
            struct timeval ct = get_cur_time();
            bench->update_schema();
            bench->pro.index_update_time += get_time_elapsed(ct, true);
        }
        logt("round %d", start, st + t + 1);
        bench->pro.rounds++;
        bench->pro.max_refine_size = max(bench->pro.max_refine_size, bench->grid_check_counter);
        bench->pro.max_filter_size = max(bench->pro.max_filter_size, bench->filter_list_index);
        bench->pro.max_bucket_num = max(bench->pro.max_bucket_num, bench->num_taken_buckets);
        bench->pro.num_pairs += bench->num_active_meetings;
        bench->pro.num_meetings += bench->meeting_counter;


        fprintf(stderr, "time=%d meeting_counter=%d\n", st + t, bench->meeting_counter);           // st+t+1
        ofstream outFile;
        outFile.open("../meetings/meetings_"+to_string(st+t)+".in", ios::out | ios::binary);
        uint cur_time = st + t;
        outFile.write((char *) &cur_time, sizeof(uint));
        outFile.write((char *) &bench->meeting_counter, sizeof(bench->meeting_counter));
        for (int i = 0; i < bench->meeting_counter; i++) {
            outFile.write((char *) &bench->meetings[i], sizeof(meeting_unit));
        }
        outFile.close();


//        if ((t + st) != 0 && bench->meeting_counter > 0) {
//            fprintf(stdout, "time=%d meeting_counter=%d\n", st + t, bench->meeting_counter);           // st+t+1
//            for (int i = 0; i < bench->meeting_counter; i++) {
//                //fprintf(stdout,"(%d,%d) %d-%d (%f,%f); ",bench->meetings[i].get_pid1(),bench->meetings[i].get_pid2(),bench->meetings[i].start,bench->meetings[i].end,bench->meetings[i].midpoint.x,bench->meetings[i].midpoint.y);
//                fprintf(stdout, "%zu (%f,%f)(%f,%f)|%d-%d;", bench->meetings[i].key,
//                        bench->meetings[i].mbr.low[0], bench->meetings[i].mbr.low[1], bench->meetings[i].mbr.high[0],
//                        bench->meetings[i].mbr.high[1],
//                        bench->meetings[i].start, bench->meetings[i].end);
//            }
//            fprintf(stdout, "\n");
//        }

//        //fprintf(stdout, "time=%d meeting_counter=%d\n", st + t, bench->meeting_counter);           // st+t+1
//        if (t % 100== 10) {
//            print_trace(222);
//            fprintf(stdout, "\n");
//        }


    }
}
