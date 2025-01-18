#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.cuh"
#include "hilbert_curve.cuh"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"
#include "../tracing/workbench.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>
#include <thrust/partition.h>

/*
 *
 * some utility functions
 *
 * */

__device__
inline double height(box *b){
    return (b->high[1]-b->low[1])/degree_per_meter_latitude_cuda;
}

__device__
inline double distance(box *b,Point *p){

    double dx = max(abs(p->x-(b->low[0]+b->high[0])/2) - (b->high[0]-b->low[0])/2, 0.0);
    double dy = max(abs(p->y-(b->low[1]+b->high[1])/2) - (b->high[1]-b->low[1])/2, 0.0);
    dy = dy/degree_per_meter_latitude_cuda;
    dx = dx/degree_per_meter_longitude_cuda(p->y);

    return sqrt(dx * dx + dy * dy);
}

__device__
inline bool contain(box *b, Point *p){            //?? double  bool
    return p->x>=b->low[0]&&
           p->x<=b->high[0]&&
           p->y>=b->low[1]&&
           p->y<=b->high[1];
}

__device__
inline bool box_contain(box *b, f_box *target){            //?? double  bool
    return target->low[0]>=b->low[0]&&
           target->high[0]<=b->high[0]&&
           target->low[1]>=b->low[1]&&
           target->high[1]<=b->high[1];
}

__device__
inline void print_box_point(f_box *b, Point *p){
    printf("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))\nPOINT(%f %f)\n",
           b->low[0],b->low[1],
           b->high[0],b->low[1],
           b->high[0],b->high[1],
           b->low[0],b->high[1],
           b->low[0],b->low[1],
           p->x,p->y);
}

__device__
inline void print_box(f_box *b){
    printf("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))\n",
           b->low[0],b->low[1],
           b->high[0],b->low[1],
           b->high[0],b->high[1],
           b->low[0],b->high[1],
           b->low[0],b->low[1]);
}

__device__
inline void mbr_update_by_point(f_box &mbr, Point *p){
    if(mbr.low[0]>p->x){
        mbr.low[0] = p->x;
    }
    if(mbr.high[0]<p->x){
        mbr.high[0] = p->x;
    }

    if(mbr.low[1]>p->y){
        mbr.low[1] = p->y;
    }
    if(mbr.high[1]<p->y){
        mbr.high[1] = p->y;
    }
}

__device__
inline void mbr_update_by_mbr(f_box &mbr, f_box &b){
    if(mbr.low[0] > b.low[0]){
        mbr.low[0] = b.low[0];
    }
    if(mbr.high[0] < b.high[0]){
        mbr.high[0] = b.high[0];
    }

    if(mbr.low[1] > b.low[1]){
        mbr.low[1] = b.low[1];
    }
    if(mbr.high[1] < b.high[1]){
        mbr.high[1] = b.high[1];
    }
}

__device__
inline void atomic_mbr_update_by_mbr(uint_box &mbr, f_box &b){
    atomicMin(&mbr.low[0], float_to_uint(b.low[0]));
    atomicMax(&mbr.high[0], float_to_uint(b.high[0]));
    atomicMin(&mbr.low[1], float_to_uint(b.low[1]));
    atomicMax(&mbr.high[1], float_to_uint(b.high[1]));
}

__device__
inline void print_point(Point *p){
    printf("Point(%f %f)\n",p->x,p->y);
}

__device__
void write_kv_box(f_box * kv_b, f_box * meeting_b){
    kv_b->low[0] = meeting_b->low[0];
    kv_b->low[1] = meeting_b->low[1];
    kv_b->high[0] = meeting_b->high[0];
    kv_b->high[1] = meeting_b->high[1];
}

__host__ __device__
uint cuda_get_key_sid(__uint128_t key){
    return (uint)((key >> (OID_BIT * 2 + DURATION_BIT + END_BIT)) & ((1ULL << SID_BIT) - 1));
}

__host__ __device__
uint cuda_get_key_oid(__uint128_t key){
    return (uint)((key >> (OID_BIT + DURATION_BIT + END_BIT)) & ((1ULL << OID_BIT) - 1));
}

__host__ __device__
uint cuda_get_key_target(__uint128_t key){
    return (uint)((key >> ( DURATION_BIT + END_BIT)) & ((1ULL << OID_BIT) - 1));
}

__host__ __device__
uint cuda_get_key_duration(__uint128_t key){
    return (uint)((key >> END_BIT) & ((1ULL << DURATION_BIT) - 1));
}

__host__ __device__
uint cuda_get_key_end(__uint128_t key){
    return (uint)(key & ((1ULL << END_BIT) - 1));
}

__host__ __device__
uint64_t serialize_mbr(f_box* b, f_box* bitmap_mbr, CTF * ctf){
    uint64_t low0 = (b->low[0] - bitmap_mbr->low[0])/(bitmap_mbr->high[0] - bitmap_mbr->low[0]) * ((1ULL << (ctf->low_x_bit)) - 1);
    uint64_t low1 = (b->low[1] - bitmap_mbr->low[1])/(bitmap_mbr->high[1] - bitmap_mbr->low[1]) * ((1ULL << (ctf->low_y_bit)) - 1);
    uint64_t x = (b->high[0] - bitmap_mbr->low[0])/(bitmap_mbr->high[0] - bitmap_mbr->low[0]) * ((1ULL << (ctf->edge_bit)) - 1);
    uint64_t y = (b->high[1] - bitmap_mbr->low[1])/(bitmap_mbr->high[1] - bitmap_mbr->low[1]) * ((1ULL << (ctf->edge_bit)) - 1);
    uint64_t value_mbr = ((uint64_t)low0 << (ctf->low_y_bit + ctf->edge_bit + ctf->edge_bit)) + ((uint64_t)low1 << (ctf->edge_bit + ctf->edge_bit))
                        + ((uint64_t)x << (ctf->edge_bit)) + (uint64_t)y;
    return value_mbr;
}

__host__ __device__
uint64_t serialize_mbr(box* b, box* bitmap_mbr, CTF * ctf){
    uint64_t low0 = (b->low[0] - bitmap_mbr->low[0])/(bitmap_mbr->high[0] - bitmap_mbr->low[0]) * ((1ULL << (ctf->low_x_bit)) - 1);
    uint64_t low1 = (b->low[1] - bitmap_mbr->low[1])/(bitmap_mbr->high[1] - bitmap_mbr->low[1]) * ((1ULL << (ctf->low_y_bit)) - 1);
    uint64_t x = (b->high[0] - bitmap_mbr->low[0])/(bitmap_mbr->high[0] - bitmap_mbr->low[0]) * ((1ULL << (ctf->edge_bit)) - 1);
    uint64_t y = (b->high[1] - bitmap_mbr->low[1])/(bitmap_mbr->high[1] - bitmap_mbr->low[1]) * ((1ULL << (ctf->edge_bit)) - 1);
    uint64_t value_mbr = ((uint64_t)low0 << (ctf->low_y_bit + ctf->edge_bit + ctf->edge_bit)) + ((uint64_t)low1 << (ctf->edge_bit + ctf->edge_bit))
                         + ((uint64_t)x << (ctf->edge_bit)) + (uint64_t)y;
    return value_mbr;
}

/*
 *
 * kernel functions
 *
 * */

__global__
void cuda_cleargrids(workbench *bench){
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=bench->grids_stack_capacity){
        return;
    }
    bench->grid_counter[gid] = 0;
}

__global__
void cuda_reset_bench(workbench *bench){
    bench->grid_check_counter = 0;
    //bench->meeting_counter = 0;
    bench->num_active_meetings = 0;
    bench->num_taken_buckets = 0;
    bench->filter_list_index = 0;
    bench->split_list_index = 0;
    bench->merge_list_index = 0;
}

__global__
void cuda_clean_buckets(workbench *bench){
    size_t bid = blockIdx.x*blockDim.x+threadIdx.x;
    if(bid>=bench->config->num_meeting_buckets){
        return;
    }
    bench->meeting_buckets[bid].key = ULL_MAX;
}

//  partition with cuda
__global__
void cuda_partition(workbench *bench){
    int pid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=bench->config->num_objects){
        return;
    }

    // search the tree to get in which grid
    uint curnode = 0;
    uint gid = 0;

    Point *p = bench->points+pid;
    uint last_valid = 0;
    while(true){
        int loc = (p->y>bench->schema[curnode].mid_y)*2 + (p->x>bench->schema[curnode].mid_x);
        curnode = bench->schema[curnode].children[loc];

        // not near the right and top border
        if(p->x+bench->config->x_buffer<bench->schema[curnode].mbr.high[0]&&
           p->y+bench->config->y_buffer<bench->schema[curnode].mbr.high[1]){
            last_valid = curnode;
        }

        // is leaf
        if(bench->schema[curnode].type==LEAF){
            gid = bench->schema[curnode].grid_id;
            break;
        }
    }

    // insert current pid to proper memory space of the target gid
    // todo: consider the situation that grid buffer is too small
    uint *cur_grid = bench->grids+bench->grid_capacity*gid;
    uint cur_loc = atomicAdd(bench->grid_counter+gid,1);
    if(cur_loc<bench->grid_capacity){
        *(cur_grid+cur_loc) = pid;
    }
    uint glid = atomicAdd(&bench->grid_check_counter,1);
    bench->grid_check[glid].pid = pid;
    bench->grid_check[glid].gid = gid;
    bench->grid_check[glid].offset = 0;
    bench->grid_check[glid].inside = true;

    if(last_valid!=curnode){
        uint stack_index = atomicAdd(&bench->filter_list_index,1);
        assert(stack_index<bench->filter_list_capacity);
        bench->filter_list[stack_index*2] = pid;
        bench->filter_list[stack_index*2+1] = last_valid;
    }

}


/*
 *
 * functions for filtering
 *
 * */

#define PER_STACK_SIZE 5

__global__
void cuda_pack_lookup(workbench *bench){
    int pid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=bench->config->num_objects){
        return;
    }

    uint idx = atomicAdd(&bench->filter_list_index,1);
    assert(idx<bench->filter_list_capacity);
    bench->filter_list[idx*2] = pid;
    bench->filter_list[idx*2+1] = 0;
}

__global__
void cuda_filtering(workbench *bench, int start_idx, int batch_size, bool include_contain){
    int cur_idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idx = cur_idx + start_idx;
    if(cur_idx>=batch_size){
        return;
    }
    int pid = bench->filter_list[idx*2];
    int nodeid = bench->filter_list[idx*2+1];

    // get the block shared stack
    int block_stack_size = 1024*2*PER_STACK_SIZE;
    int stack_offset = blockIdx.x*block_stack_size;

    assert(stack_offset+block_stack_size<bench->tmp_space_capacity);

    int *cur_stack_idx = (int *)bench->tmp_space+stack_offset;
    int *cur_worker_idx = (int *)bench->tmp_space+stack_offset+1;
    uint *cur_stack = bench->tmp_space+stack_offset+2;

    *cur_stack_idx = 0;
    *cur_worker_idx = 0;
    __syncthreads();

    int stack_index = atomicAdd(cur_stack_idx, 1);
    cur_stack[2*stack_index] = pid;
    cur_stack[2*stack_index+1] = nodeid;

    //printf("%d:\tinit push %d\n",threadIdx.x,stack_index);
    __syncthreads();

    while(true){
        bool busy = false;
        stack_index = atomicSub(cur_stack_idx, 1)-1;
        //printf("%d:\tpop %d\n",threadIdx.x, stack_index);
        __syncthreads();
        if(stack_index<0){
            stack_index = atomicAdd(cur_stack_idx, 1);
            //printf("%d:\tinc %d\n",threadIdx.x, stack_index);
        }else{
            busy = true;
            atomicAdd(cur_worker_idx, 1);
        }
        __syncthreads();

        //printf("num workers: %d\n",*cur_worker_idx);
        if(*cur_worker_idx==0){
            break;
        }
        if(busy){
            uint pid = cur_stack[2*stack_index];
            uint curnode = cur_stack[2*stack_index+1];
            Point *p = bench->points+pid;
            //printf("process: %d %d %d\n",stack_index,pid,curnode);

            for(int i=0;i<4;i++){
                uint child_offset = bench->schema[curnode].children[i];
                double dist = distance(&bench->schema[child_offset].mbr, p);
                if(dist<=bench->config->reach_distance){
                    if(bench->schema[child_offset].type==LEAF){
                        uint gid = bench->schema[child_offset].grid_id;
                        assert(gid<bench->grids_stack_capacity);
                        if(include_contain&&contain(&bench->schema[child_offset].mbr,p)){
                            uint *cur_grid = bench->grids+bench->grid_capacity*gid;
                            uint cur_loc = atomicAdd(bench->grid_counter+gid,1);
                            if(cur_loc<bench->grid_capacity){
                                *(cur_grid+cur_loc) = pid;
                            }
                            uint glid = atomicAdd(&bench->grid_check_counter,1);
                            assert(glid<bench->grid_check_capacity);
                            bench->grid_check[glid].pid = pid;
                            bench->grid_check[glid].gid = gid;
                            bench->grid_check[glid].offset = 0;
                            bench->grid_check[glid].inside = true;
                        }else if(p->y<bench->schema[child_offset].mbr.low[1]||
                                 (p->y<bench->schema[child_offset].mbr.high[1]
                                  && p->x<bench->schema[child_offset].mbr.low[0])){
                            uint glid = atomicAdd(&bench->grid_check_counter,1);
                            assert(glid<bench->grid_check_capacity);
                            bench->grid_check[glid].pid = pid;
                            bench->grid_check[glid].gid = gid;
                            bench->grid_check[glid].offset = 0;
                            bench->grid_check[glid].inside = false;
                        }
                    }else{
                        stack_index = atomicAdd(cur_stack_idx, 1);
                        //printf("%d:\tnew push %d\n",threadIdx.x,stack_index);
                        assert(stack_index<PER_STACK_SIZE*1024);
                        cur_stack[2*stack_index] = pid;
                        cur_stack[2*stack_index+1] = child_offset;
                    }
                }
            }
            atomicSub(cur_worker_idx, 1);
        }
        __syncthreads();
    }
}


/*
 *
 * kernel functions for the refinement step
 *
 * */

__global__
void cuda_unroll(workbench *bench, uint inistial_size){
    int glid = blockIdx.x*blockDim.x+threadIdx.x;
    if(glid>=inistial_size){
        return;
    }

    uint grid_size = min(bench->grid_counter[bench->grid_check[glid].gid],bench->grid_capacity);
    // the first batch already inserted during the partition and lookup steps
    uint offset = bench->config->zone_capacity;
    while(offset<grid_size){
        uint cu_index = atomicAdd(&bench->grid_check_counter, 1);
        if(cu_index>=bench->grid_check_capacity){
            printf("%d %d %d\n",bench->grid_counter[bench->grid_check[glid].gid],cu_index,bench->grid_check_capacity);
        }
        //assert(cu_index<bench->grid_check_capacity);
        bench->grid_check[cu_index] = bench->grid_check[glid];
        bench->grid_check[cu_index].offset = offset;
        offset += bench->config->zone_capacity;
    }
}


__global__
void cuda_refinement(workbench *bench){

    // the objects in which grid need be processed
    int loc = threadIdx.y;
    int pairid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pairid>=bench->grid_check_counter){
        return;
    }

    uint gid = bench->grid_check[pairid].gid;
    uint offset = bench->grid_check[pairid].offset;

    uint size = min(bench->grid_counter[gid],bench->grid_capacity)-offset;
    if(bench->config->unroll && size>bench->config->zone_capacity){
        size = bench->config->zone_capacity;
    }
    if(loc>=size){
        return;
    }
    uint pid = bench->grid_check[pairid].pid;
    uint target_pid = *(bench->grids+bench->grid_capacity*gid+offset+loc);
    if(!bench->grid_check[pairid].inside||pid<target_pid){
//        Point *p1 = &bench->points[pid];
//        Point *p2 = &bench->points[target_pid];
        Point *p1 = bench->points+pid;
        Point *p2 = bench->points+target_pid;
        double dist = distance(bench->points[pid].x, bench->points[pid].y, bench->points[target_pid].x, bench->points[target_pid].y);
        if(dist<=bench->config->reach_distance){
            uint pid1 = min(pid,target_pid);
            uint pid2 = max(target_pid,pid);
            size_t key = ((size_t)pid1+pid2)*(pid1+pid2+1)/2+pid2;
            if(bench->config->save_meetings_pers){
                uint meeting_idx = atomicAdd(&bench->num_active_meetings,1);
                bench->d_meetings_ps[meeting_idx].key = key;
                bench->d_meetings_ps[meeting_idx].start = bench->cur_time;
                bench->d_meetings_ps[meeting_idx].end = bench->cur_time;
                bench->d_meetings_ps[meeting_idx].mbr.low[0] = 100000.0;
                bench->d_meetings_ps[meeting_idx].mbr.low[1] = 100000.0;
                bench->d_meetings_ps[meeting_idx].mbr.high[0] = -100000.0;
                bench->d_meetings_ps[meeting_idx].mbr.high[1] = -100000.0;
                mbr_update_by_point(bench->d_meetings_ps[meeting_idx].mbr, p1);
                mbr_update_by_point(bench->d_meetings_ps[meeting_idx].mbr, p2);
            }

            size_t slot = key%bench->config->num_meeting_buckets;
            int ite = 0;
            while (ite++<5){
                unsigned long long prev = atomicCAS((unsigned long long *)&bench->meeting_buckets[slot].key, ULL_MAX, (unsigned long long)key);
                //printf("%ld\n",prev,ULL_MAX,bench->meeting_buckets[bench->current_bucket][slot].key);
                if(prev == key){
                    bench->meeting_buckets[slot].end = bench->cur_time;
                    //mbr_update(bench->meeting_buckets[slot].mbr, bench->points[pid]);                     //"Point::~Point"
                    mbr_update_by_point(bench->meeting_buckets[slot].mbr, p1);
                    mbr_update_by_point(bench->meeting_buckets[slot].mbr, p2);
                    break;
                }else if (prev == ULL_MAX){
                    bench->meeting_buckets[slot].key = key;
                    bench->meeting_buckets[slot].start = bench->cur_time;
                    bench->meeting_buckets[slot].end = bench->cur_time;
                    bench->meeting_buckets[slot].mbr.low[0] = 100000.0;
                    bench->meeting_buckets[slot].mbr.low[1] = 100000.0;
                    bench->meeting_buckets[slot].mbr.high[0] = -100000.0;
                    bench->meeting_buckets[slot].mbr.high[1] = -100000.0;
                    mbr_update_by_point(bench->meeting_buckets[slot].mbr, p1);
                    mbr_update_by_point(bench->meeting_buckets[slot].mbr, p2);

                    break;
                }
                slot = (slot + 1)%bench->config->num_meeting_buckets;
            }
        }
    }
}

__global__
void cuda_refinement_unroll(workbench *bench, uint offset){

    // the objects in which grid need be processed
    int loc = threadIdx.y;
    int pairid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pairid>=bench->grid_check_counter){
        return;
    }

    uint gid = bench->grid_check[pairid].gid;

    uint size = min(bench->grid_counter[gid],bench->grid_capacity);
    if(loc+offset>=size){
        return;
    }
    uint pid = bench->grid_check[pairid].pid;
    uint target_pid = *(bench->grids+bench->grid_capacity*gid+offset+loc);
    if(!bench->grid_check[pairid].inside||pid<target_pid){
        double dist = distance(bench->points[pid].x, bench->points[pid].y, bench->points[target_pid].x, bench->points[target_pid].y);
        if(dist<=bench->config->reach_distance){
            uint pid1 = min(pid,target_pid);
            uint pid2 = max(target_pid,pid);
            size_t key = ((size_t)pid1+pid2)*(pid1+pid2+1)/2+pid2;
            size_t slot = key%bench->config->num_meeting_buckets;
            int ite = 0;
            while (ite++<5){
                unsigned long long prev = atomicCAS((unsigned long long *)&bench->meeting_buckets[slot].key, ULL_MAX, (unsigned long long)key);
                //printf("%ld\n",prev,ULL_MAX,bench->meeting_buckets[bench->current_bucket][slot].key);
                if(prev == key){
                    bench->meeting_buckets[slot].end = bench->cur_time;
                    break;
                }else if (prev == ULL_MAX){
                    bench->meeting_buckets[slot].key = key;
                    bench->meeting_buckets[slot].start = bench->cur_time;
                    bench->meeting_buckets[slot].end = bench->cur_time;
                    break;
                }
                slot = (slot + 1)%bench->config->num_meeting_buckets;
            }
        }
    }
}

__global__
void insert_meetings_to_buckets(workbench *bench){
    size_t mid = blockIdx.x*blockDim.x + threadIdx.x;
    size_t slot = bench->d_meetings_ps[mid].key % bench->config->num_meeting_buckets;
    int ite = 0;
    while (ite++<5){
        unsigned long long prev = atomicCAS((unsigned long long *)&bench->meeting_buckets[slot].key, ULL_MAX, (unsigned long long)bench->d_meetings_ps[mid].key);
        //printf("%ld\n",prev,ULL_MAX,bench->meeting_buckets[bench->current_bucket][slot].key);
        if(prev == bench->d_meetings_ps[mid].key){
            bench->meeting_buckets[slot].end = bench->cur_time;
            mbr_update_by_mbr(bench->meeting_buckets[slot].mbr, bench->d_meetings_ps[mid].mbr);
            break;
        }else if (prev == ULL_MAX){
            bench->meeting_buckets[slot].key = bench->d_meetings_ps[mid].key;
            bench->meeting_buckets[slot].start = bench->cur_time;
            bench->meeting_buckets[slot].end = bench->cur_time;
            bench->meeting_buckets[slot].mbr = bench->d_meetings_ps[mid].mbr;
            break;
        }
        slot = (slot + 1)%bench->config->num_meeting_buckets;
    }

}

/*
 * kernel function for identify completed meetings
 *
 * */

__global__
void cuda_profile_meetings(workbench *bench){

    size_t bid = blockIdx.x*blockDim.x+threadIdx.x;
    if(bid>=bench->config->num_meeting_buckets){
        return;
    }
    // empty
    if(bench->meeting_buckets[bid].key==ULL_MAX){
        return;
    }
    if(bench->config->profile){
        atomicAdd((unsigned long long *)&bench->num_taken_buckets, (unsigned long long)1);
    }
    // is still active
    if(bench->meeting_buckets[bid].end==bench->cur_time){
        if(bench->config->profile){
            atomicAdd((unsigned long long *)&bench->num_active_meetings, (unsigned long long)1);
        }
        return;
    }
}

__global__
void cuda_identify_meetings(workbench *bench) {
    size_t bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= bench->config->num_meeting_buckets) {
        return;
    }
    // empty
    if (bench->meeting_buckets[bid].key == ULL_MAX) {
        return;
    }
    if (bench->meeting_buckets[bid].end == bench->cur_time) {               // is still active
        if(bench->search_single) {
            if (bench->cur_time - bench->meeting_buckets[bid].start >= bench->config->min_meet_time + 1) {
                if (bench->search_single_pid == getpid1(bench->meeting_buckets[bid].key)) {
                    uint meeting_idx = atomicAdd(&bench->single_find_count, 1);
                    assert(bench->single_find_count < bench->config->search_single_capacity);
                    //bench->search_multi_list[meeting_idx].pid = bench->search_single_pid;
                    bench->search_single_list[meeting_idx].target = getpid2(bench->meeting_buckets[bid].key);
                    bench->search_single_list[meeting_idx].start = bench->meeting_buckets[bid].start;
                    bench->search_single_list[meeting_idx].end = bench->meeting_buckets[bid].end;                  //real end
                    bench->search_single_list[meeting_idx].low0 = bench->meeting_buckets[bid].mbr.low[0];
                    bench->search_single_list[meeting_idx].low1 = bench->meeting_buckets[bid].mbr.low[1];
                    bench->search_single_list[meeting_idx].high0 = bench->meeting_buckets[bid].mbr.high[0];
                    bench->search_single_list[meeting_idx].high1 = bench->meeting_buckets[bid].mbr.high[1];
                }
                else if (bench->search_single_pid == getpid2(bench->meeting_buckets[bid].key)) {
                    uint meeting_idx = atomicAdd(&bench->single_find_count, 1);
                    assert(bench->single_find_count < bench->config->search_single_capacity);
                    bench->search_single_list[meeting_idx].target = getpid1(bench->meeting_buckets[bid].key);
                    bench->search_single_list[meeting_idx].start = bench->meeting_buckets[bid].start;
                    bench->search_single_list[meeting_idx].end = bench->meeting_buckets[bid].end;                  //real end
                    bench->search_single_list[meeting_idx].low0 = bench->meeting_buckets[bid].mbr.low[0];
                    bench->search_single_list[meeting_idx].low1 = bench->meeting_buckets[bid].mbr.low[1];
                    bench->search_single_list[meeting_idx].high0 = bench->meeting_buckets[bid].mbr.high[0];
                    bench->search_single_list[meeting_idx].high1 = bench->meeting_buckets[bid].mbr.high[1];
                }
            }
        }
        if(bench->search_multi) {
            if (bench->cur_time - bench->meeting_buckets[bid].start >= bench->config->min_meet_time + 1) {
                for(int i = 0;i<bench->search_multi_length;i++){
                    if (bench->search_multi_pid[i] == getpid1(bench->meeting_buckets[bid].key)) {
                        uint meeting_idx = atomicAdd(&bench->multi_find_count, 1);
                        assert(bench->multi_find_count < bench->config->search_multi_capacity);
                        bench->search_multi_list[meeting_idx].pid = bench->search_multi_pid[i];
                        bench->search_multi_list[meeting_idx].target = getpid2(bench->meeting_buckets[bid].key);
                        bench->search_multi_list[meeting_idx].start = bench->meeting_buckets[bid].start;
                        bench->search_multi_list[meeting_idx].end = bench->meeting_buckets[bid].end;                  //real end
                        bench->search_multi_list[meeting_idx].low0 = bench->meeting_buckets[bid].mbr.low[0];
                        bench->search_multi_list[meeting_idx].low1 = bench->meeting_buckets[bid].mbr.low[1];
                        bench->search_multi_list[meeting_idx].high0 = bench->meeting_buckets[bid].mbr.high[0];
                        bench->search_multi_list[meeting_idx].high1 = bench->meeting_buckets[bid].mbr.high[1];
                    }
                    if (bench->search_multi_pid[i] == getpid2(bench->meeting_buckets[bid].key)) {
                        uint meeting_idx = atomicAdd(&bench->multi_find_count, 1);
                        assert(bench->multi_find_count < bench->config->search_multi_capacity);
                        bench->search_multi_list[meeting_idx].pid = bench->search_multi_pid[i];
                        bench->search_multi_list[meeting_idx].target = getpid1(bench->meeting_buckets[bid].key);
                        bench->search_multi_list[meeting_idx].start = bench->meeting_buckets[bid].start;
                        bench->search_multi_list[meeting_idx].end = bench->meeting_buckets[bid].end;                  //real end
                        bench->search_multi_list[meeting_idx].low0 = bench->meeting_buckets[bid].mbr.low[0];
                        bench->search_multi_list[meeting_idx].low1 = bench->meeting_buckets[bid].mbr.low[1];
                        bench->search_multi_list[meeting_idx].high0 = bench->meeting_buckets[bid].mbr.high[0];
                        bench->search_multi_list[meeting_idx].high1 = bench->meeting_buckets[bid].mbr.high[1];
                    }
                }
            }
        }
        return;
    }
    if (bench->cur_time - bench->meeting_buckets[bid].start >= bench->config->min_meet_time + 1) {
        //printf("%d %d\n", bench->cur_time, bench->meeting_buckets[bid].start);
        if(!box_contain(&bench->mbr, &bench->meeting_buckets[bid].mbr)){
            bench->meeting_buckets[bid].key = ULL_MAX;
            return;
        }

//        float longer_edge = max(bench->meeting_buckets[bid].mbr.high[1] - bench->meeting_buckets[bid].mbr.low[1] , bench->meeting_buckets[bid].mbr.high[0] - bench->meeting_buckets[bid].mbr.low[0]);
//        if(longer_edge > 0.007){
//            bench->meeting_buckets[bid].key = ULL_MAX;
//            return;
//        }

//        float area = (bench->meeting_buckets[bid].mbr.high[1] - bench->meeting_buckets[bid].mbr.low[1])*(bench->meeting_buckets[bid].mbr.high[0] - bench->meeting_buckets[bid].mbr.low[0]);
//        //area
//        if(area > 0.00005){              //0.007*0.007
//            bench->meeting_buckets[bid].key = ULL_MAX;
//            return;
//        }

        uint duration = bench->meeting_buckets[bid].end - bench->meeting_buckets[bid].start;
        if(duration >= 990){
            if(duration < 2000){
                atomicAdd(&bench->larger_than_1000s, 1);
            }
            if(duration >= 2000 && duration < 3000){
                atomicAdd(&bench->larger_than_2000s, 1);
            }
            if(duration >= 3000 && duration < 4000){
                atomicAdd(&bench->larger_than_3000s, 1);
            }
            if(duration >= 4000){
                atomicAdd(&bench->larger_than_4000s, 1);
            }
        }

//        atomicMin(&bench->start_time_min,bench->meeting_buckets[bid].start);
//        atomicMax(&bench->start_time_max,bench->meeting_buckets[bid].start);

        uint pid, target;
        pid = getpid1(bench->meeting_buckets[bid].key);
        target = getpid2(bench->meeting_buckets[bid].key);

        if(duration > bench->config->max_meet_time){
            atomicAdd(&bench->oversize_oid_count, 1);
            bench->d_sids[pid] = 1;              //over size
            bench->d_sids[target] = 1;              //over size
        }

        uint meeting_idx = atomicAdd(&bench->kv_count, 2);
        assert(meeting_idx < bench->config->kv_capacity);

        for (int k = 0; k < 2; k++) {
            if(k==1){
                meeting_idx++;
                uint swap = pid;
                pid = target;
                target = swap;
            }
            bench->d_keys[meeting_idx] = ((__uint128_t)pid << (OID_BIT + DURATION_BIT + END_BIT)) + ((__uint128_t)target << (DURATION_BIT + END_BIT)) +
                                         ((__uint128_t)(bench->meeting_buckets[bid].end - bench->meeting_buckets[bid].start) << END_BIT) +
                                         (__uint128_t)(bench->meeting_buckets[bid].end - bench->end_time_min);
            write_kv_box(&bench->kv_boxs[meeting_idx], &bench->meeting_buckets[bid].mbr);

            uint if_zero = atomicAdd(&bench->same_pid_count[pid], 1);                   //but mid_xys shuold also atomic get and write
            if(!if_zero){
                atomicAdd(&bench->sid_count, 1);
            }
            atomic_mbr_update_by_mbr(bench->o_boxs[pid], bench->meeting_buckets[bid].mbr);
        }
    }
    bench->meeting_buckets[bid].key = ULL_MAX;
}

__global__
void set_oid(workbench *bench){                         //0~10000000
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= bench->config->num_objects){
        return;
    }
    bench->d_oids[id] = id;
    if(bench->same_pid_count[id] >= 1){
        float low0 = uint_to_float(bench->o_boxs[id].low[0]);
        float high0 = uint_to_float(bench->o_boxs[id].high[0]);
        float low1 = uint_to_float(bench->o_boxs[id].low[1]);
        float high1 = uint_to_float(bench->o_boxs[id].high[1]);

        float area = (high1 - low1)*(high0 - low0);
        if(area > 0.00005){         //0.00005
            //printf("%d %d %d %d  %d real cut id ,area %f\n",bench->o_boxs[id].low[0], bench->o_boxs[id].high[0], bench->o_boxs[id].low[1], bench->o_boxs[id].high[1], id, area);
            atomicAdd(&bench->oversize_oid_count, 1);
            bench->d_sids[id] = 1;              //over size
            return;
        }

//        float longer_edge = max(high1 - low1 , high0 - low0);
//        bench->d_longer_edges[id] = longer_edge;

        uint ave_mid0 = bench->o_boxs[id].low[0] / 2 + bench->o_boxs[id].high[0] / 2;
        uint ave_mid1 = bench->o_boxs[id].high[1] / 2 + bench->o_boxs[id].high[1] / 2;
        bench->mid_xys[id] = ( (uint64_t)ave_mid0 << 32 ) + (uint64_t)ave_mid1;
    }
}

__global__
void narrow_xy_to_y(workbench *bench){
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    uint zero_one_count = (bench->config->num_objects - bench->sid_count) + bench->oversize_oid_count;
    if(id >= bench->config->num_objects || id < zero_one_count){
        return;
    }
    bench->mid_xys[id] = bench->mid_xys[id] & ((1ULL << 32) - 1);     //xy -> y
}

__global__
void make_sid(workbench *bench) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    uint zero_one_count = (bench->config->num_objects - bench->sid_count) + bench->oversize_oid_count;
    if(id >= bench->config->num_objects || id < zero_one_count){
        return;
    }
    //uint part_key_capacity = bench->sid_count / bench->config->split_num / bench->config->split_num;              //wrong
    uint x_part_capacity = (bench->sid_count - bench->oversize_oid_count) / bench->config->split_num + 1;                   //x_part_capacity
    uint part_key_capacity = x_part_capacity / bench->config->split_num + 1;
    uint x_index = (id - zero_one_count) / x_part_capacity;
    uint y_index = (id - zero_one_count - x_index * x_part_capacity) / part_key_capacity;
    uint sid = x_index * bench->config->split_num + y_index + 2;          //0 -> not appear, 1 -> too large
    uint oid = bench->d_oids[id];
    bench->d_sids[oid] = sid;
    //printf("%d %d\n", sid, oid);
}

__global__
void write_sid_in_key(workbench *bench) {
    uint kid = blockIdx.x * blockDim.x + threadIdx.x;
    if (kid >= bench->kv_count) {
        return;
    }
    uint oid = cuda_get_key_oid(bench->d_keys[kid]);
    bench->d_keys[kid] = (bench->d_keys[kid] & (((__uint128_t)1 << (OID_BIT*2 + DURATION_BIT + END_BIT)) - 1))
                         + ((__uint128_t)bench->d_sids[oid] << (OID_BIT*2 + DURATION_BIT + END_BIT));
}

__global__
void get_CTF_capacity(workbench *bench) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    uint zero_one_count = (bench->config->num_objects - bench->sid_count) + bench->oversize_oid_count;
    if(id >= bench->config->num_objects || id < zero_one_count){
        return;
    }
    uint oid = bench->d_oids[id];
    uint CTF_id = bench->d_sids[oid] - 2;
    atomicAdd(&bench->d_CTF_capacity[CTF_id], bench->same_pid_count[oid]);
}

__global__
void cuda_get_limit(workbench *bench) {
    uint ctf_id = blockIdx.x;
    if (ctf_id >= bench->config->CTF_count) {
        return;
    }
    __shared__ int key_index;
    key_index = 0;
    if (threadIdx.x == 0) {
        for(uint i = 0; i < ctf_id; i++){
            key_index += bench->d_CTF_capacity[i];
        }
    }
    __syncthreads();

    // Shared memory for reduction
    __shared__ volatile float local_low[2][BLOCK_DIM];
    __shared__ volatile float local_high[2][BLOCK_DIM];
    __shared__ volatile int local_start_time_min[BLOCK_DIM];
    __shared__ volatile int local_start_time_max[BLOCK_DIM];
    uint kid;
    float key_low0 = 100000.0, key_low1 = 100000.0, key_high0 = -100000.0, key_high1 = -100000.0;
    uint duration = 0, end = 0, start_tim_min = 1 << 30, start_tim_max = 0;
    for (kid = key_index + threadIdx.x; kid < key_index + bench->d_CTF_capacity[ctf_id]; kid += BLOCK_DIM) {
        key_low0 = min(key_low0, bench->kv_boxs[kid].low[0]);
        key_low1 = min(key_low1, bench->kv_boxs[kid].low[1]);
        key_high0 = max(key_high0, bench->kv_boxs[kid].high[0]);
        key_high1 = max(key_high1, bench->kv_boxs[kid].high[1]);
        duration = cuda_get_key_duration(bench->d_keys[kid]);
        end = cuda_get_key_end(bench->d_keys[kid]) + bench->end_time_min;
        assert(end >= duration);
        start_tim_min = min(start_tim_min, end - duration);
        start_tim_max = max(start_tim_max, end - duration);
    }

    // Load data into shared memory
    local_low[0][threadIdx.x] = key_low0;
    local_low[1][threadIdx.x] = key_low1;
    local_high[0][threadIdx.x] = key_high0;
    local_high[1][threadIdx.x] = key_high1;
    local_start_time_min[threadIdx.x] = start_tim_min;
    local_start_time_max[threadIdx.x] = start_tim_max;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = BLOCK_DIM / 2; offset > 32; offset /= 2) {
        if (threadIdx.x < offset) {
            local_low[0][threadIdx.x] = min(local_low[0][threadIdx.x], local_low[0][threadIdx.x + offset]);
            local_low[1][threadIdx.x] = min(local_low[1][threadIdx.x], local_low[1][threadIdx.x + offset]);
            local_high[0][threadIdx.x] = max(local_high[0][threadIdx.x], local_high[0][threadIdx.x + offset]);
            local_high[1][threadIdx.x] = max(local_high[1][threadIdx.x], local_high[1][threadIdx.x + offset]);
            local_start_time_min[threadIdx.x] = min(local_start_time_min[threadIdx.x], local_start_time_min[threadIdx.x + offset]);
            local_start_time_max[threadIdx.x] = max(local_start_time_max[threadIdx.x], local_start_time_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, true); // 确保所有 warp 线程参与
        // 使用 __shfl_down_sync 实现 warp 归约
        for (int offset = 16; offset > 0; offset /= 2) {
            local_low[0][threadIdx.x] = min(local_low[0][threadIdx.x], __shfl_down_sync(mask, local_low[0][threadIdx.x], offset));
            local_low[1][threadIdx.x] = min(local_low[1][threadIdx.x], __shfl_down_sync(mask, local_low[1][threadIdx.x], offset));
            local_high[0][threadIdx.x] = max(local_high[0][threadIdx.x], __shfl_down_sync(mask, local_high[0][threadIdx.x], offset));
            local_high[1][threadIdx.x] = max(local_high[1][threadIdx.x], __shfl_down_sync(mask, local_high[1][threadIdx.x], offset));
            local_start_time_min[threadIdx.x] = min(local_start_time_min[threadIdx.x], __shfl_down_sync(mask, local_start_time_min[threadIdx.x], offset));
            local_start_time_max[threadIdx.x] = max(local_start_time_max[threadIdx.x], __shfl_down_sync(mask, local_start_time_max[threadIdx.x], offset));
        }
    }
    __syncthreads();
    // Write results back to global memory
    if (threadIdx.x == 0) {
        bench->d_ctfs[ctf_id].CTF_kv_capacity = bench->d_CTF_capacity[ctf_id];
        bench->d_ctfs[ctf_id].start_time_min = local_start_time_min[0];
        bench->d_ctfs[ctf_id].start_time_max = local_start_time_max[0];
        bench->d_ctfs[ctf_id].end_time_min = bench->end_time_min;
        bench->d_ctfs[ctf_id].end_time_max = bench->cur_time;
        bench->d_ctfs[ctf_id].ctf_mbr.low[0] = local_low[0][0];
        bench->d_ctfs[ctf_id].ctf_mbr.low[1] = local_low[1][0];
        bench->d_ctfs[ctf_id].ctf_mbr.high[0] = local_high[0][0];
        bench->d_ctfs[ctf_id].ctf_mbr.high[1] = local_high[1][0];
    }
}

__global__
void cuda_write_o_bitmap(workbench *bench, uint oversize_key_count){
    uint kid = blockIdx.x * blockDim.x + threadIdx.x;
    if(kid >= oversize_key_count){
        return;
    }
    kid += bench->kv_count;

    uint low0 = (bench->kv_boxs[kid].low[0] - bench->mbr.low[0])/(bench->mbr.high[0] - bench->mbr.low[0]) * DEFAULT_bitmap_edge;
    uint low1 = (bench->kv_boxs[kid].low[1] - bench->mbr.low[1])/(bench->mbr.high[1] - bench->mbr.low[1]) * DEFAULT_bitmap_edge;
    uint high0 = (bench->kv_boxs[kid].high[0] - bench->mbr.low[0])/(bench->mbr.high[0] - bench->mbr.low[0]) * DEFAULT_bitmap_edge;
    uint high1 = (bench->kv_boxs[kid].high[1] - bench->mbr.low[1])/(bench->mbr.high[1] - bench->mbr.low[1]) * DEFAULT_bitmap_edge;
    uint bit_pos = 0;
    for(uint i=low0;i<=high0;i++){
        for(uint j=low1;j<=high1;j++){
            bit_pos = i + j * DEFAULT_bitmap_edge;
            unsigned int *bitmap_ptr = reinterpret_cast<unsigned int *>(bench->d_bitmaps);
            atomicOr(&bitmap_ptr[0 + bit_pos / 32], (1 << (bit_pos % 32)));
        }
    }
}

__global__
void get_o_limit(workbench *bench, uint oversize_key_count) {
    uint ctf_id = 0;
    __shared__ volatile int local_start_time_min[BLOCK_DIM];
    __shared__ volatile int local_start_time_max[BLOCK_DIM];
    __shared__ volatile int local_end_time_min[BLOCK_DIM];
    __shared__ volatile int local_end_time_max[BLOCK_DIM];
    __syncthreads();
    uint kid;
    uint duration = 0, end = 0, start_tim_min = 1 << 30, start_tim_max = 0, end_min = 1 << 30, end_max = 0;
    for (uint i = threadIdx.x; i < oversize_key_count; i += BLOCK_DIM) {
        kid = bench->kv_count + i;
        duration = cuda_get_key_duration(bench->d_keys[kid]);
        end = cuda_get_key_end(bench->d_keys[kid]) + bench->end_time_min;
        assert(end >= duration);
        end_min = min(end_min, end);
        end_max = max(end_max, end);
        start_tim_min = min(start_tim_min, end - duration);
        start_tim_max = max(start_tim_max, end - duration);
    }

    // Load data into shared memory
    local_end_time_min[threadIdx.x] = end_min;
    local_end_time_max[threadIdx.x] = end_max;
    local_start_time_min[threadIdx.x] = start_tim_min;
    local_start_time_max[threadIdx.x] = start_tim_max;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = BLOCK_DIM / 2; offset > 32; offset /= 2) {
        if (threadIdx.x < offset) {
            local_end_time_min[threadIdx.x] = min(local_end_time_min[threadIdx.x], local_end_time_min[threadIdx.x + offset]);
            local_end_time_max[threadIdx.x] = max(local_end_time_max[threadIdx.x], local_end_time_max[threadIdx.x + offset]);
            local_start_time_min[threadIdx.x] = min(local_start_time_min[threadIdx.x], local_start_time_min[threadIdx.x + offset]);
            local_start_time_max[threadIdx.x] = max(local_start_time_max[threadIdx.x], local_start_time_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, true); // 确保所有 warp 线程参与
        // 使用 __shfl_down_sync 实现 warp 归约
        for (int offset = 16; offset > 0; offset /= 2) {
            local_end_time_min[threadIdx.x] = min(local_end_time_min[threadIdx.x], __shfl_down_sync(mask, local_end_time_min[threadIdx.x], offset));
            local_end_time_max[threadIdx.x] = max(local_end_time_max[threadIdx.x], __shfl_down_sync(mask, local_end_time_max[threadIdx.x], offset));
            local_start_time_min[threadIdx.x] = min(local_start_time_min[threadIdx.x], __shfl_down_sync(mask, local_start_time_min[threadIdx.x], offset));
            local_start_time_max[threadIdx.x] = max(local_start_time_max[threadIdx.x], __shfl_down_sync(mask, local_start_time_max[threadIdx.x], offset));
        }
    }
    __syncthreads();
    // Write results back to global memory
    if (threadIdx.x == 0) {
        bench->d_ctfs[ctf_id].CTF_kv_capacity = bench->d_CTF_capacity[ctf_id];
        bench->d_ctfs[ctf_id].start_time_min = local_start_time_min[0];
        bench->d_ctfs[ctf_id].start_time_max = local_start_time_max[0];
        bench->d_ctfs[ctf_id].end_time_min = bench->end_time_min;
        bench->d_ctfs[ctf_id].end_time_max = bench->cur_time;
    }
}

__global__
void cuda_write_bitmap(workbench *bench){
    uint ctf_id = blockIdx.x;
    if(ctf_id >= bench->config->CTF_count){
        return;
    }
    __shared__ int key_index;
    key_index = 0;
    if (threadIdx.x == 0) {
        for(uint i = 0; i < ctf_id; i++){
            key_index += bench->d_CTF_capacity[i];
        }
    }
    __syncthreads();

    f_box * CTF_mbr = &bench->d_ctfs[ctf_id].ctf_mbr;
    for (uint kid = key_index + threadIdx.x; kid < key_index + bench->d_CTF_capacity[ctf_id]; kid += BLOCK_DIM) {
        uint low0 = (bench->kv_boxs[kid].low[0] - CTF_mbr->low[0])/(CTF_mbr->high[0] - CTF_mbr->low[0]) * bench->d_ctfs[ctf_id].x_grid;
        uint low1 = (bench->kv_boxs[kid].low[1] - CTF_mbr->low[1])/(CTF_mbr->high[1] - CTF_mbr->low[1]) * bench->d_ctfs[ctf_id].y_grid;
        uint high0 = (bench->kv_boxs[kid].high[0] - CTF_mbr->low[0])/(CTF_mbr->high[0] - CTF_mbr->low[0]) * bench->d_ctfs[ctf_id].x_grid;
        uint high1 = (bench->kv_boxs[kid].high[1] - CTF_mbr->low[1])/(CTF_mbr->high[1] - CTF_mbr->low[1]) * bench->d_ctfs[ctf_id].y_grid;
//        if(high1 >= 256){
//
//        }
        //assert(high1 < 256);

        uint bit_pos = 0;
        for(uint i=low0;i<=high0;i++){
            for(uint j=low1;j<=high1;j++){
                bit_pos = i + j * bench->d_ctfs[ctf_id].x_grid;
                unsigned int *bitmap_ptr = reinterpret_cast<unsigned int *>(bench->d_bitmaps);
                atomicOr(&bitmap_ptr[ctf_id * (bench->bit_count / 32) + bit_pos / 32], (1 << (bit_pos % 32)));
            }
        }
    }
}

//__global__
//void mbr_bitmap(workbench *bench){
//    __shared__ volatile int local_low[BLOCK_DIM][2];
//    __shared__ volatile int local_high[BLOCK_DIM][2];
//
//    uint temp_low[2] = {10000,10000};
//    uint temp_high[2] = {0, 0};
//    for (uint bit_pos = threadIdx.x; bit_pos < bench->bit_count; bit_pos += blockDim.x) {                   //bit_pos < bench->bit_count    //need overflow return
//        if ( bench->d_bitmaps[blockIdx.x*(bench->bit_count/8)+bit_pos/8] & (1<<(bit_pos%8)) ) {
//            uint temp[2];
//            d2xy(SID_BIT/2, bit_pos, temp[0], temp[1]);
//            temp_low[0] = min(temp_low[0], temp[0]);
//            temp_low[1] = min(temp_low[1], temp[1]);
//            temp_high[0] = max(temp_high[0], temp[0]);
//            temp_high[1] = max(temp_high[1], temp[1]);
//        }
//    }
//    local_low[threadIdx.x][0] = temp_low[0];
//    local_low[threadIdx.x][1] = temp_low[1];
//    local_high[threadIdx.x][0] = temp_high[0];
//    local_high[threadIdx.x][1] = temp_high[1];
//    __syncthreads();
//
//    for (int j = blockDim.x >> 1; j > 32; j >>= 1) {
//        if (threadIdx.x < j) {
//            local_low[threadIdx.x][0] = min(local_low[threadIdx.x][0], local_low[threadIdx.x + j][0]);
//            local_low[threadIdx.x][1] = min(local_low[threadIdx.x][1], local_low[threadIdx.x + j][1]);
//            local_high[threadIdx.x][0] = max(local_high[threadIdx.x][0], local_high[threadIdx.x + j][0]);
//            local_high[threadIdx.x][1] = max(local_high[threadIdx.x][1], local_high[threadIdx.x + j][1]);
//        }
//        __syncthreads();
//    }
//
//    if (threadIdx.x < 32) {
//        for(int j = 32; j >= 1; j >>= 1){
//            local_low[threadIdx.x][0] = min(local_low[threadIdx.x][0], local_low[threadIdx.x + j][0]);
//            local_low[threadIdx.x][1] = min(local_low[threadIdx.x][1], local_low[threadIdx.x + j][1]);
//            local_high[threadIdx.x][0] = max(local_high[threadIdx.x][0], local_high[threadIdx.x + j][0]);
//            local_high[threadIdx.x][1] = max(local_high[threadIdx.x][1], local_high[threadIdx.x + j][1]);
//        }
//    }
//    if (threadIdx.x == 0) {
//        bench->d_bitmap_mbrs[blockIdx.x].low[0] = (double)local_low[0][0]/(1ULL << (SID_BIT/2))*(bench->mbr.high[0] - bench->mbr.low[0]) + bench->mbr.low[0];
//        bench->d_bitmap_mbrs[blockIdx.x].low[1] = (double)local_low[0][1]/(1ULL << (SID_BIT/2))*(bench->mbr.high[1] - bench->mbr.low[1]) + bench->mbr.low[1];
//        bench->d_bitmap_mbrs[blockIdx.x].high[0] = (double)local_high[0][0]/(1ULL << (SID_BIT/2))*(bench->mbr.high[0] - bench->mbr.low[0]) + bench->mbr.low[0];
//        bench->d_bitmap_mbrs[blockIdx.x].high[1] = (double)local_high[0][1]/(1ULL << (SID_BIT/2))*(bench->mbr.high[1] - bench->mbr.low[1]) + bench->mbr.low[1];
//        //printf("bench->d_bitmap_mbrs[blockIdx.x].low[0] %lf\n",bench->d_bitmap_mbrs[blockIdx.x].low[0]);
//    }
//}

__global__
void write_key_and_mbr(workbench *bench){
    uint ctf_id = blockIdx.x;
    if(ctf_id >= bench->config->CTF_count){
        return;
    }

    __shared__ int key_index;
    __shared__ int bytes_index;
    key_index = 0, bytes_index = 0;
    if (threadIdx.x == 0) {
        for(uint i = 0; i < ctf_id; i++){
            key_index += bench->d_CTF_capacity[i];
            bytes_index += bench->d_CTF_capacity[i] * bench->d_ctfs[i].key_bit / 8;
        }
    }
    __syncthreads();
    CTF * ctf = &bench->d_ctfs[ctf_id];
    for (uint kid = key_index + threadIdx.x; kid < key_index + bench->d_CTF_capacity[ctf_id]; kid += BLOCK_DIM) {
        __uint128_t pid, target, duration, end;
        extract_fields(bench->d_keys[kid], pid, target, duration, end);
        // box = left_bottom + width + height
        __uint128_t value_mbr = serialize_mbr(&bench->kv_boxs[kid], &ctf->ctf_mbr, ctf);
        // key =  oid target duration end box
        __uint128_t temp_key = ((__uint128_t)pid << (ctf->id_bit + ctf->duration_bit + ctf->end_bit + ctf->mbr_bit)) + ((__uint128_t)target << (ctf->duration_bit + ctf->end_bit + ctf->mbr_bit))
                               + ((__uint128_t)duration << (ctf->end_bit + ctf->mbr_bit)) + ((__uint128_t)end << (ctf->mbr_bit)) + value_mbr;
        //uint ave_ctf_size = bench->config->kv_restriction / bench->config->CTF_count * sizeof(__uint128_t);
        uint key_Bytes = ctf->key_bit / 8;
        memcpy(&bench->d_ctf_keys[bytes_index + (kid - key_index) * key_Bytes], &temp_key, key_Bytes);
//        uint8_t *d_ctf_keys_ptr = &bench->d_ctf_keys[bytes_index + kid * key_Bytes];
//        for (uint i = 0; i < key_Bytes; i++) {
//            d_ctf_keys_ptr[i] = (temp_key >> (8 * (key_Bytes - 1 - i))) & 0xFF;
//        }
    }
}


/*
 * kernel functions for index update
 *
 * */
__global__
void cuda_update_schema_split(workbench *bench, uint size){
    uint sidx = blockIdx.x*blockDim.x+threadIdx.x;
    if(sidx>=size){
        return;
    }
    uint curnode = bench->split_list[sidx];
    //printf("split: %d\n",curnode);
    //schema[curnode].mbr.print();
    bench->schema[curnode].type = BRANCH;
    // reuse by one of its child
    uint gid = bench->schema[curnode].grid_id;

    double xhalf = bench->schema[curnode].mid_x-bench->schema[curnode].mbr.low[0];
    double yhalf = bench->schema[curnode].mid_y-bench->schema[curnode].mbr.low[1];

    for(int i=0;i<4;i++){
        // pop space for schema and grid
        uint idx = atomicAdd(&bench->schema_stack_index, 1);
        assert(idx<bench->schema_stack_capacity);
        uint child = bench->schema_stack[idx];
        //printf("sidx: %d %d\n",idx,child);
        bench->schema[curnode].children[i] = child;

        if(i>0){
            idx = atomicAdd(&bench->grids_stack_index,1);
            assert(idx<bench->grids_stack_capacity);
            gid = bench->grids_stack[idx];
            //printf("gidx: %d %d\n",idx,gid);
        }
        bench->schema[child].grid_id = gid;
        bench->grid_counter[gid] = 0;
        bench->schema[child].level = bench->schema[curnode].level+1;
        bench->schema[child].type = LEAF;
        bench->schema[child].overflow_count = 0;
        bench->schema[child].underflow_count = 0;

        bench->schema[child].mbr.low[0] = bench->schema[curnode].mbr.low[0]+(i%2==1)*xhalf;
        bench->schema[child].mbr.low[1] = bench->schema[curnode].mbr.low[1]+(i/2==1)*yhalf;
        bench->schema[child].mbr.high[0] = bench->schema[curnode].mid_x+(i%2==1)*xhalf;
        bench->schema[child].mbr.high[1] = bench->schema[curnode].mid_y+(i/2==1)*yhalf;
        bench->schema[child].mid_x = (bench->schema[child].mbr.low[0]+bench->schema[child].mbr.high[0])/2;
        bench->schema[child].mid_y = (bench->schema[child].mbr.low[1]+bench->schema[child].mbr.high[1])/2;
    }
}

__global__
void cuda_update_schema_merge(workbench *bench, uint size){
    uint sidx = blockIdx.x*blockDim.x+threadIdx.x;
    if(sidx>=size){
        return;
    }
    uint curnode = bench->merge_list[sidx];
    //reclaim the children
    uint gid = 0;
    for(int i=0;i<4;i++){
        uint child_offset = bench->schema[curnode].children[i];
        assert(bench->schema[child_offset].type==LEAF);
        //bench->schema[child_offset].mbr.print();
        // push the bench->schema and grid spaces to stack for reuse

        bench->grid_counter[bench->schema[child_offset].grid_id] = 0;
        if(i<3){
            // push to stack
            uint idx = atomicSub(&bench->grids_stack_index,1)-1;
            bench->grids_stack[idx] = bench->schema[child_offset].grid_id;
        }else{
            // reused by curnode
            gid = bench->schema[child_offset].grid_id;
        }
        bench->schema[child_offset].type = INVALID;
        uint idx = atomicSub(&bench->schema_stack_index,1)-1;
        bench->schema_stack[idx] = child_offset;
    }
    bench->schema[curnode].type = LEAF;
    // reuse the grid of one of its child
    bench->schema[curnode].grid_id = gid;
}

__global__
void cuda_update_schema_collect(workbench *bench){
    uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
    if(curnode>=bench->schema_stack_capacity){
        return;
    }
    if(bench->schema[curnode].type==LEAF){
        if(height(&bench->schema[curnode].mbr)>2*bench->config->reach_distance&&
           bench->grid_counter[bench->schema[curnode].grid_id]>bench->config->grid_capacity){
            // this node is overflowed a continuous number of times, split it
            if(++bench->schema[curnode].overflow_count>=bench->config->schema_update_delay){
                uint sidx = atomicAdd(&bench->split_list_index,1);
                bench->split_list[sidx] = curnode;
                bench->schema[curnode].overflow_count = 0;
            }
        }else{
            bench->schema[curnode].overflow_count = 0;
        }
    }else if(bench->schema[curnode].type==BRANCH){
        int leafchild = 0;
        int ncounter = 0;
        for(int i=0;i<4;i++){
            uint child_node = bench->schema[curnode].children[i];
            if(bench->schema[child_node].type==LEAF){
                leafchild++;
                ncounter += bench->grid_counter[bench->schema[child_node].grid_id];
            }
        }
        // this one need update
        if(leafchild==4&&ncounter<bench->config->grid_capacity){
            // this node need be merged
            if(++bench->schema[curnode].underflow_count>=bench->config->schema_update_delay){
                //printf("%d\n",curnode);
                uint sidx = atomicAdd(&bench->merge_list_index,1);
                bench->merge_list[sidx] = curnode;
                bench->schema[curnode].underflow_count = 0;
            }
        }else{
            bench->schema[curnode].underflow_count = 0;
        }
    }
}


__global__
void cuda_init_schema_stack(workbench *bench){
    uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
    if(curnode>=bench->schema_stack_capacity){
        return;
    }
    bench->schema_stack[curnode] = curnode;
}
__global__
void cuda_init_grids_stack(workbench *bench){
    uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
    if(curnode>=bench->grids_stack_capacity){
        return;
    }
    bench->grids_stack[curnode] = curnode;
}
__global__
void cuda_init_o_boxs(workbench *bench){
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id >= bench->config->num_objects){
        return;
    }
    bench->o_boxs[id].low[0] = 100000000;
    bench->o_boxs[id].low[1] = 100000000;
    bench->o_boxs[id].high[0] = 0;
    bench->o_boxs[id].high[1] = 0;
}



#define one_dim 16384
//#define one_dim 8

__global__
void cuda_build_qtree(workbench *bench){
    uint pid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=bench->config->num_objects){
        return;
    }
    uint x = (bench->points[pid].x-bench->mbr.low[0])/(bench->mbr.high[0]-bench->mbr.low[0])*one_dim;
    uint y = (bench->points[pid].y-bench->mbr.low[1])/(bench->mbr.high[1]-bench->mbr.low[1])*one_dim;
    atomicAdd(&bench->part_counter[x+y*one_dim],1);
}

__global__
void cuda_clean_cells(workbench *bench){
    uint pid = blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=one_dim*one_dim){
        return;
    }
    bench->schema_assigned[pid] = 0;
    bench->part_counter[pid] = 0;
    if(pid==0){
        bench->grids_stack_index = 0;
        bench->schema_stack_index = 1;
    }
}

__global__
void cuda_merge_qtree(workbench *bench, uint gap){
    uint pid = blockIdx.x*blockDim.x+threadIdx.x;
    uint xdim = one_dim/gap;
    if(pid>=(xdim*xdim)){
        return;
    }

    uint x = pid%xdim;
    uint y = pid/xdim;
    if(gap==1){
        if(bench->part_counter[pid]>bench->config->grid_capacity){
            uint node = atomicAdd(&bench->schema_stack_index,1);
            bench->schema[node].type = BRANCH;
            bench->schema[node].mbr.low[0] = bench->mbr.low[0]+x*(bench->mbr.high[0]-bench->mbr.low[0])/xdim;
            bench->schema[node].mbr.low[1] = bench->mbr.low[1]+y*(bench->mbr.high[1]-bench->mbr.low[1])/xdim;
            bench->schema[node].mbr.high[0] = bench->mbr.low[0]+(x+1)*(bench->mbr.high[0]-bench->mbr.low[0])/xdim;
            bench->schema[node].mbr.high[1] = bench->mbr.low[1]+(y+1)*(bench->mbr.high[1]-bench->mbr.low[1])/xdim;
            bench->schema[node].mid_x = (bench->schema[node].mbr.low[0]+bench->schema[node].mbr.high[0])/2;
            bench->schema[node].mid_y = (bench->schema[node].mbr.low[1]+bench->schema[node].mbr.high[1])/2;
            double xhalf = bench->schema[node].mid_x-bench->schema[node].mbr.low[0];
            double yhalf = bench->schema[node].mid_y-bench->schema[node].mbr.low[1];

            for(uint i=0;i<4;i++){
                uint cnode = atomicAdd(&bench->schema_stack_index,1);
                bench->schema[cnode].grid_id = atomicAdd(&bench->grids_stack_index,1);
                bench->schema[cnode].type = LEAF;
                bench->schema[node].children[i] = cnode;
                bench->grid_counter[bench->schema[cnode].grid_id] = 0;
                bench->schema[cnode].type = LEAF;
                bench->schema[cnode].overflow_count = 0;
                bench->schema[cnode].underflow_count = 0;
                bench->schema[cnode].mbr.low[0] = bench->schema[node].mbr.low[0]+(i%2==1)*xhalf;
                bench->schema[cnode].mbr.low[1] = bench->schema[node].mbr.low[1]+(i/2==1)*yhalf;
                bench->schema[cnode].mbr.high[0] = bench->schema[node].mid_x+(i%2==1)*xhalf;
                bench->schema[cnode].mbr.high[1] = bench->schema[node].mid_y+(i/2==1)*yhalf;
                bench->schema[cnode].mid_x = (bench->schema[cnode].mbr.low[0]+bench->schema[cnode].mbr.high[0])/2;
                bench->schema[cnode].mid_y = (bench->schema[cnode].mbr.low[1]+bench->schema[cnode].mbr.high[1])/2;
                //print_box(&bench->schema[cnode].mbr);
            }
            bench->schema_assigned[pid] = node;
        }
    }else{

        uint step = gap/2;
        uint p[4];
        p[0] = y*gap*one_dim+x*gap;
        p[1] = y*gap*one_dim+x*gap+step;
        p[2] = y*gap*one_dim+step*one_dim+x*gap;
        p[3] = y*gap*one_dim+step*one_dim+x*gap+step;
        uint size = 0;
        for(uint i=0;i<4;i++){
            size += bench->part_counter[p[i]];
        }
        // parent node
        if(size>bench->config->grid_capacity){
            uint node = 0;
            // node 0 is for the root only
            if(xdim!=1){
                node = atomicAdd(&bench->schema_stack_index,1);
            }
            bench->schema[node].type = BRANCH;
            bench->schema[node].mbr.low[0] = bench->mbr.low[0]+x*(bench->mbr.high[0]-bench->mbr.low[0])/xdim;
            bench->schema[node].mbr.low[1] = bench->mbr.low[1]+y*(bench->mbr.high[1]-bench->mbr.low[1])/xdim;
            bench->schema[node].mbr.high[0] = bench->mbr.low[0]+(x+1)*(bench->mbr.high[0]-bench->mbr.low[0])/xdim;
            bench->schema[node].mbr.high[1] = bench->mbr.low[1]+(y+1)*(bench->mbr.high[1]-bench->mbr.low[1])/xdim;
            bench->schema[node].mid_x = (bench->schema[node].mbr.low[0]+bench->schema[node].mbr.high[0])/2;
            bench->schema[node].mid_y = (bench->schema[node].mbr.low[1]+bench->schema[node].mbr.high[1])/2;

            double xhalf = bench->schema[node].mid_x-bench->schema[node].mbr.low[0];
            double yhalf = bench->schema[node].mid_y-bench->schema[node].mbr.low[1];
            for(uint i=0;i<4;i++){
                uint cnode = 0;
                if(bench->schema_assigned[p[i]]!=0){
                    cnode = bench->schema_assigned[p[i]];
                }else{
                    cnode = atomicAdd(&bench->schema_stack_index,1);
                    bench->schema[cnode].grid_id = atomicAdd(&bench->grids_stack_index,1);
                    bench->schema[cnode].type = LEAF;
                    bench->grid_counter[bench->schema[cnode].grid_id] = 0;
                    bench->schema[cnode].type = LEAF;
                    bench->schema[cnode].overflow_count = 0;
                    bench->schema[cnode].underflow_count = 0;
                    bench->schema[cnode].mbr.low[0] = bench->schema[node].mbr.low[0]+(i%2==1)*xhalf;
                    bench->schema[cnode].mbr.low[1] = bench->schema[node].mbr.low[1]+(i/2==1)*yhalf;
                    bench->schema[cnode].mbr.high[0] = bench->schema[node].mid_x+(i%2==1)*xhalf;
                    bench->schema[cnode].mbr.high[1] = bench->schema[node].mid_y+(i/2==1)*yhalf;
                    bench->schema[cnode].mid_x = (bench->schema[cnode].mbr.low[0]+bench->schema[cnode].mbr.high[0])/2;
                    bench->schema[cnode].mid_y = (bench->schema[cnode].mbr.low[1]+bench->schema[cnode].mbr.high[1])/2;
                    //print_box(&bench->schema[cnode].mbr);
                }
                bench->schema[node].children[i] = cnode;
            }
            bench->schema_assigned[p[0]] = node;
        }
        // for next upper level
        bench->part_counter[p[0]] = size;
    }
}

workbench *cuda_create_device_bench(workbench *bench, gpu_info *gpu){
    log("GPU memory:");
    struct timeval start = get_cur_time();
    gpu->clear();
    // use h_bench as a container to copy in and out GPU
    workbench h_bench(bench);
    // space for the raw points data
    h_bench.points = (Point *)gpu->allocate(bench->config->num_objects*sizeof(Point));
    size_t size = bench->config->num_objects*sizeof(Point);
    log("\t%.2f MB\tpoints",1.0*size/1024/1024);

    // space for the pids of all the grids
    h_bench.grids = (uint *)gpu->allocate(bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint));
    h_bench.grid_counter = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));
    h_bench.grids_stack = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));
    size = bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint)+bench->grids_stack_capacity*sizeof(uint)+bench->grids_stack_capacity*sizeof(uint);
    log("\t%.2f MB\tgrids",1.0*size/1024/1024);

    // space for the QTtree schema
    h_bench.schema = (QTSchema *)gpu->allocate(bench->schema_stack_capacity*sizeof(QTSchema));
    h_bench.schema_stack = (uint *)gpu->allocate(bench->schema_stack_capacity*sizeof(uint));
    size = bench->schema_stack_capacity*sizeof(QTSchema)+bench->schema_stack_capacity*sizeof(uint);
    log("\t%.2f MB\tschema",1.0*size/1024/1024);

    // space for the pid-zid pairs
    h_bench.grid_check = (checking_unit *)gpu->allocate(bench->grid_check_capacity*sizeof(checking_unit));
    size = bench->grid_check_capacity*sizeof(checking_unit);
    log("\t%.2f MB\trefine list",1.0*size/1024/1024);

    size = 2*bench->filter_list_capacity*sizeof(uint);
    h_bench.filter_list = (uint *)gpu->allocate(size);
    log("\t%.2f MB\tfiltering list",1.0*size/1024/1024);

    // space for processing stack
    h_bench.tmp_space = (uint *)gpu->allocate(bench->tmp_space_capacity*sizeof(uint));
    size = bench->tmp_space_capacity*sizeof(uint);
    h_bench.merge_list = h_bench.tmp_space;
    h_bench.split_list = h_bench.tmp_space+bench->tmp_space_capacity/2;
    log("\t%.2f MB\ttemporary space",1.0*size/1024/1024);

    h_bench.meeting_buckets = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(meeting_unit));
    size = bench->config->num_meeting_buckets*sizeof(meeting_unit);
    log("\t%.2f MB\thash table",1.0*size/1024/1024);

    h_bench.part_counter = (uint *)gpu->allocate(one_dim*one_dim*sizeof(uint));
    size = one_dim*one_dim*sizeof(uint);
    log("\t%.2f MB\tpart_counter",1.0*size/1024/1024);
    h_bench.schema_assigned = (uint *)gpu->allocate(one_dim*one_dim*sizeof(uint));
    size = one_dim*one_dim*sizeof(uint);
    log("\t%.2f MB\tschema_assigned",1.0*size/1024/1024);
    logt("old Glint GPU allocating space %ld MB", start,gpu->size_allocated()/1024/1024);

    if(bench->config->save_meetings_pers || bench->config->load_meetings_pers){
        h_bench.d_meetings_ps = (meeting_unit *)gpu->allocate(bench->config->num_objects / 10 * sizeof(meeting_unit));
        size = bench->config->num_objects / 10 *sizeof(meeting_unit);
        log("\t%.2f MB\t d_meetings_ps",1.0*size/1024/1024);

        h_bench.active_meeting_count_ps = (uint *)gpu->allocate(100 * sizeof(uint));
        size = 100 * sizeof(uint);
        log("\t%.2f MB\t active_meeting_count_ps",1.0*size/1024/1024);
    }

//	h_bench.meetings = (meeting_unit *)gpu->allocate(bench->meeting_capacity*sizeof(meeting_unit));
//	size = bench->meeting_capacity*sizeof(meeting_unit);
//	log("\t%.2f MB\tmeetings",1.0*size/1024/1024);

    //cuda sort
    cudaMallocHost((void **)&bench->h_keys[0], bench->config->kv_capacity*sizeof(__uint128_t));
    cudaMallocHost((void **)&bench->h_keys[1], bench->config->kv_capacity*sizeof(__uint128_t));

    h_bench.d_keys = (__uint128_t *)gpu->allocate(bench->config->kv_capacity*sizeof(__uint128_t));
    size = bench->config->kv_capacity*sizeof(__uint128_t);
    log("\t%.2f MB\td_keys",1.0*size/1024/1024);

    h_bench.d_ctf_keys = (uint8_t *)gpu->allocate(size);
    log("\t%.2f MB\td_ctf_keys",1.0*size/1024/1024);
//    h_bench.d_values = (uint64_t *)gpu->allocate(bench->config->kv_capacity*sizeof(uint64_t));
//    size = bench->config->kv_capacity*sizeof(uint64_t);
//    log("\t%.2f MB\td_values",1.0*size/1024/1024);

    h_bench.mid_xys = (uint64_t *)gpu->allocate(bench->config->num_objects*sizeof(uint64_t));
    size = bench->config->num_objects*sizeof(uint64_t);
    log("\t%.2f MB\tmid_xys",1.0*size/1024/1024);
    cudaMemset(h_bench.mid_xys, 0, bench->config->num_objects*sizeof(uint64_t));

    h_bench.same_pid_count = (uint *)gpu->allocate(bench->config->num_objects * sizeof(uint));
    size = bench->config->num_objects*sizeof(uint);
    log("\t%.2f MB\tsame_name_count", 1.0 * size / 1024 / 1024);
    cudaMemset(h_bench.same_pid_count, 0, bench->config->num_objects * sizeof(uint));

    //cuda search
    h_bench.search_single_list = (search_info_unit *)gpu->allocate(bench->config->search_single_capacity*sizeof(search_info_unit));
    size = bench->config->search_single_capacity*sizeof(search_info_unit);
    log("\t%.2f MB\tsearch_single_list",1.0*size/1024/1024);
    h_bench.search_multi_pid = (uint *)gpu->allocate(bench->config->search_single_capacity*sizeof(uint));
    size = bench->config->search_single_capacity*sizeof(uint);
    log("\t%.2f MB\tsearch_single_list",1.0*size/1024/1024);
    h_bench.search_multi_list = (search_info_unit *)gpu->allocate(bench->config->search_multi_capacity*sizeof(search_info_unit));
    size = bench->config->search_single_capacity*sizeof(search_info_unit);
    log("\t%.2f MB\tsearch_single_list",1.0*size/1024/1024);

//    h_bench.d_longer_edges = (float *)gpu->allocate(bench->config->num_objects*sizeof(float));
//    size = bench->config->num_objects*sizeof(float);
//    log("\t%.2f MB\td_longer_edges",1.0*size/1024/1024);

    if(bench->config->bloom_filter) {
        //bloom filter
        h_bench.d_pstFilter = (unsigned char *) gpu->allocate(bench->dwFilterSize);
        size = bench->dwFilterSize;
        log("\t%.2f MB\td_pstFilter", 1.0 * size / 1024 / 1024);
        cudaMemset(h_bench.d_pstFilter, 0, bench->dwFilterSize);
    }


    if(true) {
        h_bench.d_sids = (unsigned short*)gpu->allocate(bench->config->num_objects*sizeof(unsigned short));
        size = bench->config->num_objects*sizeof(unsigned short);
        log("\t%.2f MB\td_sids", 1.0 * size / 1024 / 1024);
        cudaMemset(h_bench.d_sids, 0, bench->config->num_objects*sizeof(unsigned short));

        h_bench.d_oids = (uint*)gpu->allocate(bench->config->num_objects*sizeof(uint));
        size = bench->config->num_objects*sizeof(uint);
        log("\t%.2f MB\td_oids", 1.0 * size / 1024 / 1024);
        cudaMemset(h_bench.d_oids, 0, bench->config->num_objects*sizeof(uint));

        h_bench.o_boxs = (uint_box*)gpu->allocate(bench->config->num_objects*sizeof(uint_box));
        size = bench->config->num_objects*sizeof(uint_box);
        log("\t%.2f MB\to_boxs", 1.0 * size / 1024 / 1024);

        h_bench.kv_boxs = (f_box *)gpu->allocate(bench->config->kv_capacity*sizeof(f_box));
        size = bench->config->kv_capacity*sizeof(f_box);
        log("\t%.2f MB\tkv_boxs",1.0*size/1024/1024);

        h_bench.d_CTF_capacity = (uint *)gpu->allocate(bench->config->CTF_count * sizeof(uint));
        size = bench->config->CTF_count * sizeof(uint);
        log("\t%.2f MB\td_CTF_capacity",1.0*size/1024/1024);

        h_bench.d_bitmaps = (unsigned char *) gpu->allocate(bench->bitmaps_size);
        size = bench->bitmaps_size;
        log("\t%.2f MB\td_bitmaps", 1.0 * size / 1024 / 1024);
        cudaMemset(h_bench.d_bitmaps, 0, size);

        h_bench.d_bitmap_mbrs = (box *)gpu->allocate(bench->config->CTF_count * sizeof(box));
        size = bench->config->CTF_count * sizeof(box);
        log("\t%.2f MB\td_bitmap_mbrs",1.0*size/1024/1024);

        h_bench.d_ctfs = (CTF *)gpu->allocate(bench->config->CTF_count*sizeof(CTF));
        size = bench->config->CTF_count*sizeof(CTF);
        log("\t%.2f MB\t d_ctfs",1.0*size/1024/1024);
    }

    // space for the configuration
    h_bench.config = (configuration *)gpu->allocate(sizeof(configuration));
    // space for the mapping of bench in GPU
    workbench *d_bench = (workbench *)gpu->allocate(sizeof(workbench));

    // the configuration and schema are fixed
    CUDA_SAFE_CALL(cudaMemcpy(h_bench.schema, bench->schema, bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(h_bench.config, bench->config, sizeof(configuration), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

    cuda_init_grids_stack<<<bench->grids_stack_capacity/1024 + 1, 1024>>>(d_bench);
    cuda_init_schema_stack<<<bench->schema_stack_capacity/1024 + 1, 1024>>>(d_bench);
    cuda_clean_buckets<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);

    cuda_init_o_boxs<<<bench->config->num_objects/1024 + 1, 1024>>>(d_bench);

    logt("GPU allocating space %ld MB", start,gpu->size_allocated()/1024/1024);

    return d_bench;
}

/*
 *
 * check the reachability of objects in a list of partitions
 * ctx.data contains the list of
 *
 * */
void process_with_gpu(workbench *bench, workbench* d_bench, gpu_info *gpu){
    struct timeval start = get_cur_time();
    //gpu->print();
    assert(bench);
    assert(d_bench);
    assert(gpu);
    cudaSetDevice(gpu->device_id);

    /* 1. copy data */
    // setup the current time and points for this round
    workbench h_bench(bench);
    CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
    if(bench->search_single) {
        h_bench.search_single = true;
        h_bench.single_find_count = 0;
    }
    else {
        h_bench.search_single = false;
    }
    if(bench->search_multi) {
        h_bench.search_multi = true;
        h_bench.multi_find_count = 0;
        h_bench.search_multi_length = bench->search_multi_length;
        CUDA_SAFE_CALL(cudaMemcpy(h_bench.search_multi_pid, bench->search_multi_pid, bench->search_multi_length * sizeof(search_info_unit), cudaMemcpyHostToDevice));
    }
    else {
        h_bench.search_multi = false;
    }
    h_bench.cur_time = bench->cur_time;
    h_bench.end_time_min = bench->end_time_min;
    CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

    cout << "bench->config->min_meet_time" << bench->config->min_meet_time << endl;

    if(bench->config->load_meetings_pers){
        CUDA_SAFE_CALL(cudaMemcpy(h_bench.active_meeting_count_ps, bench->active_meeting_count_ps, 100 * sizeof(uint), cudaMemcpyHostToDevice));
        uint prefix_sum = 0;
        uint second_index = bench->cur_time % 100;
        for(uint i = 1; i < bench->cur_time % 100 ; i++){
            prefix_sum += bench->active_meeting_count_ps[i - 1];
        }
        CUDA_SAFE_CALL(cudaMemcpy(h_bench.d_meetings_ps, bench->h_meetings_ps + prefix_sum, bench->active_meeting_count_ps[second_index] * sizeof(meeting_unit), cudaMemcpyHostToDevice));
        bench->pro.copy_time += get_time_elapsed(start,false);
        logt("copy in meetings_ps", start);

        insert_meetings_to_buckets<<<bench->active_meeting_count_ps[second_index]/1024+1,1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
    }
    else{
        CUDA_SAFE_CALL(cudaMemcpy(h_bench.points, bench->points, bench->config->num_objects*sizeof(Point), cudaMemcpyHostToDevice));
        bench->pro.copy_time += get_time_elapsed(start,false);
        logt("copy in points", start);

        if(!bench->config->dynamic_schema){
            struct timeval newstart = get_cur_time();
            cuda_clean_cells<<<one_dim*one_dim/1024+1,1024>>>(d_bench);
            cuda_build_qtree<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
//		check_execution();
//		cudaDeviceSynchronize();
//		logt("build qtree", newstart);

            for(uint i=1;i<=one_dim;i*=2){
                uint num = one_dim*one_dim/(i*i);
                cuda_merge_qtree<<<num/1024+1,1024>>>(d_bench,i);
//			check_execution();
//			cudaDeviceSynchronize();
//			CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
//			logt("merge qtree %d %d %d %d", newstart,i, h_bench.schema_stack_index, h_bench.grids_stack_index, h_bench.grid_check_counter);
            }
            CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            bench->pro.index_update_time += get_time_elapsed(start,false);
            logt("build qtree %d nodes %d partitions", start, h_bench.schema_stack_index, h_bench.grids_stack_index);
            //exit(0);
        }

        /* 2. filtering */
        if(bench->config->phased_lookup){
            // do the partition
            cuda_partition<<<bench->config->num_objects/1024+1,1024>>>(d_bench);

            CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
            bench->pro.partition_time += get_time_elapsed(start,false);
            logt("partition data %d still need lookup", start,h_bench.filter_list_index);
            bench->filter_list_index = h_bench.filter_list_index;
        }else{
            cuda_pack_lookup<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
            check_execution();
            cudaDeviceSynchronize();
            CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        }

        uint batch_size = bench->tmp_space_capacity/(PER_STACK_SIZE*2+1);
        for(int i=0;i<h_bench.filter_list_index;i+=batch_size){
            int bs = min(batch_size,h_bench.filter_list_index-i);
            cuda_filtering<<<bs/1024+1,1024>>>(d_bench, i, bs, !bench->config->phased_lookup);
            check_execution();
            cudaDeviceSynchronize();
        }
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.filter_time += get_time_elapsed(start,false);
        logt("filtering with %d checkings", start,h_bench.grid_check_counter);

        /* 3. refinement step */
        if(false){
            for(uint offset=0;offset<bench->grid_capacity;offset+=bench->config->zone_capacity){
                struct timeval ss = get_cur_time();
                bench->grid_check_counter = h_bench.grid_check_counter;
                uint thread_y = bench->config->zone_capacity;
                uint thread_x = 1024/thread_y;
                dim3 block(thread_x, thread_y);
                cuda_refinement_unroll<<<h_bench.grid_check_counter/thread_x+1,block>>>(d_bench,offset);
                check_execution();
                cudaDeviceSynchronize();
                logt("process %d",ss,offset);
            }
            CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
            bench->pro.refine_time += get_time_elapsed(start,false);
            logt("refinement step", start);
        }else{
            if(bench->config->unroll){
                cuda_unroll<<<h_bench.grid_check_counter/1024+1,1024>>>(d_bench,h_bench.grid_check_counter);
                check_execution();
                cudaDeviceSynchronize();
                CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
                bench->pro.refine_time += get_time_elapsed(start,false);
                logt("%d pid-grid-offset tuples need be checked", start,h_bench.grid_check_counter);
            }

            bench->grid_check_counter = h_bench.grid_check_counter;
            uint thread_y = bench->config->unroll?bench->config->zone_capacity:bench->grid_capacity;
            uint thread_x = 1024/thread_y;
            dim3 block(thread_x, thread_y);
            cuda_refinement<<<h_bench.grid_check_counter/thread_x+1,block>>>(d_bench);
            check_execution();
            cudaDeviceSynchronize();
            CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
            bench->pro.refine_time += get_time_elapsed(start,false);
            logt("refinement step", start);
        }
    }

    if(bench->config->save_meetings_pers){
        bench->num_active_meetings = h_bench.num_active_meetings;
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_meetings_ps + bench->h_meetings_count, h_bench.d_meetings_ps, bench->num_active_meetings * sizeof(meeting_unit), cudaMemcpyDeviceToHost));
        uint index = bench->cur_time % 100;
        bench->active_meeting_count_ps[index] = bench->num_active_meetings;
        bench->total_meetings_this100s += bench->num_active_meetings;
    }
    else{             //if save_meetings_pers, skip
        /* 4. identify the completed meetings */
        if(bench->config->profile){
            cuda_profile_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
            check_execution();
            cudaDeviceSynchronize();
            logt("profile meetings",start);
        }
        int before_kv = h_bench.kv_count;
        cuda_identify_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.meeting_identify_time += get_time_elapsed(start,false);
        int kv_increase = h_bench.kv_count - before_kv;
        printf("second %d finished meetings : %d\n",bench->cur_time , kv_increase);
        bench->num_taken_buckets = h_bench.num_taken_buckets;
        bench->num_active_meetings = h_bench.num_active_meetings;
        logt("meeting identify: %d taken %d active %d new meetings found", start, h_bench.num_taken_buckets, h_bench.num_active_meetings, kv_increase);
        //cerr << "coefficient "<< h_bench.num_active_meetings * 1.0 / bench->config->num_meeting_buckets << endl;

        if(bench->cur_time == 2200){
            cout <<"1000~2000 " << h_bench.larger_than_1000s << " 2000~3000 " << h_bench.larger_than_2000s << " 3000~4000 " << h_bench.larger_than_3000s << " 4000~4096 " << h_bench.larger_than_4000s <<endl;
        }
    }

    //4.5 cuda sort
    if(h_bench.kv_count>bench->config->kv_restriction){
        struct timeval sort_start = get_cur_time();
        cout <<"1000~2000 " << h_bench.larger_than_1000s << " 2000~3000 " << h_bench.larger_than_2000s << " 3000~4000 " << h_bench.larger_than_3000s << " 4000~4096 " << h_bench.larger_than_4000s <<endl;
        bench->meeting_cut_count = h_bench.meeting_cut_count;
        bench->start_time_min = min(bench->start_time_min, h_bench.start_time_min);
        bench->start_time_max = max(bench->start_time_max, h_bench.start_time_max);

        uint offset = 0;
        if(bench->ctb_count % 2 == 1){
            offset = bench->config->MemTable_capacity/2;
        }
        offset += bench->MemTable_count;

        struct timeval STR_start = get_cur_time();
        set_oid<<<bench->config->num_objects / 1024 + 1, 1024>>>(d_bench);                  //thrust::sequence
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("set_oid: ",start);

        thrust::device_ptr<unsigned short> d_ptr_sids = thrust::device_pointer_cast(h_bench.d_sids);
        thrust::device_ptr<__uint128_t> d_ptr_keys = thrust::device_pointer_cast(h_bench.d_keys);
        __uint128_t * box128 = reinterpret_cast<__uint128_t *>(h_bench.kv_boxs);
        thrust::device_ptr<__uint128_t> d_ptr_boxes = thrust::device_pointer_cast(box128);
        //thrust::device_vector<__uint128_t> d_vector_keys(d_ptr_keys, d_ptr_keys + h_bench.kv_count);

        // predicate fuction
        auto predicate = [d_ptr_sids] __device__ (__uint128_t key) {
            uint oid = cuda_get_key_oid(key);
            return d_ptr_sids[oid] != 1;
        };

        // zip_iterator
        auto first = thrust::make_zip_iterator(thrust::make_tuple(d_ptr_keys, d_ptr_boxes));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(d_ptr_keys + h_bench.kv_count, d_ptr_boxes + h_bench.kv_count));

        // remove_if is wrong, use partition
        auto new_end = thrust::partition(first, last, [predicate] __device__ (thrust::tuple<__uint128_t, __uint128_t> t) {
            return predicate(thrust::get<0>(t));
        });
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("remove: ",start);

        // new length
        uint old_kv_count = h_bench.kv_count;
        h_bench.kv_count = thrust::distance(first, new_end);
        uint oversize_key_count = old_kv_count - h_bench.kv_count;
        assert(oversize_key_count < bench->config->oversize_buffer_capacity);
        bench->h_oversize_buffers[offset].oversize_kv_count = oversize_key_count;
        cout << "key cut count" << oversize_key_count << " oversize_oid_count:" << h_bench.oversize_oid_count << endl;
        cout << "slim h_bench.kv_count :" << h_bench.kv_count << endl;
        CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));   //update

        thrust::sort_by_key(d_ptr_keys + h_bench.kv_count, d_ptr_keys + old_kv_count, d_ptr_boxes + h_bench.kv_count);      //buffer sort
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));

        get_o_limit<<<1, 1024>>>(d_bench, oversize_key_count);
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_ctfs[offset], h_bench.d_ctfs, sizeof(CTF), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start, false);
        logt("o_limit: ",start);
        bench->h_oversize_buffers[offset].start_time_min = bench->h_ctfs[offset][0].start_time_min;
        bench->h_oversize_buffers[offset].start_time_max = bench->h_ctfs[offset][0].start_time_max;
        bench->h_oversize_buffers[offset].end_time_min = bench->h_ctfs[offset][0].end_time_min;
        bench->h_oversize_buffers[offset].end_time_max = bench->h_ctfs[offset][0].end_time_max;

        cuda_write_o_bitmap<<<oversize_key_count / 1024 + 1, 1024>>>(d_bench, oversize_key_count);
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_oversize_buffers[offset].o_bitmaps, h_bench.d_bitmaps, bench->bit_count / 8, cudaMemcpyDeviceToHost));
        cudaMemset(h_bench.d_bitmaps, 0, bench->bit_count / 8);
        bench->pro.cuda_sort_time += get_time_elapsed(start, false);
        logt("cuda_write_o_bitmap_time: ",start);

//        cerr << "output picked o bitmap" << endl;
//        Point * bit_points = new Point[bench->bit_count];
//        uint count_p;
//        cerr<<endl;
//        count_p = 0;
//        for(uint i=0;i<bench->bit_count;i++){
//            if(bench->h_oversize_buffers[offset].o_bitmaps[0 + i/8] & (1<<(i%8))){
//                Point bit_p;
//                uint x=0,y=0;
//                x = i % DEFAULT_bitmap_edge;
//                y = i / DEFAULT_bitmap_edge;
//                bit_p.x = (double)x/DEFAULT_bitmap_edge*(bench->mbr.high[0] - bench->mbr.low[0]) + bench->mbr.low[0];
//                bit_p.y = (double)y/DEFAULT_bitmap_edge*(bench->mbr.high[1] - bench->mbr.low[1]) + bench->mbr.low[1];
//                bit_points[count_p] = bit_p;
//                count_p++;
//            }
//        }
//        cout<<"bit_points.size():"<<count_p<<endl;
//        print_points(bit_points,count_p);
//        //cerr << "process output bitmap finish" << endl;
//        delete[] bit_points;

        thrust::device_ptr<uint64_t> d_vector_xys = thrust::device_pointer_cast(h_bench.mid_xys);
        thrust::device_ptr<uint> d_vector_oids = thrust::device_pointer_cast(h_bench.d_oids);
        thrust::sort_by_key(d_vector_xys, d_vector_xys + bench->config->num_objects, d_vector_oids);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("cuda_sort_by_x_time: ",start);

        narrow_xy_to_y<<<bench->config->num_objects / 1024 + 1, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("narrow_xy_to_y: ",start);

        uint zero_one_sid_count = (bench->config->num_objects - h_bench.sid_count) + h_bench.oversize_oid_count;
        cout <<"zero_one_sid_count"<< zero_one_sid_count << endl;
        uint x_part_capacity = (h_bench.sid_count - h_bench.oversize_oid_count) / bench->config->split_num + 1;
        for(uint i = 0; i < bench->config->split_num - 1; i++){
            thrust::sort_by_key(d_vector_xys + zero_one_sid_count + x_part_capacity * i,
                                d_vector_xys + zero_one_sid_count + x_part_capacity * (i + 1),
                                d_vector_oids + zero_one_sid_count + x_part_capacity * i);
            cudaDeviceSynchronize();
            check_execution();
        }
        thrust::sort_by_key(d_vector_xys + zero_one_sid_count + x_part_capacity * (bench->config->split_num - 1),                    //last part
                            d_vector_xys + bench->config->num_objects,
                            d_vector_oids + zero_one_sid_count + x_part_capacity * (bench->config->split_num - 1));
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("cuda_sort_by_y_time ",start);

        make_sid<<<bench->config->num_objects / 1024 + 1, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("make_sid: ",start);

        write_sid_in_key<<<h_bench.kv_count / 1024 + 1, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("write_sid_in_key: ",start);

        thrust::sort_by_key(d_ptr_keys, d_ptr_keys + h_bench.kv_count, d_ptr_boxes);
        logt("all ctf sort: ",start);

        double STR_time = get_time_elapsed(STR_start);
        cerr << "STR_time " << STR_time << endl;
        struct timeval encoding_start = get_cur_time();

        get_CTF_capacity<<<bench->config->num_objects / 1024 + 1, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_CTF_capacity[offset], h_bench.d_CTF_capacity, bench->config->CTF_count * sizeof(uint), cudaMemcpyDeviceToHost));
        logt("get_CTF_capacity: ",start);

//        cout << "print ctf capacity" << endl;
//        for(uint i = 0; i < bench->config->CTF_count; i++){
//            cout << bench->h_CTF_capacity[offset][i] << endl;
//        }


//        f_box * h_kv_box = new f_box[h_bench.kv_count];
//        CUDA_SAFE_CALL(cudaMemcpy(h_kv_box, h_bench.kv_boxs, h_bench.kv_count * sizeof(f_box), cudaMemcpyDeviceToHost));
//        for(uint i = 0; i < 300; i++){
//            h_kv_box[i].print();
//        }
//        cerr << "random box" << endl;
//        for(uint i = bench->h_CTF_capacity[offset][0] - 100; i < bench->h_CTF_capacity[offset][0] + 100; i++){
//            h_kv_box[i].print();
//        }

//        cout << "" << endl;
//        CUDA_SAFE_CALL(cudaMemcpy(bench->h_keys[offset], h_bench.d_keys, h_bench.kv_count * sizeof(__uint128_t), cudaMemcpyDeviceToHost));
//        for(uint i = bench->h_CTF_capacity[offset][0]; i < bench->h_CTF_capacity[offset][0] + 100; i++){
//            uint pid = get_key_oid(bench->h_keys[offset][i]);
//            uint target = get_key_target(bench->h_keys[offset][i]);
//            uint duration = get_key_duration(bench->h_keys[offset][i]);
//            uint end = get_key_end(bench->h_keys[offset][i]);
//            cout << i << " " << pid << "-" << target << endl;
//        }

        //reduce
        cuda_get_limit<<<bench->config->CTF_count, 1024>>>(d_bench);
        cudaDeviceSynchronize();
        check_execution();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        logt("cuda_get_limit: ",start);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_ctfs[offset], h_bench.d_ctfs, bench->config->CTF_count * sizeof(CTF), cudaMemcpyDeviceToHost));

//        for(uint j = 0;j<bench->config->CTF_count; j++){
//            bench->h_ctfs[offset][j].ctf_mbr.print();
//        }

#pragma omp parallel for num_threads(bench->config->CTF_count)
        for(uint i = 0; i < bench->config->CTF_count; i++){
            bench->h_ctfs[offset][i].get_ctf_bits(bench->mbr, bench->config);
            //bench->h_ctfs[offset][i].print_ctf_meta();
        }
        CUDA_SAFE_CALL(cudaMemcpy(h_bench.d_ctfs, bench->h_ctfs[offset], bench->config->CTF_count * sizeof(CTF), cudaMemcpyHostToDevice));
        bench->h_ctfs[offset][65].print_ctf_meta();

        //bitmap
        cout<<"h_bench.bit_count:"<<h_bench.bit_count<<endl;
        cout<<"h_bench.bitmaps_size:"<<h_bench.bitmaps_size<<endl;
        cuda_write_bitmap<<<bench->config->CTF_count, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        logt("cuda_write_bitmap: ",start);

//            //    //output bitmap
//        CUDA_SAFE_CALL(cudaMemcpy(bench->h_bitmaps[offset], h_bench.d_bitmaps, bench->bitmaps_size, cudaMemcpyDeviceToHost));
//        cerr << "output picked bitmap" << endl;
//        Point * bit_points = new Point[bench->bit_count];
//        uint count_p;
//        for(uint j = 0;j<bench->config->CTF_count; j++){
//            //cerr<<"bitmap"<<j<<endl;
//            cerr<<endl;
//            CTF * temp_ctf = &bench->h_ctfs[offset][j];
//            count_p = 0;
//            bool is_print = false;
//            for(uint i=0;i<bench->bit_count;i++){
//                if(bench->h_bitmaps[offset][j*(bench->bit_count/8) + i/8] & (1<<(i%8))){
//                    if(!is_print){
//                        cout<<i<<"in SST"<<j<<endl;
//                        is_print = true;
//                    }
//                    Point bit_p;
//                    uint x=0,y=0;
//                    x = i % temp_ctf->x_grid;
//                    y = i / temp_ctf->x_grid;
//                    bit_p.x = (double)x/temp_ctf->x_grid*(temp_ctf->ctf_mbr.high[0] - temp_ctf->ctf_mbr.low[0]) + temp_ctf->ctf_mbr.low[0];
//                    bit_p.y = (double)y/temp_ctf->y_grid*(temp_ctf->ctf_mbr.high[1] - temp_ctf->ctf_mbr.low[1]) + temp_ctf->ctf_mbr.low[1];
//                    bit_points[count_p] = bit_p;
//                    count_p++;
//                }
//            }
//            cout<<"bit_points.size():"<<count_p<<endl;
//            print_points(bit_points,count_p);
//            //cerr << "process output bitmap finish" << endl;
//        }
//        delete[] bit_points;

        write_key_and_mbr<<<bench->config->CTF_count, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        logt("write_value_mbr: ",start);

        double encoding_time = get_time_elapsed(encoding_start);
        cerr << "encoding_time " << encoding_time << endl;

        //CUDA_SAFE_CALL(cudaMemcpy(bench->h_keys[offset], h_bench.d_keys, h_bench.kv_count * sizeof(__uint128_t), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_keys[offset], h_bench.d_ctf_keys, h_bench.kv_count * sizeof(__uint128_t), cudaMemcpyDeviceToHost));
//        double keys_to_cpu = get_time_elapsed(start, true);
//        cerr << "keys_to_cpu " << keys_to_cpu << endl;
        logt("copy to cpu, h_keys %d MB", start, h_bench.kv_count * sizeof(__uint128_t) / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_oversize_buffers[offset].keys, h_bench.d_keys + h_bench.kv_count, oversize_key_count * sizeof(__uint128_t), cudaMemcpyDeviceToHost));
        logt("h_oversize_buffers keys %d MB", start, oversize_key_count * sizeof(__uint128_t) / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_oversize_buffers[offset].boxes, h_bench.kv_boxs + h_bench.kv_count, oversize_key_count * sizeof(f_box), cudaMemcpyDeviceToHost));
        logt("h_oversize_buffers boxes %d MB", start, oversize_key_count * sizeof(f_box) / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_sids[offset], h_bench.d_sids, bench->config->num_objects * sizeof(unsigned short), cudaMemcpyDeviceToHost));
        logt("h_sids %d MB", start, bench->config->num_objects * sizeof(unsigned short) / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_CTF_capacity[offset], h_bench.d_CTF_capacity, bench->config->CTF_count * sizeof(uint), cudaMemcpyDeviceToHost));
        logt("h_CTF_capacity %d MB", start, bench->config->CTF_count * sizeof(uint) / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_bitmaps[offset], h_bench.d_bitmaps, bench->bitmaps_size, cudaMemcpyDeviceToHost));
        logt("h_bitmaps %d MB", start, bench->bitmaps_size / 1024 /1024);
        CUDA_SAFE_CALL(cudaMemcpy(bench->h_bitmap_mbrs[offset], h_bench.d_bitmap_mbrs, bench->config->CTF_count * sizeof(box), cudaMemcpyDeviceToHost));
        logt("h_bitmap_mbrs %d MB", start, bench->config->CTF_count * sizeof(box) / 1024 /1024);

//        cout << "host test oversize" << endl;
//        for(uint i = 0; i < bench->h_oversize_buffers[offset].oversize_kv_count ; i++){
//            print_parse_key(bench->h_oversize_buffers[offset].keys[i]);
//            cout << bench->h_oversize_buffers[offset].boxes[i].low[0] << endl;
//            cout << "sid" << bench->h_sids[offset][get_key_oid(bench->h_oversize_buffers[offset].keys[i]) ] << endl;
//        }


        //        f_box * temp_fbs = new f_box[bench->config->kv_restriction];
//        CUDA_SAFE_CALL(cudaMemcpy(temp_fbs, h_bench.kv_boxs, bench->config->kv_restriction * sizeof(f_box), cudaMemcpyDeviceToHost));

//        cout << "sid_count" << h_bench.sid_count << endl;
//        short * temp_same_pid_count = new short[bench->config->num_objects];
//        CUDA_SAFE_CALL(cudaMemcpy(temp_same_pid_count, h_bench.same_pid_count, bench->config->num_objects * sizeof(unsigned short), cudaMemcpyDeviceToHost));
//        for(uint i = 0; i < bench->config->num_objects; i++){
//            if(temp_same_pid_count[i]>0){
//                cout << temp_same_pid_count[i] << " ";
//            }
//        }
//        cout << endl;


        logt("cudaMemcpy kv and meta",start);


//        cout << "cu CTF_capacitys:" << endl;
//        for(uint i = 0; i < bench->config->SSTable_count; i++){
//            cout << bench->h_CTF_capacity[offset][i] << endl;
//        }

//        for(uint i = 0; i < 100; i++){
//            bench->h_ctfs[offset][0].print_key(bench->h_keys[offset][i]);
//        }


//        box b;
//        parse_mbr(bench->h_keys[offset][10], b, bench->h_bitmap_mbrs[offset][0]);
//        b.print();


//        if(bench->cur_time > 0){
//            bench->h_bitmap_mbrs[offset][25].print();
//            cerr << "output the real boxes of a sst" << endl;
//            f_box * temp_f_box = new f_box[bench->config->SSTable_kv_capacity];
//            CUDA_SAFE_CALL(cudaMemcpy(temp_f_box, h_bench.kv_boxs + 25*bench->config->SSTable_kv_capacity, bench->config->SSTable_kv_capacity * sizeof(f_box), cudaMemcpyDeviceToHost));
//            for(uint i = 0; i < bench->config->SSTable_kv_capacity/2; i++){
//                temp_f_box[i].print();
//            }
//            delete[] temp_f_box;
//        }

//        cerr << "kv box, real box, and then the bitmap_mbr"<<endl;
//        bench->h_bitmap_mbrs[offset][0].print();
//        cerr<<"bitmap_mbrs:"<<endl;
//        for(int i = 0; i < bench->config->CTF_count; i++){
//            bench->h_bitmap_mbrs[offset][i].print();
//        }

//        //longer edges sort
//        thrust::device_ptr<float> d_vec_edges = thrust::device_pointer_cast(h_bench.d_longer_edges);
//        thrust::sort(d_vec_edges, d_vec_edges + bench->config->num_objects);
//        check_execution();
//        cudaDeviceSynchronize();
//        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
//        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
//        logt("cuda_sort_time: ",start);
//        CUDA_SAFE_CALL(cudaMemcpy(bench->h_longer_edges, h_bench.d_longer_edges, bench->config->num_objects*sizeof(float), cudaMemcpyDeviceToHost));
//        cudaMemset(h_bench.d_longer_edges, 0, bench->config->num_objects*sizeof(float));
//        cout << "longest edge " <<bench->h_longer_edges[bench->config->num_objects-1] << endl;
////        cout << "long_meeting_count: " << h_bench.long_meeting_count << endl;
////        cout << "long_oid_count: " << h_bench.long_oid_count << endl;
////        h_bench.long_meeting_count = 0;
////        h_bench.long_oid_count = 0;

        //init
        cuda_init_o_boxs<<<bench->config->num_objects/1024 + 1, 1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        h_bench.start_time_min = (1ULL<<32) -1;
        h_bench.start_time_max = 0;
        h_bench.sid_count = 0;
        h_bench.oversize_oid_count = 0;
        cudaMemset(h_bench.d_sids, 0, bench->config->num_objects*sizeof(unsigned short));
        cudaMemset(h_bench.d_oids, 0, bench->config->num_objects*sizeof(uint));
        cudaMemset(h_bench.d_bitmaps, 0, bench->bitmaps_size);
        cudaMemset(h_bench.d_bitmap_mbrs, 0, bench->config->CTF_count * sizeof(box));
        cudaMemset(h_bench.same_pid_count, 0, bench->config->num_objects * sizeof(uint));
        cudaMemset(h_bench.mid_xys, 0, bench->config->num_objects*sizeof(uint64_t));
        cudaMemset(h_bench.d_CTF_capacity, 0, bench->config->CTF_count * sizeof(uint));
        cudaMemset(h_bench.d_ctfs, 0, bench->config->CTF_count * sizeof(CTF));
        cudaMemset(h_bench.d_ctf_keys, 0, bench->config->kv_capacity * sizeof(__uint128_t));
        bench->MemTable_count++;
        cout << "bench->MemTable_count " << bench->MemTable_count << "MemTable_capacity" << bench->config->MemTable_capacity <<endl;
        h_bench.kv_count = 0;
        CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));                       //update kv_count, other effect ???
        bench->pro.cuda_sort_time += get_time_elapsed(start,false);
        logt("init after sort",start);

        double sort_total = get_time_elapsed(sort_start,false);          //sort_total = organization + copy_out
        cerr << "sort_total " << sort_total << endl;
    }


    // todo do the data analyzes, for test only, should not copy out so much stuff
    if(bench->config->analyze_grid||bench->config->analyze_reach||bench->config->profile){
        CUDA_SAFE_CALL(cudaMemcpy(bench->grid_counter, h_bench.grid_counter,
                                  bench->grids_stack_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
        logt("copy out grid counting data", start);
    }
    if(bench->config->analyze_reach){
        CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench.grids,
                                  bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bench->schema, h_bench.schema,
                                  bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets, h_bench.meeting_buckets,
                                  bench->config->num_meeting_buckets*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
        bench->schema_stack_index = h_bench.schema_stack_index;
        bench->grids_stack_index = h_bench.grids_stack_index;
        logt("copy out grid, schema, meeting buckets data", start);
    }

    /* 5. update the index */
    if(bench->config->dynamic_schema){
        // update the schema for future processing
        cuda_update_schema_collect<<<bench->schema_stack_capacity/1024+1,1024>>>(d_bench);
        check_execution();
        cudaDeviceSynchronize();
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        if(h_bench.split_list_index>0){
            //cerr << "split_list_index" << h_bench.split_list_index << endl;
            //cerr << "schema_stack_index " << h_bench.schema_stack_index << endl;
            cuda_update_schema_split<<<h_bench.split_list_index/1024+1,1024>>>(d_bench, h_bench.split_list_index);
            check_execution();
            cudaDeviceSynchronize();
        }
        if(h_bench.merge_list_index>0){
            //cerr << "merge_list_index" << h_bench.merge_list_index << endl;
            cuda_update_schema_merge<<<h_bench.merge_list_index/1024+1,1024>>>(d_bench, h_bench.merge_list_index);
            check_execution();
            cudaDeviceSynchronize();
        }
        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
        bench->pro.index_update_time += get_time_elapsed(start,false);
        logt("schema update %d grids", start, h_bench.grids_stack_index);
    }

    /* 6. search kv info */
    if(bench->search_single){
//        cuda_search_single_kv<<<h_bench.kv_count/1024+1,1024>>>(d_bench);
//        check_execution();
//        cudaDeviceSynchronize();
//        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
//        bench->single_find_count = h_bench.single_find_count;
//        CUDA_SAFE_CALL(cudaMemcpy(bench->search_single_list, h_bench.search_single_list, bench->single_find_count*sizeof(search_info_unit), cudaMemcpyDeviceToHost));
//        bench->pro.cuda_search_single_kv_time += get_time_elapsed(start,false);
//        logt("search_single_kv ", start);
    }
    if(bench->search_multi){
//        cout<<"before multi_find_count"<<h_bench.multi_find_count<<endl;
//        cuda_search_multi_kv<<<h_bench.kv_count/1024+1,1024>>>(d_bench);
//        check_execution();
//        cudaDeviceSynchronize();
//        CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
//        bench->multi_find_count = h_bench.multi_find_count;
//        cout<<"after multi_find_count"<<h_bench.multi_find_count<<endl;
//        CUDA_SAFE_CALL(cudaMemcpy(bench->search_multi_list, h_bench.search_multi_list, bench->multi_find_count*sizeof(search_info_unit), cudaMemcpyDeviceToHost));
//        bench->pro.cuda_search_multi_kv_time += get_time_elapsed(start,false);
//        logt("search_multi_kv ", start);
    }

    /* 6. post-process, copy out data*/
//	if(h_bench.meeting_counter>0){
//		bench->meeting_counter = h_bench.meeting_counter;
//		CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench.meetings, min(bench->meeting_capacity, h_bench.meeting_counter)*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
//		bench->pro.copy_time += get_time_elapsed(start,false);
//		logt("copy out %d meeting data", start,h_bench.meeting_counter);
//	}
    // clean the device bench for next round of checking
    cuda_cleargrids<<<bench->grids_stack_capacity/1024+1,1024>>>(d_bench);
    cuda_reset_bench<<<1,1>>>(d_bench);
}
