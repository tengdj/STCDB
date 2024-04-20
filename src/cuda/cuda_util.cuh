/*
 * cuda_util.h
 *
 *  Created on: Jun 1, 2020
 *      Author: teng
 */

#ifndef CUDA_UTIL_CUH_
#define CUDA_UTIL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include "../util/util.h"
#include "hilbert_curve.cuh"



#define CUDA_SAFE_CALL(call) 										  	  \
	do {																  \
		cudaError_t err = call;											  \
		if (cudaSuccess != err) {										  \
			fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
					__FILE__, __LINE__, cudaGetErrorString(err) );	      \
			exit(EXIT_FAILURE);											  \
		}																  \
	} while (0);


inline void check_execution(){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		log(cudaGetErrorString(err));
	}
}

// return the distance of two segments

const static double degree_per_meter_latitude_cuda = 360.0/(40076.0*1000);

__device__
inline double degree_per_meter_longitude_cuda(double latitude){
	return 360.0/(sin((90-abs(latitude))*PI/180)*40076.0*1000.0);
}

__device__
inline double distance(const double x1, const double y1, const double x2, const double y2){
	double dx = x1-x2;
	double dy = y1-y2;
	dx = dx/degree_per_meter_longitude_cuda(y1);
	dy = dy/degree_per_meter_latitude_cuda;
	return sqrt(dx*dx+dy*dy);
}

__device__
inline uint getpid1(size_t z){
    size_t w = floor((sqrt(8.0 * z + 1) - 1)/2);
    size_t t = (w*w + w) / 2;
    uint y = (uint)(z - t);
    uint x = (uint)(w - y);
    return x;
}

__device__
inline uint getpid2(size_t z){
    size_t w = floor((sqrt(8.0 * z + 1) - 1)/2);
    size_t t = (w*w + w) / 2;
    uint y = (uint)(z - t);
    return y;
}

__device__
inline uint64_t d_MurmurHash2_x64( const void * key, int len, uint32_t seed ){
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len/8);

    while(data != end)
    {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const uint8_t * data2 = (const uint8_t*)data;

    switch(len & 7)
    {
        case 7: h ^= ((uint64_t)data2[6]) << 48;
        case 6: h ^= ((uint64_t)data2[5]) << 40;
        case 5: h ^= ((uint64_t)data2[4]) << 32;
        case 4: h ^= ((uint64_t)data2[3]) << 24;
        case 3: h ^= ((uint64_t)data2[2]) << 16;
        case 2: h ^= ((uint64_t)data2[1]) << 8;
        case 1: h ^= ((uint64_t)data2[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

__device__
inline uint float_to_uint(float xy) {
//    uint ret = 0;
//    if(xy<0){
//        ret = 10000000;
//        xy = 0-xy;
//    }
////        uint inte = (uint)xy;
////        uint decimals = (xy - inte)*10000;
//    ret += (uint)(xy*10000);
//    return ret;
    xy += 180;
    return (uint)(xy*10000);
}

__host__ __device__
float uint_to_float(uint f);

__host__ __device__
uint get_key_wid(__uint128_t key);

__host__ __device__
uint get_key_pid(__uint128_t key);

//__host__ __device__
//uint get_value_mbr_low(__uint128_t key);
//
//__host__ __device__
//uint get_value_mbr_high(__uint128_t key);

__host__ __device__
uint get_key_target(__uint128_t key);

__host__ __device__
uint get_key_duration(__uint128_t key);

__host__ __device__
uint get_key_end(__uint128_t key);

__host__
inline void print_parse_key(__uint128_t key){
    print_128(key);
    cout<<endl;
    uint wid = get_key_wid(key);
    cout<<"wid:"<<wid<<endl;
    uint first_low0, first_low1;
    d2xy(WID_BIT, wid, first_low0, first_low1);
    cout<<"pid:"<<get_key_pid(key)<<endl;
    //mbr
    cout<<"target:"<<get_key_target(key)<<endl;
    cout<<"duration:"<<get_key_duration(key)<<endl;
    cout<<"end offset:"<<get_key_end(key)<<endl;
}

#endif /* CUDA_UTIL_CUH_ */
