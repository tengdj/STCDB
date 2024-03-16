/*
 * test.cu
 *
 *  Created on: Feb 16, 2021
 *      Author: teng
 */

#include "cuda_util.cuh"
#include "../geometry/geometry.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/*
 * for test
 * */

__global__ void mykernel(Point *p1, Point *p2, double *dist) {
	*dist = distance(p1->x,p1->y,p2->x,p2->y);
	printf("gpu %f\n",*dist);
  //*addr += 10;
  //atomicAdd(addr, 10);       // only available on devices with compute capability 6.x
}

double foo(Point *p1, Point *p2) {
	Point *d_p1,*d_p2;
	double *d_dist;
	cudaMallocManaged(&d_p1, sizeof(Point));
	cudaMallocManaged(&d_p2, sizeof(Point));
	cudaMallocManaged(&d_dist, sizeof(double));

	CUDA_SAFE_CALL(cudaMemcpy(d_p1, p1, sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_p2, p2, sizeof(Point), cudaMemcpyHostToDevice));

	mykernel<<<1,1>>>(d_p1, d_p2, d_dist);
	double dist = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&dist, d_dist, sizeof(double), cudaMemcpyDeviceToHost));
	return dist;
}



void bar()
{
	struct timeval start = get_cur_time();
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  logt("generate random numbers",start);
  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;
  logt("transfer to device",start);

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());
  logt("sort",start);

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  logt("copy out",start);

}


