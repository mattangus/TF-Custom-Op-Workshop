#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"

template <typename dtype> __global__ void AddKernel(const dtype* a, const dtype* b, dtype* c, int N){

}

template <typename dtype>
void launchAddKernel(const dtype* a, const dtype* b, dtype* c, int N) {
	const int kThreadsPerBlock = 1024;

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}
