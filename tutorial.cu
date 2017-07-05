#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"

template <typename dtype> __global__ void AddKernel(const dtype* a, const dtype* b, dtype* c, int N){
	CUDA_1D_KERNEL_LOOP(index, N)
	{
		c[index] = a[index] + b[index];
	}
}

template <typename dtype>
void launchAddKernel(const dtype* a, const dtype* b, dtype* c, int N) {
	const int kThreadsPerBlock = 1024;
	
	AddKernel<dtype><<<(N + kThreadsPerBlock - 1) / kThreadsPerBlock,
					kThreadsPerBlock>>>(
			a, b, c, N);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}

//forward declaration for all the types needed
template void launchAddKernel<double>(
	const double* a, const double* b, double* c, int N);
