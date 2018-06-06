#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_device_functions.h"

#include "tutorial.h"

using GPUDevice = Eigen::GpuDevice;

template <typename dtype> __global__ void AddKernel(const dtype* a, const dtype* b, dtype* c, int N){
	for(int index : tensorflow::CudaGridRangeX(N))
	{
		c[index] = a[index] + b[index];
	}
}

template <typename dtype>
struct launchAddKernel<GPUDevice, dtype> {
	void operator()(const GPUDevice& d, const dtype* a, const dtype* b, dtype* c, int N) {
		const int kThreadsPerBlock = 1024;
		
		AddKernel<dtype><<<(N + kThreadsPerBlock - 1) / kThreadsPerBlock,
						kThreadsPerBlock, 0, d.stream()>>>(
				a, b, c, N);

		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	}
};

//forward declaration for all the types needed
typedef Eigen::GpuDevice GPUDevice;
#define ADD_KERNEL_TYPE(type)							\
	template struct launchAddKernel<GPUDevice, type>;	\

ADD_KERNEL_TYPE(int);
ADD_KERNEL_TYPE(float);
ADD_KERNEL_TYPE(double);

#undef ADD_KERNEL_TYPE