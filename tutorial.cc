#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include <iostream>
#include <cuda.h>

#include "tutorial.h"

using namespace tensorflow;
using namespace std;
using namespace shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status ShapeFn(InferenceContext* c)
{
	//check input shape has 4 dimensions 
	ShapeHandle a_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &a_shape));

	//check indices has 4 dimensions (bach, width, height, channels)
	ShapeHandle b_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &b_shape));

	//get dims for both inputs
	for(int i = 0; i < 4; i++)
	{
		DimensionHandle a_dim = c->Dim(a_shape,i);
		DimensionHandle b_dim = c->Dim(b_shape,i);
		if (c->Value(a_dim) != c->Value(b_dim))
			return errors::InvalidArgument(
			"a and b dimensions must match input dimensions");
	}

	//set output size
	c->set_output(0, c->input(0));

	return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("CustomAdd")
		.Input("a: T")
		.Input("b: T")
		.Output("c: T")
		.Attr("T: {int32, float32, float64}")
		.SetShapeFn(ShapeFn);


template <typename dtype>
struct launchAddKernel<CPUDevice, dtype> {
	void operator()(const CPUDevice& d, const dtype* a, const dtype* b, dtype* c, int N) {
		for(int i = 0; i < N; i++)
			c[i] = a[i] + b[i];
	}
};


template <typename Device, typename dtype>
class CustomAddOp : public OpKernel {
public:
	
	explicit CustomAddOp(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& a_tensor = context->input(0);
		const Tensor& b_tensor = context->input(1);
		
		//flatten tensors
		auto a_flat = a_tensor.flat<dtype>();
		auto b_flat = b_tensor.flat<dtype>();

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0,
			a_tensor.shape(),&output_tensor));

		//get flat version to fill
		auto output = output_tensor->flat<dtype>();

		const int N = output.size();

		// Call the cuda kernel launcher
		launchAddKernel<Device, dtype>()(
			context->eigen_device<Device>(),
			a_flat.data(),
			b_flat.data(),
			output.data(), N);
	}
};

//register kernel with types needed
#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CustomAdd") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		CustomAddOp<GPUDevice, type>) \

REGISTER_GPU(int);
REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#define REGISTER_CPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CustomAdd") \
		.Device(DEVICE_CPU) \
		.TypeConstraint<type>("T"), \
		CustomAddOp<CPUDevice, type>) \

REGISTER_CPU(int);
REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU