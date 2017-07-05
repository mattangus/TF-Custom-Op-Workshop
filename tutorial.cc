#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include <iostream>
#include <cuda.h>

using namespace tensorflow;
using namespace std;
using namespace shape_inference;

Status ShapeFn(InferenceContext* c)
{

	return Status::OK();
}


//declare kernel launcher
template <typename dtype>
void launchAddKernel(const dtype* a, const dtype* b, dtype* c, int N);

template <typename dtype>
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


	}
};
