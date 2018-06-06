#pragma once
#include "tensorflow/core/framework/op_kernel.h"

template <typename Device, typename dtype>
struct launchAddKernel {
  void operator()(const Device& d, const dtype* a, const dtype* b, dtype* c, int N);
};