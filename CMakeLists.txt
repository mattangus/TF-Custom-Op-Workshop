cmake_minimum_required(VERSION 3.5)

#get the include directory for tensorflow
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

include_directories(${Tensorflow_INCLUDE_DIRS})

find_package(CUDA)

#set flags based on tutorial
set (CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")

set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

#create library
cuda_add_library(
	tutorial SHARED
	tutorial.cu
	tutorial.cc)