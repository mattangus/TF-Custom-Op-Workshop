cmake_minimum_required(VERSION 3.5)

#get the include directory for tensorflow
#execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

message("tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link dir: ${Tensorflow_LIB_DIRS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories("/usr/local/")
link_directories(${Tensorflow_LIB_DIRS})

find_package(CUDA)

#set flags based on tutorial
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr" )

set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

#create library
cuda_add_library(
	tutorial SHARED
	tutorial.cu
	tutorial.cc)

target_link_libraries(tutorial "tensorflow_framework")

#copy python files to build folder (for easy testing)
file(GLOB PY_FILES
  "src/*.py"
)
file(COPY ${PY_FILES} DESTINATION .)