cmake_minimum_required(VERSION 3.5)

project(reduction_ops)

find_package(CUDA QUIET REQUIRED)

set(
  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60;)

cuda_add_executable(reduction_ops reduction_ops.cu)
