cmake_minimum_required(VERSION 3.5)

project(softmax)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60;)

cuda_add_executable(softmax softmax.cu)
