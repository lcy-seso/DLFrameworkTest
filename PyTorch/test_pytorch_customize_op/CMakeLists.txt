cmake_minimum_required(VERSION 3.8)
project(PT_customized_op CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

include(python)
find_package(Torch REQUIRED)

add_library(myadd SHARED my_add_kernel.cu)
target_link_libraries(myadd "${TORCH_LIBRARIES}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
