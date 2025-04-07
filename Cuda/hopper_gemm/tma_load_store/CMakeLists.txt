cmake_minimum_required(VERSION 3.25.0)
project(hopper_gemm LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../cmake")
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../third_party/")

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard whose features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    17
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

include(select_compute_arch)
cuda_select_nvcc_arch_flags(ARCH_LIST Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_LIST})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_90a,code=sm_90a")

include_directories("${PROJECT_SOURCE_DIR}/../")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")

# add_executable(hopper_gemm gemm.cu) set_target_properties(hopper_gemm
# PROPERTIES CUDA_ARCHITECTURES 90a)

# add_executable(tma_copy tma_copy.cu) set_target_properties(tma_copy PROPERTIES
# CUDA_ARCHITECTURES 90a)

add_executable(tma_copy_multicast tma_copy_multicast.cu)
set_target_properties(tma_copy_multicast PROPERTIES CUDA_ARCHITECTURES 90a)
