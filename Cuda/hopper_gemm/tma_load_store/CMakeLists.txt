cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(hopper_gemm LANGUAGES C CXX CUDA)
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

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++20)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -std=c++20 -O0)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -std=c++20 -O3)

include(select_compute_arch)
cuda_select_nvcc_arch_flags(ARCH_LIST Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS
     "-Werror -Wall --use_fast_math --expt-relaxed-constexpr")

message(STATUS "NVCC FLAGS = ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA detected: " ${CUDA_VERSION})
message(STATUS "CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})

include_directories("${PROJECT_SOURCE_DIR}/../")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")

add_executable(hopper_gemm main.cu)
add_executable(tma_load tma_load.cu)
