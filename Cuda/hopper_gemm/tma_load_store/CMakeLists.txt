cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(hopper_gemm LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../third_party/")

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "The C++ standard whose features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    20
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall --expt-relaxed-constexpr")

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

message(STATUS "CUDA detected: " ${CUDA_VERSION})
message(STATUS "CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})

include_directories("${PROJECT_SOURCE_DIR}/../")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")

add_executable(hopper_gemm main.cu)
