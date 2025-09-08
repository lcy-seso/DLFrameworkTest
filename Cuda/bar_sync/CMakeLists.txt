cmake_minimum_required(VERSION 3.25.0)
project(test_bar_sync LANGUAGES CXX CUDA)

set(TARGET bar_sync)

set(CMAKE_BUILD_TYPE Release)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")
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

include_directories(${PROJECT_SOURCE_DIR}/..)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -std=c++17
    -Xcompiler
    -fPIC
    -rdc=true
    --extended-lambda
    --expt-relaxed-constexpr)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                    "-gencode arch=compute_100a,code=sm_100a")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

add_executable(${TARGET} bar_sync.cu ../../curand_fp16.cu)
set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES 100a)
target_link_libraries(${TARGET} PRIVATE CUDA::curand)
