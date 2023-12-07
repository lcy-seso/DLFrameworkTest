cmake_minimum_required(VERSION 3.18)
project(cute_copy CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -std=c++17 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++17 -fPIC -O3 -Wall
    -Wno-sign-compare")

find_package(CUDA QUIET REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -std=c++17 -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -std=c++17 -w -O3
                            ${ARCH_FLAGS})

set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
  "../../build/third_party/cutlass/src/extern_cutlass/include")
include_directories(
  "../../build/third_party/cutlass/src/extern_cutlass/tools/util/include")
include_directories("../../")
cuda_add_executable(mma mma.cu)

cuda_add_executable(ldmatrix ldmatrix.cu)
