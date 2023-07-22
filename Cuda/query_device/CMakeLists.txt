cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

project(access_pattern CXX C)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall -Wno-sign-compare")

find_package(CUDA QUIET REQUIRED)

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 ${ARCH_FLAGS})

include_directories("../")

cuda_add_executable(device_info main.cu)
target_link_libraries(device_info ${CUDA_curand_LIBRARY})
