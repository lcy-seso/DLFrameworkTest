cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

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

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -w -gencode arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -w -gencode
                          arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 -gencode
                            arch=compute_75,code=sm_75)

find_package(CUDA QUIET REQUIRED)


include_directories("../")

cuda_add_executable(load_test main.cu)
target_link_libraries(load_test
                      ${CUDA_curand_LIBRARY})
