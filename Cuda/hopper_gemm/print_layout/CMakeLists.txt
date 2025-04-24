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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

include_directories("${THIRD_PARTY_DIR}/cutlass/include")

add_executable(print_layout print_layout.cu)
