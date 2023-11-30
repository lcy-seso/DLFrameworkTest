#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "curand_fp16.h"

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

inline void __cudaCheck(cudaError err, const char* file, int line) {
#ifndef NDEBUG
  if (err != cudaSuccess) {
    fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
#endif
}
#define CudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

#define cutlassErrCheck(stat)                                             \
  {                                                                       \
    cutlass::Status error = stat;                                         \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}

inline void __cublasCheck(cublasStatus_t err, const char* file, int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s(%d): Cublas error: %s.\n", file, line,
            cublasGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define CublasCheck(call) __cublasCheck(call, __FILE__, __LINE__)

__global__ void ConvertFp16ToFp32(float* out, const __half* in, int64_t numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    out[tid] = __half2float(in[tid]);
    // printf("[%d] = %.6f\n", tid, out[tid]);
  }
}

__global__ void ConvertFp32ToFp16(__half* out, const float* in, int64_t numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    printf("[%d] = %.6f\n", tid, in[tid]);
    out[tid] = __float2half(in[tid]);
  }
}

__global__ void CheckDiff(const __half* data1, const __half* data2,
                          int64_t numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    printf("%.6f vs. %.6f\n", __half2float(data1[tid]),
           __half2float(data2[tid]));
  }
}

__global__ void InitSeq(float* data, int numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    data[tid] = tid;
    // printf("%d: %f\n", tid, __half2float(data[tid]));
  }
}

__global__ void InitHalfs(__half* data, int64_t numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numel) {
    data[tid] = __float2half(tid);
    // printf("[%d] = %.4f\n", tid, __half2float(data[tid]));
  }
}

__global__ void InitHalfZeros(__half* data, int64_t numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numel) {
    data[tid] = __float2half(0.);
  }
}

void CheckDiff(const float* data1, const float* data2, int numel) {
  float* data1_cpu = (float*)malloc(numel * sizeof(float));
  CudaCheck(cudaMemcpy(data1_cpu, data1, numel * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float* data2_cpu = (float*)malloc(numel * sizeof(float));
  CudaCheck(cudaMemcpy(data2_cpu, data2, numel * sizeof(float),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) {
    if (abs(data1_cpu[i] - data2_cpu[i]) >= 1e-3) {
      std::cout << i << ": " << data1_cpu[i] << " vs. " << data2_cpu[i]
                << std::endl;
    }
  }

  free(data1_cpu);
  free(data2_cpu);
}

void InitRandomHalfs(__half* data, int N) {
  constexpr auto rng = CURAND_RNG_PSEUDO_XORWOW;
  curand_fp16::generator_t generator;
  curand_fp16::create(generator, rng);
  curand_fp16::set_seed(generator, 0);

  curand_fp16::normal(generator, data, N, 1e-3, 0.05);

  curand_fp16::destroy(generator);
}

void PrintHalfs(const __half* data, int64_t numel, int64_t delimiter_num = 0) {
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0);

  float* d_data;
  CudaCheck(cudaMalloc((void**)&d_data, sizeof(float) * numel));
  const int threads = 128;
  int blocks = CEIL_DIV(numel, threads);
  ConvertFp16ToFp32<<<blocks, threads>>>(d_data, data, numel);

  float* h_data;
  CudaCheck(cudaMallocHost((void**)&h_data, sizeof(float) * numel));
  CudaCheck(cudaMemcpy(h_data, d_data, sizeof(float) * numel,
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) {
    std::cout << h_data[i] << ", ";
    if (delimiter_num && i % (delimiter_num + 1) == 0) std::cout << std::endl;
  }

  CudaCheck(cudaFree(d_data));
  CudaCheck(cudaFreeHost(h_data));
}

template <typename T>
__global__ void FillZeros(T* data, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num) data[tid] = static_cast<T>(0.);
}

void PrintFloats(float* data, int numel) {
  // data is on the device
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5);

  float* h_data;
  CudaCheck(cudaMallocHost((void**)&h_data, sizeof(float) * numel));
  CudaCheck(
      cudaMemcpy(h_data, data, sizeof(float) * numel, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) {
    std::cout << h_data[i] << ",";
    if ((i + 1) % 8 == 0) std::cout << std::endl;
  }

  CudaCheck(cudaFreeHost(h_data));
}

void FillRandomFloats(float* data, int numel) {
  float mean = 0.;
  float stddev = 1e-2;
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateNormal(prng, data, numel, mean, stddev);
}
