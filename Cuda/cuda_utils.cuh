#pragma once

#include "curand_fp16.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <vector>

#define CEIL_DIV(N, D) (((N) + (D) - 1) / (D))

#define CHECK_CU(call)                                                \
  do {                                                                \
    CUresult err = call;                                              \
    if (err != CUDA_SUCCESS) {                                        \
      const char* error_str;                                          \
      cuGetErrorString(err, &error_str);                              \
      fprintf(stderr, "CUDA Driver API error in %s at line %d: %s\n", \
              __FILE__, __LINE__, error_str);                         \
      throw std::runtime_error(error_str);                            \
    }                                                                 \
  } while (0)

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

int GetMaxSharedMemoryPerBlock() {
  int device_id;
  CudaCheck(cudaGetDevice(&device_id));

  cudaDeviceProp prop;
  CudaCheck(cudaGetDeviceProperties(&prop, device_id));
  return prop.sharedMemPerBlock;
}

int GetMaxSharedMemoryPerSM() {
  int device_id;
  CudaCheck(cudaGetDevice(&device_id));

  cudaDeviceProp prop;
  CudaCheck(cudaGetDeviceProperties(&prop, device_id));
  return prop.sharedMemPerMultiprocessor;
}

void InitRandomHalfs(__half* data, int N) {
  constexpr auto rng = CURAND_RNG_PSEUDO_XORWOW;
  curand_fp16::generator_t generator;
  curand_fp16::create(generator, rng);
  curand_fp16::set_seed(generator, 0);

  curand_fp16::normal(generator, data, N, 1e-3, 0.05);

  curand_fp16::destroy(generator);
}

void FillRandomFloats(float* data, int numel) {
  float mean = 0.;
  float stddev = 1e-2;
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateNormal(prng, data, numel, mean, stddev);
}

template <typename DType, const int kM, const int kN>
__forceinline__ __host__ __device__ void print_values(const DType* tensor,
                                                      int start = 256,
                                                      int cutoff = 128) {
  printf("\n");
  for (int i = start; i < kM * kN; ++i) {
    printf("%.0f, ", static_cast<float>(tensor[i]));
    if ((i + 1) % 16 == 0) printf("\n");
    if (i == (start + cutoff - 1)) break;
  }
  printf("\n");
}

__nv_bfloat16 rand_bfloat16() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> distribution(0.0f, 0.5f);

  const float rand_value = std::clamp(distribution(gen), -1.0f, 1.0f);
  return __nv_bfloat16(rand_value);
}

float rand_float(float a = 1e-3, float b = 1) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

float rand_normal(float mean = 0.0f, float stddev = 1.0f) {
  // Box-Muller transform to generate random numbers with Normal distribution
  float u1 = ((float)rand()) / (float)RAND_MAX;
  float u2 = ((float)rand()) / (float)RAND_MAX;

  // Avoid log(0) by ensuring u1 is not zero
  if (u1 < 1e-10f) u1 = 1e-10f;

  // Compute Gaussian random value
  float r = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);

  // Scale and shift to get desired mean and standard deviation
  return mean + stddev * r;
}

template <typename DType>
int check_results(const DType* value1, const DType* value2, int kNumel) {
  // Check if h_src and h_dst are the same
  bool match = true;
  for (int i = 0; i < kNumel; ++i) {
    if (value1[i] != value2[i]) {
      std::cerr << "Verification failed: Mismatch found at index " << i
                << ": value1[" << i << "] = " << value1[i] << ", value2[" << i
                << "] = " << value2[i] << std::endl;
      match = false;
      break;  // Stop at the first mismatch
    }
  }

  if (match) {
    std::cout
        << "Verification successful: h_src and h_dst contain the same values."
        << std::endl;
  } else {
    std::cerr << "Verification failed: h_src and h_dst differ." << std::endl;
    return 1;
  }

#if 0
  int start = 256;
  int end = start + 64;
  std::cout << "\nsrc:\n";
  for (int i = start; i < end; ++i) {
    std::cout << std::fixed << std::setprecision(3) << value1[i] << ", ";
    if (i && (i + 1) % 16 == 0) std::cout << "\n";
  }

  std::cout << "\n\ndst:\n";
  for (int i = start; i < end; ++i) {
    std::cout << std::fixed << std::setprecision(3) << value2[i] << ", ";
    if (i && (i + 1) % 16 == 0) std::cout << "\n";
  }
#endif

  return 0;
}

__device__ __forceinline__ uint32_t get_lane_id() {
  uint32_t lane_id;
  asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ uint32_t get_warp_idx() {
  return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__device__ __forceinline__ uint32_t get_warp_group_idx() {
  return __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
}
