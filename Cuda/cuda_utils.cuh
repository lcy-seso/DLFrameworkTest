#pragma once

#include <cublas_v2.h>
#include <cuda.h>

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

template <typename T>
__global__ void FillZeros(T* data, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num) data[tid] = static_cast<T>(0.);
}

void PrintFloats(float* data, int numel) {
  // data is on the device
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);

  float* h_data;
  CudaCheck(cudaMallocHost((void**)&h_data, sizeof(float) * numel));
  CudaCheck(
      cudaMemcpy(h_data, data, sizeof(float) * numel, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) {
    std::cout << i << ":" << h_data[i] << std::endl;
  }

  CudaCheck(cudaFreeHost(h_data));
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
