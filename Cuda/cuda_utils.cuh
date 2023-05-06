#pragma once

#include <cuda.h>

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

template <typename T>
__global__ void fillValue(T* data, int N, T val) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) data[tid] = val;
}

template <typename T>
void printValue(const T* data, int numel) {
  // data is on the device
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);

  T* h_data;
  CudaCheck(cudaMallocHost((void**)&h_data, sizeof(T) * numel));
  CudaCheck(
      cudaMemcpy(h_data, data, sizeof(T) * numel, cudaMemcpyDeviceToHost));
  std::cout << "[printValue] numel = " << numel << std::endl;

  for (int i = 0; i < numel; ++i) {
    std::cout << i << ":" << h_data[i] << std::endl;
  }

  CudaCheck(cudaFreeHost(h_data));
}
