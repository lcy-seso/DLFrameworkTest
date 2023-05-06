#include <gtest/gtest.h>
#include <stdio.h>

#include <iostream>

#include "cuda_utils.cuh"
#include "scan.cuh"
#include "utils.h"

template <typename T>
void checkResults(const T* data, int N, T init_val) {
  T val = init_val;
  for (size_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(val, data[i]) << "the " << i << "-th element";
    val += static_cast<T>(1);
  }
}

int main(int argc, char* argv[]) {
  InitGLOG("scan_test");

  const unsigned int blockSize = 32;
  int height = 1;

  // int width = 2;
  // int width = blockSize + 17;
  // int width = (blockSize * 2) + 17;
  int width = (blockSize * 2) * (blockSize * 2) + 17;

  int numel = height * width;

  float* h_input;
  float* h_output;
  float* d_input;
  float* d_output;

  CudaCheck(cudaMalloc((void**)&d_input, sizeof(float) * numel));
  CudaCheck(cudaMalloc((void**)&d_output, sizeof(float) * numel));

  int blocks = (numel + blockSize - 1) / blockSize;
  std::cout << "num blocks: " << blocks << std::endl;

  fillValue<float><<<blocks, blockSize>>>(d_input, numel, 1.);

  fullExlusiveScan<float, blockSize>(d_input, d_output, numel);

  // check data
  CudaCheck(cudaMallocHost((void**)&h_input, sizeof(float) * numel));
  CudaCheck(cudaMemcpy(h_input, d_input, sizeof(float) * numel,
                       cudaMemcpyDeviceToHost));

  CudaCheck(cudaMallocHost((void**)&h_output, sizeof(float) * numel));
  CudaCheck(cudaMemcpy(h_output, d_output, sizeof(float) * numel,
                       cudaMemcpyDeviceToHost));
  // checkResults<float>(h_output, numel, 0.);
  // printValue<float>(d_output, numel);

  CudaCheck(cudaFreeHost(h_output));
  CudaCheck(cudaFree(d_input));
  CudaCheck(cudaFree(d_output));

  return 0;
}
