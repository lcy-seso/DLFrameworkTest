
#include <assert.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>

#include "access1.cuh"
#include "access2.cuh"
#include "cuda_timer.cuh"
#include "cuda_utils.cuh"
#include "utils.h"

#define BANDWIDTH 616.

template <typename T>
void RandomInput(T* data, int n, T min = 0., T max = 1.) {
  for (size_t i = 0; i < n; ++i)
    data[i] = ((T)rand() / RAND_MAX) * (max - min) + min;
}

template <typename T, int GRID_SIZE, int BLOCK_SIZE>
float TestCopy1(int height, int width) {
  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(6);
  srand(time(NULL));

  CudaTimer timer;
  int numel = height * width;

  // create input
  T* h_input;
  CudaCheck(cudaMallocHost((void**)&h_input, sizeof(T) * numel));
  RandomInput<float>(h_input, numel);
  T* d_input;
  CudaCheck(cudaMalloc((void**)&d_input, sizeof(T) * numel));
  CudaCheck(
      cudaMemcpy(d_input, h_input, sizeof(T) * numel, cudaMemcpyHostToDevice));

  T* d_output;
  CudaCheck(cudaMalloc((void**)&d_output, sizeof(T) * numel));
  T* h_output;
  CudaCheck(cudaMallocHost((void**)&h_output, sizeof(T) * numel));

  int blocks = CEIL_DIV(width, BLOCK_SIZE);
  CopyTest1<T, GRID_SIZE, BLOCK_SIZE>
      <<<height, BLOCK_SIZE, sizeof(T) * width>>>(d_input, d_output, height,
                                                  width);

  timer.Start();
  for (size_t i = 0; i < 10; ++i) {
    CopyTest1<T, GRID_SIZE, BLOCK_SIZE>
        <<<height, BLOCK_SIZE, sizeof(T) * width>>>(d_input, d_output, height,
                                                    width);
  }
  float elapsed = timer.Stop();

  CudaCheck(cudaMemcpy(h_output, d_output, sizeof(T) * numel,
                       cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < numel; ++i) {
    assert(FloatEqual(h_output[i], h_input[i]));
  }

  CudaCheck(cudaFreeHost(h_input));
  CudaCheck(cudaFree(d_input));
  CudaCheck(cudaFreeHost(h_output));
  CudaCheck(cudaFree(d_output));
  return elapsed / 10.;
}

template <typename T, int GRID_SIZE, int BLOCK_SIZE>
float TestCopy2(int height, int width) {
  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(6);
  srand(time(NULL));

  CudaTimer timer;
  int numel = height * width;

  // create input
  T* h_input;
  CudaCheck(cudaMallocHost((void**)&h_input, sizeof(T) * numel));
  RandomInput<float>(h_input, numel);
  T* d_input;
  CudaCheck(cudaMalloc((void**)&d_input, sizeof(T) * numel));
  CudaCheck(
      cudaMemcpy(d_input, h_input, sizeof(T) * numel, cudaMemcpyHostToDevice));

  T* d_output;
  CudaCheck(cudaMalloc((void**)&d_output, sizeof(T) * numel));
  T* h_output;
  CudaCheck(cudaMallocHost((void**)&h_output, sizeof(T) * numel));

  DirectLoad<T, T> load(d_input, width);
  DirectStore<T, T> store(d_output, width);

  int blocks = CEIL_DIV(width, BLOCK_SIZE);

  const int smem = width * sizeof(T);

  if (width % 2 == 0) {
    CopyTest2<T, decltype(load), decltype(store), GRID_SIZE, BLOCK_SIZE, 2>
        <<<height, BLOCK_SIZE, smem>>>(load, store, height, width);
  } else {
    CopyTest2<T, decltype(load), decltype(store), GRID_SIZE, BLOCK_SIZE, 1>
        <<<height, BLOCK_SIZE, smem>>>(load, store, height, width);
  }

  timer.Start();
  for (size_t i = 0; i < 10; ++i) {
    if (width % 2 == 0) {
      CopyTest2<T, decltype(load), decltype(store), GRID_SIZE, BLOCK_SIZE, 2>
          <<<height, BLOCK_SIZE, smem>>>(load, store, height, width);
    } else {
      CopyTest2<T, decltype(load), decltype(store), GRID_SIZE, BLOCK_SIZE, 1>
          <<<height, BLOCK_SIZE, smem>>>(load, store, height, width);
    }
  }
  float elapsed = timer.Stop();

  CudaCheck(cudaMemcpy(h_output, d_output, sizeof(T) * numel,
                       cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < numel; ++i) {
    // std::cout << h_output[i] << " vs. " << h_input[i] << std::endl;
    assert(FloatEqual(h_output[i], h_input[i]));
  }

  CudaCheck(cudaFreeHost(h_input));
  CudaCheck(cudaFree(d_input));
  CudaCheck(cudaFreeHost(h_output));
  CudaCheck(cudaFree(d_output));
  return elapsed / 10.;
}

int main(int argc, char* argv[]) {
  int height = 30000;
  int width = 1024;

  // std::cout << TestCopy1<float, 3000, 128>(height, width) << std::endl;
  // std::cout << TestCopy2<float, 3000, 128>(height, width) << std::endl;
  // {
  //   // block size 128
  //   std::cout << TestCopy1<float, 3000, 128>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 6000, 128>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 15000, 128>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 30000, 128>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 60000, 128>(height, width) << std::endl;

  //   // block size 256
  //   std::cout << TestCopy1<float, 3000, 256>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 6000, 256>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 15000, 256>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 30000, 256>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 60000, 256>(height, width) << std::endl;

  //   // block size 512
  //   std::cout << TestCopy1<float, 3000, 512>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 6000, 512>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 15000, 512>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 30000, 512>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 60000, 512>(height, width) << std::endl;

  //   // block size 1024
  //   std::cout << TestCopy1<float, 3000, 1024>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 6000, 1024>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 15000, 1024>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 30000, 1024>(height, width) << std::endl;
  //   std::cout << TestCopy1<float, 60000, 1024>(height, width) << std::endl;
  // }

  {
    // block size 128
    std::cout << TestCopy2<float, 3000, 128>(height, width) << std::endl;
    std::cout << TestCopy2<float, 6000, 128>(height, width) << std::endl;
    std::cout << TestCopy2<float, 15000, 128>(height, width) << std::endl;
    std::cout << TestCopy2<float, 30000, 128>(height, width) << std::endl;
    std::cout << TestCopy2<float, 60000, 128>(height, width) << std::endl;

    // block size 256
    std::cout << TestCopy2<float, 3000, 256>(height, width) << std::endl;
    std::cout << TestCopy2<float, 6000, 256>(height, width) << std::endl;
    std::cout << TestCopy2<float, 15000, 256>(height, width) << std::endl;
    std::cout << TestCopy2<float, 30000, 256>(height, width) << std::endl;
    std::cout << TestCopy2<float, 60000, 256>(height, width) << std::endl;

    // block size 512
    std::cout << TestCopy2<float, 3000, 512>(height, width) << std::endl;
    std::cout << TestCopy2<float, 6000, 512>(height, width) << std::endl;
    std::cout << TestCopy2<float, 15000, 512>(height, width) << std::endl;
    std::cout << TestCopy2<float, 30000, 512>(height, width) << std::endl;
    std::cout << TestCopy2<float, 60000, 512>(height, width) << std::endl;

    // block size 1024
    std::cout << TestCopy2<float, 3000, 1024>(height, width) << std::endl;
    std::cout << TestCopy2<float, 6000, 1024>(height, width) << std::endl;
    std::cout << TestCopy2<float, 15000, 1024>(height, width) << std::endl;
    std::cout << TestCopy2<float, 30000, 1024>(height, width) << std::endl;
    std::cout << TestCopy2<float, 60000, 1024>(height, width) << std::endl;
  }

  return 0;
}
