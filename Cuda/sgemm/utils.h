#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <stdio.h>

#include <iostream>
#include <sstream>

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

#define cudaErrCheck(stat) \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char* file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}

#define cutlassErrCheck(stat)                                             \
  {                                                                       \
    cutlass::Status error = stat;                                         \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

namespace {
template <typename T>
__global__ void naiveFillKernel(T* data, int num, T value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num) data[tid] = value;
}
}  // namespace

void fillZeros(float* data, int numel, float val = 0.) {
  int block = 512;
  int grid = (numel + block - 1) / block;

  naiveFillKernel<float><<<grid, block>>>(data, numel, val);
}

void fillRandom(float* A, int elementNum) {
  // create a pseudo-random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // set the seed for the random number generator using the system clock
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // fill the array with random numbers on the device
  curandGenerateUniform(prng, A, elementNum);
}

void printer(float* data, int numel) {
  float* data_cpu = (float*)malloc(numel * sizeof(float));
  cudaErrCheck(cudaMemcpy(data_cpu, data, numel * sizeof(float),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) std::cout << data_cpu[i] << std::endl;

  free(data_cpu);
}

void CheckDiff(const float* data1, const float* data2, int numel) {
  float* data1_cpu = (float*)malloc(numel * sizeof(float));
  cudaErrCheck(cudaMemcpy(data1_cpu, data1, numel * sizeof(float),
                          cudaMemcpyDeviceToHost));

  float* data2_cpu = (float*)malloc(numel * sizeof(float));
  cudaErrCheck(cudaMemcpy(data2_cpu, data2, numel * sizeof(float),
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

std::string strLaunchConfig(dim3 blocks, dim3 threads) {
  std::stringstream ss;
  ss << "Grid : {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
     << std::endl
     << "Blocks: {" << threads.x << ", " << threads.y << ", " << threads.z
     << "}";
  return ss.str();
}

// Matrix A, B and C are stored in row major fashion
float testCuBLASGemmRowMajorABC(int m, int n, int k, const float* d_A,
                                const float* d_B, float* d_C, int kIters = 10) {
  // Here suppose the input matrices A and B are row-major.
  // Compute C = A @ B, both A and B are not transposed.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // NOTE: cuBLAS is column major
  float elapsed = 0.;

  float alf = 1.0, bet = 0.0;
  const float* alpha = &alf;
  const float* beta = &bet;
  cublasHandle_t handle;
  // create cuBlas handler
  cublasCreate(&handle);

  // warmup invocation.
  cublasSgemm(handle, CUBLAS_OP_N /* trans_b */, CUBLAS_OP_N /* trans_a */, n,
              m, k, alpha, d_B, n, d_A, k, beta, d_C, n);
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < kIters; ++i)
    cublasSgemm(handle, CUBLAS_OP_N /* trans_b */, CUBLAS_OP_N /* trans_a */, n,
                m, k, alpha, d_B, n, d_A, k, beta, d_C, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed, start, stop);

  return elapsed / kIters;
}

// Matrix A, B and C are stored in column major fashion
float testCuBLASGemmColumnMajorABC(int m, int n, int k, const float* d_A,
                                   const float* d_B, float* d_C,
                                   int kIters = 10) {
  // Here suppose the input matrices A and B are row-major.
  // Compute C = A @ B, both A and B are not transposed.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // NOTE: cuBLAS is column major
  float elapsed = 0.;

  float alf = 1.0, bet = 0.0;
  const float* alpha = &alf;
  const float* beta = &bet;
  cublasHandle_t handle;
  // create cuBlas handler
  cublasCreate(&handle);

  // warmup invocation.
  cublasSgemm(handle, CUBLAS_OP_N /* trans_b */, CUBLAS_OP_N /* trans_a */, m,
              n, k, alpha, d_A, m, d_B, k, beta, d_C, m);
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < kIters; ++i) {
    cublasSgemm(handle, CUBLAS_OP_N /* trans_b */, CUBLAS_OP_N /* trans_a */, m,
                n, k, alpha, d_A, m, d_B, k, beta, d_C, m);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed, start, stop);

  return elapsed / kIters;
}
