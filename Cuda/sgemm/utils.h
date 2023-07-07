#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <sstream>

#include "curand_fp16.h"

__global__ void ConvertFp16ToFp32(float* out, const __half* in, int64_t numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    out[tid] = __half2float(in[tid]);
    // printf("[%d] = %.6f\n", tid, out[tid]);
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

__global__ void InitSeq(__half* data, int numel) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < numel) {
    data[tid] = __float2half(tid);
    // printf("%d: %f\n", tid, __half2float(data[tid]));
  }
}

void InitRandomHalfs(__half* data, int N) {
  constexpr auto rng = CURAND_RNG_PSEUDO_XORWOW;
  curand_fp16::generator_t generator;
  curand_fp16::create(generator, rng);
  curand_fp16::set_seed(generator, 0);

  curand_fp16::normal(generator, data, N, 1e-3, 0.05);

  curand_fp16::destroy(generator);
}
