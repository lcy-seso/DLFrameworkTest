#pragma once

#include <cuda_fp16.h>

typedef __nv_bfloat16 bfloat16;
typedef __nv_bfloat162 bfloat162;

template <typename T>
__host__ __device__ float ToFloat(T val);

template <typename T>
__host__ __device__ T FromFloat(float val);

template <>
__host__ __device__ float ToFloat(bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__host__ __device__ bfloat16 FromFloat(float val) {
  return __float2bfloat16(val);
}
