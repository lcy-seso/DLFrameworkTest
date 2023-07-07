#pragma once
#include "cuda_timer.cuh"
#include "utils.h"

// A, B and C are stored in row-major style.
template <typename T>
float TestCublasGemm(const T* A, const T* B, T* C, int m, int n, int k);

template <>
float TestCublasGemm(const float* A, const float* B, float* C, int m, int n,
                     int k) {
  return 0.;
}

template <>
float TestCublasGemm<__half>(const __half* A, const __half* B, __half* C, int m,
                             int n, int k) {
  cublasStatus_t stat;
  cublasHandle_t cublas_handle;
  CublasCheck(cublasCreate(&cublas_handle));

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  // cuBlas is a Fortran-style(column-major) BLAS library.
  // When A, B, C are lay out in row major,
  // slyly call cublas as it compute C^T = (AB)^T = (B^T)(A^T).

  for (int i = 0; i < 5; ++i) {  // warmup
    CublasCheck(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, B, n, A, k, &beta, C, n));
  }

  CudaTimer timer;
  timer.Start();
  const int iters = 20;
  for (int i = 0; i < iters; ++i) {  // warmup
    CublasCheck(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, B, n, A, k, &beta, C, n));
  }

  CublasCheck(cublasDestroy(cublas_handle));
  return timer.Stop() / iters;
}
