#pragma once

#include <cublas_v2.h>

#include <iomanip>
#include <iostream>

void print_matrix(const __half* data, const int kM, const int kN,
                  int cutoff = -1) {
  std::cout << std::endl << "Matrix: [" << kM << ", " << kN << "]" << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  if (cutoff == -1) {
    cutoff = kM * kN;
  }

  for (int i = 0; i < cutoff; ++i) {
    std::cout << __half2float(data[i]) << ", ";

    if ((i + 1) % 16 == 0) std::cout << std::endl;
  }
}

// In this implementation, matrices A and C are laid out in row-major order,
// while matrix B is laid out in column-major order:
//    C[m, n] = A[m, k] @ B[k, n].
//    In cuBLAS, matrices are by default in column-major order.
//    Therefore, we compute: C^T = (A @ B)^T = B^T @ A^T,
//         [n, m] = [n, k] @ [k, m]. As a result:
//
// 1. The resulting matrix C effectively has the shape [m, n] in row-major
//    order.
// 2. The original matrix B, with shape [k, n] in column-major order,
//    becomes the first operand of GEMM, and the transpose flag is set when
//    calling cublasHgemm.
// 3. The original matrix A, with shape [m, k] in row-major order,
//    becomes the second operand of GEMM. Its transposition results in a
//    column-major matrix, and the transpose flag is not set when calling
//    cublasHgemm.
void cublas_hgemm(int64_t kM, int64_t kN, int64_t kK,  // problem shape
                  const __half* A, const __half* B, __half* C) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  __half alf = static_cast<__half>(1.);
  __half bet = static_cast<__half>(0.);

  cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM, kK, &alf, B,
              kK, A, kK, &bet, C, kN);
  cudaDeviceSynchronize();

  cublasDestroy(handle);
}

void check_result(const __half* h_c, const __half* h_c_ref, int kNumel) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Checking result..." << std::endl;

  const float eps = 1e-3f;
  bool passed = true;

  for (int i = 0; i < kNumel; ++i) {
    float c = __half2float(h_c[i]);
    float c_ref = __half2float(h_c_ref[i]);
    float diff = std::abs(c - c_ref);

    if (diff > eps) {
      std::cout << "Verification failed: Mismatch found at index " << i
                << ": value1[" << i << "] = " << c << ", value2[" << i
                << "] = " << c_ref << ", diff = " << diff << std::endl;
      passed = false;
    }
  }

  if (passed) {
    std::cout << "Verification successful" << std::endl;
  } else {
    std::cout << "Verification failed" << std::endl;
  }
}
