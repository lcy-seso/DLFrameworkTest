#pragma once

#include <cublasLt.h>
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

void cublas_bf16_gemm(int64_t kM, int64_t kN, int64_t kK,  // problem shape
                      const __nv_bfloat16* A, const __nv_bfloat16* B,
                      __nv_bfloat16* C) {
  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  cublasLtMatrixLayout_t A_desc, B_desc, C_desc;

  // Original layouts:
  //     A (row-major M×K), B (column-major K×N), C (row-major M×N)
  // For cuBLAS column-major, we compute C^T = B^T @ A^T
  //     [N, M] = [N, K] @ [K, M]
  cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, kK, kN, kK);
  cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, kK, kM, kK);
  cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16BF, kN, kM, kN);

  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  // For cuBLAS column-major, we compute C^T = B^T @ A^T
  //     [N, M] = [N, K] @ [K, M]
  // Compute C^T = B^T @ A^T
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;

  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb, sizeof(transb));

  cublasDataType_t scale_type = CUDA_R_32F;
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scale_type, sizeof(scale_type));

  float alpha = 1.0f;
  float beta = 0.0f;

  size_t workspace_size = 32 * 1024 * 1024;  // 32 MiB workspace
  void* workspace = nullptr;
  cudaMalloc(&workspace, workspace_size);

  cublasLtMatmulPreference_t preference;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference,
                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspace_size, sizeof(workspace_size));

  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_results = 0;

  cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_desc, C_desc,
                                 C_desc, preference, 1, &heuristic_result,
                                 &returned_results);

  if (returned_results == 0) {
    std::cerr << "No algorithm found!" << std::endl;
    exit(EXIT_FAILURE);
  }

  cublasLtMatmul(handle, matmul_desc,  //
                 &alpha, B,
                 A_desc,            // A matrix
                 A, B_desc, &beta,  // B matrix
                 C, C_desc,         //
                 C, C_desc,         //
                 &heuristic_result.algo, workspace, workspace_size, 0);

  cudaDeviceSynchronize();
  cublasLtMatmulDescDestroy(matmul_desc);
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(C_desc);
  cublasLtDestroy(handle);
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

// Naive CPU matrix multiplication: C = A * B
// A: row-major (M×K), B: column-major (K×N), C: row-major (M×N)
void cpu_naive_gemm(int64_t M, int64_t N, int64_t K, const __nv_bfloat16* A,
                    const __nv_bfloat16* B, __nv_bfloat16* C) {
  for (int64_t i = 0; i < M * N; ++i) {
    C[i] = __float2bfloat16(0.0f);
  }

  // Compute C[i,j] = sum_k A[i,k] * B[k,j]
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        // A[i,k]: row-major layout, element at (i,k) is at index i*K + k
        float a_val = __bfloat162float(A[i * K + k]);

        // B[k,j]: column-major layout, element at (k,j) is at index k + j*K
        float b_val = __bfloat162float(B[k + j * K]);

        sum += a_val * b_val;
      }
      // C[i,j]: row-major layout, element at (i,j) is at index i*N + j
      C[i * N + j] = __float2bfloat16(sum);
    }
  }
}
