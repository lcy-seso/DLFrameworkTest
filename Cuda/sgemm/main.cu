#include <iomanip>
#include <iostream>

#include "cuda_timer.cuh"
#include "utils.h"

float TestGemmKernel(int test_num, int m, int n, int k, const float* d_A,
                     const float* d_B, float* d_C,
                     float* d_ground_truth = nullptr, const int kIters = 1) {
  float alpha = 1.;
  float beta = 0.;

  float elapsed = 0.;
  switch (test_num) {
    case -1:
      elapsed = testCuBLASGemmRowMajorABC(m, n, k, d_A, d_B, d_C, kIters);
      break;
  }

  if (d_ground_truth) {
    CheckDiff(d_C, d_ground_truth, m * n);
  }

  return elapsed;
}

int main(int argc, char** argv) {
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(4);

  int m = 256;
  int k = 512;
  int n = 1024;

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_base = nullptr;
  float* d_C = nullptr;

  int64_t size_A = m * k;
  int64_t size_B = k * n;
  int64_t size_C = m * n;

  cudaErrCheck(cudaMalloc(&d_A, size_A * sizeof(float)));
  cudaErrCheck(cudaMalloc(&d_B, size_B * sizeof(float)));
  cudaErrCheck(cudaMalloc(&d_C, size_C * sizeof(float)));
  cudaErrCheck(cudaMalloc(&d_base, size_C * sizeof(float)));

  fillRandom(d_A, size_A);
  fillRandom(d_B, size_B);
  fillZeros(d_base, size_C);
  fillZeros(d_C, size_C);

  std::cout << "Shape[m,n,k]\tTest Name\tElapsed Time(ms)\tRatio" << std::endl;

  // cublas as the baseline.
  float base = TestGemmKernel(-1 /*cuBLAS*/, m, n, k, d_A, d_B, d_base);
  fillZeros(d_C, size_C);

  std::stringstream ss;
  ss << "[" << m << "," << n << "," << k << "]\t";
  std::cout << ss.str() << "cuBLAS\t" << base << "\t1." << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_base);

  return 0;
}
