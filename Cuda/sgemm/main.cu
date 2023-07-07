#include <iomanip>
#include <iostream>

#include "cuda_timer.cuh"
#include "cutlass_warp_gemm.cuh"
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

template <typename Mma,               /// Warp-level matrix multiply-accumulate
          typename ThreadblockShape,  /// Size of threadblock-scoped shape used
                                      /// to store SMEM
          typename WholeShape,        /// Size of kernel-scoped shape
                                      /// The inner product operation
                                      /// performed by GEMM
          typename Operator = cutlass::arch::OpMultiplyAdd>
float CutlassGemmKernel(const half* A, const half* B, half* C) {
  // using Mma = Mma_;
  // using ThreadblockShape = ThreadblockShape_;
  // using WholeShape = WholeWShape_;
  // using Operator = Operator_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename cutlass::layout::RowMajor;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename cutlass::layout::ColumnMajor;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

  const uint BLOCKM = CEIL_DIV(WholeShape::kM, ThreadblockShape::kM);
  const uint BLOCKN = CEIL_DIV(WholeShape::kN, ThreadblockShape::kN);
  const uint BLOCKk = WholeShape::kK / ThreadblockShape::kK;
  dim3 gridDim(BLOCKN, BLOCKM);

  const uint WARPM = CEIL_DIV(ThreadblockShape::kM, Shape::kM);
  const uint WARPN = CEIL_DIV(ThreadblockShape::kN, Shape::kN);
  const uint WARPSIZE = 32;
  const uint THREADNUM = WARPSIZE * WARPM * WARPN;
  dim3 blockDim(THREADNUM, 1, 1);

  WarpAPI_GEMM<Mma, ThreadblockShape, WholeShape, THREADNUM>
      <<<gridDim, blockDim>>>(C, A, B);
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

  // using WholeShape = cutlass::gemm::GemmShape<256, 256, 128>;
  // using BlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  // using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  // using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  // using Element = cutlass::half_t;
  // using ElementC = float;

  // using GLayoutA = cutlass::layout::RowMajor;
  // using GLayoutB = cutlass::layout::ColumnMajor;
  // using SLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
  //     cutlass::sizeof_bits<Element>::value, 64>;
  // using SLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
  //     cutlass::sizeof_bits<Element>::value, 64>;

  // using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
  //     WarpShape, InstructionShape, Element, SLayoutA, Element, SLayoutB,
  //     ElementC, cutlass::layout::RowMajor>::Type;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_base);

  return 0;
}
