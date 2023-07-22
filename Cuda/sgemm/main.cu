#include <iomanip>
#include <iostream>

#include "../cuda_utils.cuh"
#include "cublass_gemm.h"
#include "cutlass_warp_gemm.cuh"

int main(int argc, char** argv) {
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(5);

  const int m = 64;
  const int n = 64;
  const int k = 64;

  int64_t size_A = m * k;
  int64_t size_B = k * n;
  int64_t size_C = m * n;

  __half *dA_fp16, *dB_fp16, *dC_fp16_base, *dC_fp16;

  // fp32 counterparts
  const int threads = 128;
  float *dA_fp32, *dB_fp32, *dC_fp32;
  CudaCheck(cudaMalloc(&dA_fp32, size_A * sizeof(float)));
  int blocks = CEIL_DIV(size_A, threads);
  CudaCheck(cudaMalloc(&dB_fp32, size_B * sizeof(float)));
  blocks = CEIL_DIV(size_B, threads);
  CudaCheck(cudaMalloc(&dC_fp32, size_C * sizeof(float)));

  // fp16 inputs and outputs
  CudaCheck(cudaMalloc(&dA_fp16, size_A * sizeof(__half)));
  blocks = CEIL_DIV(size_A, threads);
  //   InitSeq<<<threads, blocks>>>(dA_fp16, size_A);
  //   ConvertFp16ToFp32<<<threads, blocks>>>(dA_fp32, dA_fp16, size_A);
  //   PrintValue<__half>(dA_fp16, size_A);
  InitRandomHalfs(dA_fp16, size_A);

  CudaCheck(cudaMalloc(&dB_fp16, size_B * sizeof(__half)));
  blocks = CEIL_DIV(size_B, threads);
  //   InitSeq<<<threads, blocks>>>(dB_fp16, size_B);
  InitRandomHalfs(dB_fp16, size_B);

  CudaCheck(cudaMalloc(&dC_fp16_base, size_C * sizeof(__half)));
  CudaCheck(cudaMalloc(&dC_fp16, size_C * sizeof(__half)));
  FillZeros<__half><<<threads, blocks>>>(dC_fp16_base, size_C);  // for cublass
  FillZeros<__half><<<threads, blocks>>>(dC_fp16, size_C);       // for cutlass

  float time1 = TestCublasGemm(dA_fp16, dB_fp16, dC_fp16_base, m, n, k);

  using Element = cutlass::half_t;
  using ElementC = cutlass::half_t;

  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WholeShape = cutlass::gemm::GemmShape<64, 64, 64>;

  // global memory tile layout
  using GLayoutA = cutlass::layout::RowMajor;
  using GLayoutB = cutlass::layout::ColumnMajor;

  // Crosswise means the contiguous dimension is K dimension, the strided
  // dimension is M or N dimension.
  using SLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16 /*crosswise*/>;
  using SLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;

  // mma computed by a warp
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  // mma shape computed by a tensor core API
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  // warp-level mma API.
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element /*element type of A*/,
      SLayoutA /*the layout of the A tile on shared memory*/,
      Element /*element type of B*/,
      SLayoutB /*the layout of the B tile on the shared memory*/, ElementC,
      cutlass::layout::RowMajor>::Type;

  CutlassGemm<MmaTensorOp, ThreadBlockShape, WholeShape>(
      reinterpret_cast<cutlass::half_t*>(dC_fp16),
      reinterpret_cast<cutlass::half_t*>(dA_fp16),
      reinterpret_cast<cutlass::half_t*>(dB_fp16));

  blocks = CEIL_DIV(size_C, threads);
  CheckDiff<<<blocks, threads>>>(dC_fp16_base, dC_fp16, size_C);

  //   cudaFree(dA_fp32);
  //   cudaFree(dB_fp32);
  //   cudaFree(dC_fp32);

  cudaFree(dA_fp16);
  cudaFree(dB_fp16);
  cudaFree(dC_fp16);
  cudaFree(dC_fp16_base);

  return 0;
}
