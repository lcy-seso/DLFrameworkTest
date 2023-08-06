#include <assert.h>

#include <iomanip>
#include <iostream>

#include "../cuda_utils.cuh"
#include "cublass_gemm.h"
#include "cutlass_warp_gemm.cuh"

namespace {
__global__ void InitHalfs(__half* data, int64_t numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numel) {
    data[tid] = __float2half(tid / 1000.);
    // printf("[%d] = %.4f\n", tid, __half2float(data[tid]));
  }
}

void PrintHalfs(const __half* data, int64_t numel) {
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);

  float* d_data;
  CudaCheck(cudaMalloc((void**)&d_data, sizeof(float) * numel));
  const int threads = 128;
  int blocks = CEIL_DIV(numel, threads);
  ConvertFp16ToFp32<<<blocks, threads>>>(d_data, data, numel);

  float* h_data;
  CudaCheck(cudaMallocHost((void**)&h_data, sizeof(float) * numel));
  CudaCheck(cudaMemcpy(h_data, d_data, sizeof(float) * numel,
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < numel; ++i) {
    std::cout << i << ":" << h_data[i] << std::endl;
  }

  CudaCheck(cudaFree(d_data));
  CudaCheck(cudaFreeHost(h_data));
}
}  // namespace

int main(int argc, char** argv) {
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(5);

  const int m = 64;
  const int n = 64;
  const int k = 64;

  int64_t size_A = m * k;
  int64_t size_B = k * n;
  int64_t size_C = m * n;

  int threads = 128;
  int blocks = CEIL_DIV(size_A, threads);

  __half *dA_fp16, *dB_fp16, *dC_fp16_base, *dC_fp16;
  __half* dA_fp16_2;

  CudaCheck(cudaMalloc(&dA_fp16, size_A * sizeof(__half)));
  CudaCheck(cudaMalloc(&dA_fp16_2, size_A * sizeof(__half)));

  CudaCheck(cudaMalloc(&dB_fp16, size_B * sizeof(__half)));
  InitHalfs<<<blocks, threads>>>(dA_fp16, size_A);
  FillZeros<<<blocks, threads>>>(dA_fp16_2, size_A);
  //   std::cout << "Input matrix A: " << std::endl << std::endl;
  //   PrintHalfs(dA_fp16, size_A);

  InitHalfs<<<blocks, threads>>>(dB_fp16, size_B);

  //   std::cout << std::endl
  //             << std::endl
  //             << "Input matrix B: " << std::endl
  //             << std::endl;
  //   PrintHalfs(dB_fp16, size_B);

  //   InitRandomHalfs(dA_fp16, size_A);
  //   InitRandomHalfs(dB_fp16, size_B);

  CudaCheck(cudaMalloc(&dC_fp16_base, size_C * sizeof(__half)));
  CudaCheck(cudaMalloc(&dC_fp16, size_C * sizeof(__half)));

  blocks = CEIL_DIV(size_C, threads);
  FillZeros<__half><<<threads, blocks>>>(dC_fp16_base, size_C);  // for cublass
  FillZeros<__half><<<threads, blocks>>>(dC_fp16, size_C);       // for cutlass

  float time1 = TestCublasGemm(dA_fp16, dB_fp16, dC_fp16_base, m, n, k);
  //   PrintHalfs(dC_fp16_base, size_C);

  using WholeShape = cutlass::gemm::GemmShape<m, n, k>;

  // thread-block tile shape.
  // const int M_s = 32;
  // const int N_s = 32;
  // const int K_s = 64;
  const int M_s = m;
  const int N_s = n;
  const int K_s = k;
  using ThreadBlockShape = cutlass::gemm::GemmShape<M_s, N_s, K_s>;

  // warp-tile shape.
  const int M_w = 32;
  const int N_w = 32;
  const int K_w = 16;
  using WarpShape = cutlass::gemm::GemmShape<M_w, N_w, K_w>;

  // tensor core instruction shape.
  const int gemm_threads = (M_s / M_w) * (N_s / N_w) * 32;
  // std::cout << "threads = " << gemm_threads << std::endl;

  const int M_i = 16;
  const int N_i = 8;
  const int K_i = 8;
  // mma shape computed by a tensor core API
  using InstructionShape = cutlass::gemm::GemmShape<M_i, N_i, K_i>;

  using Element = cutlass::half_t;
  using ElementC = cutlass::half_t;

  const int crosswise = 32;
  // two contiguous dimensions occupy a single shared memory cache line
  // The Target Layout
  using SLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;
  using SLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;
  using SLayoutC = cutlass::layout::RowMajor;

  // warp-level mma API.
  // compute
  // using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
  //     WarpShape /*warp-tile shape*/,
  //     InstructionShape /*tensor core instruction shape*/,
  //     Element /*element type of A*/,
  //     SLayoutA /*the layout of the A tile on shared memory*/,
  //     Element /*element type of B*/,
  //     SLayoutB /*the layout of the B tile on the shared memory*/,
  //     ElementC /*type of accumulator*/, SLayoutC>::Type;

  IterateATest<WholeShape, ThreadBlockShape, WarpShape, SLayoutA>(
      reinterpret_cast<cutlass::half_t*>(dA_fp16),
      reinterpret_cast<cutlass::half_t*>(dA_fp16_2));

  // CutlassGemm<MmaTensorOp, WholeShape, ThreadBlockShape>(
  //     reinterpret_cast<cutlass::half_t*>(dC_fp16),
  //     reinterpret_cast<cutlass::half_t*>(dA_fp16),
  //     reinterpret_cast<cutlass::half_t*>(dB_fp16));

  // blocks = CEIL_DIV(size_C, threads);
  // CheckDiff<<<blocks, threads>>>(dC_fp16_base, dC_fp16, size_C);

  //   cudaFree(dA_fp32);
  //   cudaFree(dB_fp32);
  //   cudaFree(dC_fp32);

  cudaFree(dA_fp16);
  cudaFree(dB_fp16);
  cudaFree(dC_fp16);
  cudaFree(dC_fp16_base);

  return 0;
}
