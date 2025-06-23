#include "cuda_utils.cuh"
#include "gemm.cuh"

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <typename DType, const int kM, const int kN>
void print_values(const DType* tensor, int start = 256, int cutoff = 128) {
  std::cout << std::fixed << std::setprecision(3);
  for (int i = start; i < kM * kN; ++i) {
    std::cout << static_cast<float>(tensor[i]) << ", ";
    if ((i + 1) % 16 == 0) std::cout << std::endl;

    if (i == (start + cutoff - 1)) break;
  }
}

int main() {
  using DType = __nv_bfloat16;
  // using DType = __nv_fp8_e4m3;

  static constexpr uint64_t kM = 256;
  static constexpr uint64_t kN = 128;
  static constexpr uint64_t kK = 128;

  static constexpr uint64_t kTM = 32;
  static constexpr uint64_t kTN = 64;
  static constexpr uint64_t kTK = 64;

  using Traits = GemmTraits<DType, kM, kN, kK, kTM, kTN, kTK>;

  thrust::host_vector<DType> h_a(kM * kK);
  thrust::host_vector<DType> h_b(kK * kN);
  thrust::host_vector<DType> h_c(kM * kN);

  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<DType>(i % 1024);
    // h_a[i] = static_cast<DType>(rand_float());
  }

  for (int i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<DType>(i % 1024);
    // h_b[i] = static_cast<DType>(rand_float());
  }

  thrust::fill(h_c.begin(), h_c.end(), static_cast<DType>(0));
  CHECK_CUDA(cudaDeviceSynchronize());

  thrust::device_vector<DType> d_a = h_a;
  thrust::device_vector<DType> d_b = h_b;
  thrust::device_vector<DType> d_c = h_c;

  TMADescriptor<DType> tma_desc_a;
  TMADescriptor<DType> tma_desc_b;
  TMADescriptor<DType> tma_desc_c;

  // operand A is laid out in row-major order
  uint64_t global_dim_a[2] = {kM, kK};
  uint32_t shared_dim_a[2] = {kTM, kTK};
  tma_desc_a.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_a.data()),  // Global address
      global_dim_a,                          // Global dimensions
      shared_dim_a,               // Shared memory dimensions (box dimensions)
      kK,                         // Global stride in bytes
      CU_TENSOR_MAP_SWIZZLE_NONE  // Swizzle mode
  );

  // operand B is laid out in column-major order
  uint64_t global_dim_b[2] = {kK, kN};
  uint32_t shared_dim_b[2] = {kTK, kTN};
  tma_desc_b.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_b.data()),  // Global address
      global_dim_b,                          // Global dimensions
      shared_dim_b,               // Shared memory dimensions (box dimensions)
      kK,                         // Global stride in bytes
      CU_TENSOR_MAP_SWIZZLE_NONE  // Swizzle mode
  );

  // operand C is laid out in row-major order
  uint64_t global_dim_c[2] = {kM, kN};
  uint32_t shared_dim_c[2] = {kTM, kTN};
  tma_desc_c.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_c.data()),  // Global address
      global_dim_c,                          // Global dimensions
      shared_dim_c,               // Shared memory dimensions (box dimensions)
      kN,                         // Global stride in bytes
      CU_TENSOR_MAP_SWIZZLE_NONE  // Swizzle mode
  );

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  uint32_t num_sms = deviceProp.multiProcessorCount;

  dim3 blocks(num_sms, 1);
  dim3 threads(Traits::kThreads, 1, 1);

  std::cout << "num_sms: " << num_sms << std::endl;
  std::cout << "threads: " << threads.x << std::endl;
  std::cout << "shared memory size: " << Traits::kSharedMemSize << std::endl;
  std::cout << "shared memory per block: " << deviceProp.sharedMemPerBlock
            << std::endl;

  auto kernel = &hopper_gemm<DType, Traits>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  Traits::kSharedMemSize));

  kernel<<<blocks, threads, Traits::kSharedMemSize>>>(
      tma_desc_a.get_tma_desc(), tma_desc_b.get_tma_desc(),
      tma_desc_c.get_tma_desc());

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  return 0;
}
