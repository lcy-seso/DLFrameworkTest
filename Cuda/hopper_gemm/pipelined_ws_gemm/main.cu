#include "cuda_utils.cuh"
#include "gemm.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>

template <typename DType, const int kM_, const int kN_, const int kK_,
          const int kTM_, const int kTN_, const int kTK_,
          const int kNumStages_ = 1, const int kNumTmaMulticast_ = 1>
struct GemmTraits {
  static constexpr int kM = kM_;
  static constexpr int kN = kN_;
  static constexpr int kK = kK_;

  static constexpr int kTM = kTM_;
  static constexpr int kTN = kTN_;
  static constexpr int kTK = kTK_;

  static constexpr int kNumTmaMulticast = kNumTmaMulticast_;

  static constexpr int kNumStages = kNumStages_;

  // the size of shared memory for each operand and result
  static constexpr int kSizeA = kTM * kTK * sizeof(DType);
  static constexpr int kSizeB = kTN * kTK * sizeof(DType);
  static constexpr int kSizeC = kTM * kTN * sizeof(DType);

  static_assert(kSizeC % 1024 == 0,
                "Shared memory of output tensor must be aligned to 1024 bytes");
  static_assert(kSizeA % 1024 == 0,
                "Shared memory of operand A must be aligned to 1024 bytes");
  static_assert(kSizeB % 1024 == 0,
                "Shared memory of operand B must be aligned to 1024 bytes");

  static constexpr int kSharedDataSize =
      (kSizeA + kSizeB) * kNumStages + kSizeC;
  static constexpr int kSharedMemSize =  // data + barriers
      kSharedDataSize + kNumStages * sizeof(uint64_t) * 2;

  static constexpr int kExpectedTmaBytes = kSizeA + kSizeB;

  static constexpr uint32_t kKShapeAllStages = kNumStages * kTK;

  static_assert(kK % kKShapeAllStages == 0,
                "kK must be divisible by kKShapeAllStages");

  static constexpr uint32_t kKNumIterations = CeilDiv<kK, kKShapeAllStages>;

  static constexpr uint32_t kNumWarpGroup = 2;
  static constexpr uint32_t kThreads = 128 * kNumWarpGroup;
  // thread 0 ~ kMathThreads - 1: consumer
  // thread kMathThreads ~ kThreads - 1: producer
  static constexpr uint32_t kMathThreads = 128;

  // register reconfigurations
  static constexpr uint32_t kNumTMARegisters = 40;
  static constexpr uint32_t kNumMathRegisters = 232;

  // tile scheduler
  using Scheduler_ = Scheduler<kM, kN, kTM, kTN, kNumTmaMulticast>;
};

int main() {
  //// kernel parameters
  using DType = __nv_bfloat16;
  // using DType = __nv_fp8_e4m3;

  static constexpr uint64_t kM = 64;
  static constexpr uint64_t kN = 64;
  static constexpr uint64_t kK = 128;

  static constexpr uint64_t kTM = 64;
  static constexpr uint64_t kTN = 64;
  static constexpr uint64_t kTK = 64;

  static constexpr int kNumStages = 1;
  using Traits = GemmTraits<DType, kM, kN, kK, kTM, kTN, kTK, kNumStages>;

  /// create data
  thrust::host_vector<DType> h_a(kM * kK);
  thrust::host_vector<DType> h_b(kK * kN);
  thrust::host_vector<DType> h_c(kM * kN);

  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<DType>(i % 256);
    // h_a[i] = static_cast<DType>(rand_float());
  }
  thrust::device_vector<DType> d_a = h_a;

  // print_values<DType, kM, kK>(h_a.data(), 0, kM * kK);

  for (int i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<DType>(i % 256);
    // h_b[i] = static_cast<DType>(rand_float());
  }
  thrust::device_vector<DType> d_b = h_b;

  thrust::fill(h_c.begin(), h_c.end(), static_cast<DType>(0));
  thrust::device_vector<DType> d_c = h_c;
  CHECK_CUDA(cudaDeviceSynchronize());

  //// create TMA descriptors
  // operand A is laid out in row-major order
  TMADescriptor<DType> tma_desc_a;
  uint64_t global_dim_a[2] = {kK, kM};
  uint32_t shared_dim_a[2] = {kTK, kTM};
  tma_desc_a.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_a.data()),  // Global address
      global_dim_a,                          // Global dimensions
      shared_dim_a,  // Shared memory dimensions (box dimensions)
      kK             // Global stride in bytes
  );

  // operand B is laid out in column-major order
  TMADescriptor<DType> tma_desc_b;
  uint64_t global_dim_b[2] = {kK, kN};
  uint32_t shared_dim_b[2] = {kTK, kTN};
  tma_desc_b.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_b.data()),  // Global address
      global_dim_b,                          // Global dimensions
      shared_dim_b,  // Shared memory dimensions (box dimensions)
      kK             // Global stride in bytes
  );

  // operand C is laid out in row-major order
  TMADescriptor<DType> tma_desc_c;
  uint64_t global_dim_c[2] = {kN, kM};
  uint32_t shared_dim_c[2] = {kTN, kTM};
  tma_desc_c.create_tma_2d_desc(
      thrust::raw_pointer_cast(d_c.data()),  // Global address
      global_dim_c,                          // Global dimensions
      shared_dim_c,  // Shared memory dimensions (box dimensions)
      kN             // Global stride in bytes
  );

  //// launch kernel
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  uint32_t num_sms = deviceProp.multiProcessorCount;

  dim3 blocks(num_sms, 1);
  dim3 threads(Traits::kThreads, 1, 1);

  std::cout << "num_sms: " << num_sms << std::endl;
  std::cout << "threads: " << threads.x << std::endl;
  std::cout << "shared memory size: " << Traits::kSharedMemSize << std::endl;
  std::cout << "shared memory per block: " << deviceProp.sharedMemPerBlock
            << std::endl
            << std::endl;

  auto kernel = &hopper_gemm<DType, Traits>;

  if (Traits::kSharedMemSize > GetMaxSharedMemoryPerBlock()) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Traits::kSharedMemSize));
  }

  kernel<<<blocks, threads, Traits::kSharedMemSize>>>(
      tma_desc_a.get_tma_desc(), tma_desc_b.get_tma_desc(),
      tma_desc_c.get_tma_desc());

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  return 0;
}
