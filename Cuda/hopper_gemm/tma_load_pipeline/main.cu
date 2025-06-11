#include "cuda_utils.cuh"
#include "tma_utils.cuh"

#include <cuda.h>

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

template <typename DType, int kNumStages, int kTileM, int kTileN>
__global__ void tma_load_pipeline(
    const __grid_constant__ CUtensorMap tma_load_desc,
    const __grid_constant__ CUtensorMap tma_store_desc) {
  constexpr uint32_t kSmemSize = sizeof(DType) * kTileM * kTileN;
  static_assert(kSmemSize % 1024 == 0,
                "SMEM size must be multiples of 1024 bytes");
  extern __shared__ __align__(1024) unsigned char buf[];

  constexpr uint32_t kMathThreads = 128;  // two warp groups

  DType* smems[kNumStages];
  uint64_t* full_barriers[kNumStages];
  uint64_t* empty_barriers[kNumStages];

  auto barrier_start_ptr =
      reinterpret_cast<uint64_t*>(buf + kSmemSize * kNumStages);

#pragma unroll
  for (uint32_t i = 0; i < kNumStages; ++i) {
    smems[i] = reinterpret_cast<DType*>(buf + i * kSmemSize);
    full_barriers[i] = barrier_start_ptr + i;
    empty_barriers[i] = barrier_start_ptr + i;
  }

  const uint32_t warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  const uint32_t in_group_idx = threadIdx.x % 128;
  const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane_idx = get_lane_id();

  if (threadIdx.x == kMathThreads) {
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      // TMA thread signals `full_barriers`. since there is only one thread
      // issue TMA data transfer, the arrive count for `full_barriers` is
      // expected to be 1.
      init_barrier(full_barriers[i], 1);
      // one warp are used, the arrive count for `empty_barriers` is expected
      // to be the number of warps in use for computation.
      init_barrier(empty_barriers[i], kMathThreads / 32);
    }
  }
  // Synchronize all threads to make barrier visible in normal memory model
  __syncthreads();

  if (threadIdx.x >= kMathThreads) {  // producer, TMA copy
    warpgroup_reg_alloc<40>();
    if (threadIdx.x == kMathThreads) {  // issue TMA copy by the leader thread
    }
  } else {  // consumer
    warpgroup_reg_alloc<232>();
  }
}

int main() {
  using DType = float;
  // using DType = __nv_bfloat16;
  // using DType = __nv_fp8_e4m3;

  static constexpr uint64_t kM = 128;  // Height (2 tiles high)
  static constexpr uint64_t kN = 128;  // Width (2 tiles wide)
  static constexpr uint64_t kNumStages = 1;

  static constexpr uint64_t kTileM = 64;  // Tile height
  static constexpr uint64_t kTileN = 64;  // Tile width

  static constexpr uint64_t kNumel = kM * kN;
  static constexpr size_t kBytes = kNumel * sizeof(DType);

  DType *h_src = nullptr, *h_dst = nullptr;
  h_src = (DType*)malloc(kBytes);
  h_dst = (DType*)malloc(kBytes);

  if (h_src == nullptr || h_dst == nullptr) {
    fprintf(stderr, "Failed to allocate host memory\n");
    if (h_src) free(h_src);
    if (h_dst) free(h_dst);
    return 1;
  }

  for (int i = 0; i < kNumel; ++i) {
    h_src[i] = static_cast<DType>(rand_float());
    // h_src[i] = static_cast<DType>(i % 2048);
    h_dst[i] = static_cast<DType>(0.);
  }

#if 0
  std::cout << "Source tensor:" << std::endl;
  print_values<DType, kM, kN>(h_src, 0, 256);
  std::cout << std::endl;
#endif

  DType *d_src = nullptr, *d_dst = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_src, kBytes));
  CHECK_CUDA(cudaMalloc((void**)&d_dst, kBytes));

  CHECK_CUDA(cudaMemcpy(d_src, h_src, kBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dst, h_dst, kBytes, cudaMemcpyHostToDevice));

  TMADescriptor<DType> tma_desc_load;
  TMADescriptor<DType> tma_desc_store;
  uint64_t global_dim[2] = {kM, kN};
  uint32_t shared_dim[2] = {kTileM, kTileN};

  tma_desc_load.create_tma_2d_desc(
      d_src,                      // Global address
      global_dim,                 // Global dimensions
      shared_dim,                 // Shared memory dimensions (box dimensions)
      kN,                         // Global stride in bytes
      CU_TENSOR_MAP_SWIZZLE_NONE  // Swizzle mode
  );

  tma_desc_store.create_tma_2d_desc(
      d_dst,                      // Global address
      global_dim,                 // Global dimensions
      shared_dim,                 // Shared memory dimensions (box dimensions)
      kN,                         // Global stride in bytes
      CU_TENSOR_MAP_SWIZZLE_NONE  // Swizzle mode
  );

  int num_tiles_x = CeilDiv<kM, kTileM>;
  int num_tiles_y = CeilDiv<kN, kTileN>;
  static constexpr int kThreads = 256;

  dim3 blocks(num_tiles_x, num_tiles_y, 1);
  dim3 threads(kThreads, 1, 1);

  int smem_size = kTileM * kTileN * kNumStages * sizeof(DType) +
                  kNumStages * sizeof(uint64_t) * 2;

  std::cout << "Kernel config: blocks(" << blocks.x << "," << blocks.y << ","
            << blocks.z << "), threads(" << threads.x << "," << threads.y << ","
            << threads.z << "), smem_size=" << smem_size << std::endl;

  auto kernel = &tma_load_pipeline<DType, kNumStages, kTileM, kTileN>;
  CHECK_CUDA(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kernel<<<blocks, threads, smem_size>>>(tma_desc_load.get_tma_desc(),
                                         tma_desc_store.get_tma_desc());

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_dst, d_dst, kBytes, cudaMemcpyDeviceToHost));

#if 0
  std::cout << std::endl
            << "Destination tensor (first 256 elements):" << std::endl;
  print_values<DType, kM, kN>(h_dst, 0, 256);
#endif

  check_results(h_src, h_dst, kNumel);

  // Cleanup
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  free(h_src);
  free(h_dst);

  std::cout << "TMA example completed successfully!" << std::endl;
  return 0;
}
