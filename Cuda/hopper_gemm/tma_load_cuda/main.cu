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

template <typename DType, int kTileM, int kTileN>
__global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap tma_load_desc,
    const __grid_constant__ CUtensorMap tma_store_desc) {
  extern __shared__ __align__(128) unsigned char buf_[];
  auto* smem = reinterpret_cast<DType*>(buf_);
  __shared__ uint64_t mbarrier;

  if (threadIdx.x == 0) {
    // 0. Prefetch TMA descriptors
    prefetch_tma_descriptor(&tma_load_desc);
    prefetch_tma_descriptor(&tma_store_desc);

    // 1. Initialize barrier
    init_barrier(&mbarrier, 1);

    // 2. Arrive and expect transaction size (tile_size * sizeof(DType))
    uint32_t expected_bytes = kTileM * kTileN * sizeof(DType);
    // **thread 0**  arrives at the mbarrier.
    arrive_and_expect_tx(&mbarrier, expected_bytes);

    // N dimension (width) coordinate [0, 1]
    int coord_0 = blockIdx.x * kTileM;
    // M dimension (height) coordinate [0, 1]
    int coord_1 = blockIdx.y * kTileN;

    // 3. Perform TMA load from global memory to shared memory
    tma_load(&tma_load_desc, &mbarrier, smem, coord_0, coord_1);
  }

  // 4. Wait for TMA load completion - ALL threads must wait for the barrier
  __syncthreads();

  // `try_wait` is a blocking operation, and **ALL threads** wait for TMA
  // barrier completion.
  // The write to SMEM done by the TMA load is made visible to all threads that
  // invoked the mbarrier wait.
  // the thread sleeps until that phase bit of the mbarrier flips.
  wait(&mbarrier, 0);

  // TMA store uses a memory fence to enforce memory consistency
  tma_store_fence();  // Fence before TMA store

  // 5. TMA Store operations
  if (threadIdx.x == 0) {
    int coord_0 = blockIdx.x * kTileM;  // N dimension coordinate
    int coord_1 = blockIdx.y * kTileN;  // M dimension coordinate

    // Perform TMA store from shared memory to global memory
    tma_store(&tma_store_desc, smem, coord_0, coord_1);

    // Commit the TMA store operation
    tma_store_arrive();
  }

  // 6. Wait for all TMA store operations to complete
  tma_store_wait_group<0>();
}

int main() {
  using DType = float;
  // using DType = __nv_fp8_e4m3;

  static constexpr uint64_t kM = 128;     // Height (2 tiles high)
  static constexpr uint64_t kN = 128;     // Width (2 tiles wide)
                                          //
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

  dim3 blocks(num_tiles_x, num_tiles_y, 1);
  dim3 threads(128, 1, 1);

  int smem_size = kTileM * kTileN * sizeof(DType);

  std::cout << "Kernel config: blocks(" << blocks.x << "," << blocks.y << ","
            << blocks.z << "), threads(" << threads.x << "," << threads.y << ","
            << threads.z << "), smem_size=" << smem_size << std::endl;

  auto kernel = &tma_copy_kernel<DType, kTileM, kTileN>;
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
