#include "cuda_utils.cuh"
#include "tma.cuh"

#include <cuda.h>
#include <cuda_fp8.h>
#include <cute/tensor.hpp>

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <typename DType, const int kM, const int kN>
void print_values(const DType* tensor, int start = 256, int cutoff = 128) {
  std::cout << std::fixed << std::setprecision(0);
  for (int i = start; i < kM * kN; ++i) {
    std::cout << static_cast<float>(tensor[i]) << ", ";
    if ((i + 1) % 16 == 0) std::cout << std::endl;

    if (i == (start + cutoff - 1)) break;
  }
}

template <typename DType, int kTileM, int kTileN>
__global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap tma_load_desc, DType* output) {
  // 128-byte alignment for TMA
  extern __shared__ __align__(128) unsigned char buf_[];
  auto* smem = reinterpret_cast<DType*>(buf_);
  __shared__ uint64_t mbarrier;

  if (threadIdx.x == 0) {
    // 0. Prefetch TMA descriptor
    prefetch_tma_descriptor(&tma_load_desc);

    // 1. Initialize barrier
    mbarrier_init(&mbarrier, 1);

    // 2. Arrive and expect transaction size (tile_size * sizeof(float))
    uint32_t expected_bytes = kTileM * kTileN * sizeof(DType);
    mbarrier_arrive_expect_tx(&mbarrier, expected_bytes);

    int coord_0 = blockIdx.x * kTileM;  // N dimension (width) coordinate [0, 1]
    int coord_1 =
        blockIdx.y * kTileN;  // M dimension (height) coordinate [0, 1]
    // 4. Perform TMA load from global memory to shared memory
    tma_copy(&tma_load_desc, &mbarrier, smem, coord_0, coord_1, 1);

    printf("coords=(%d,%d),TMA load complete\n", blockIdx.y, blockIdx.x);
  }

  // 4. Wait for TMA completion - ALL threads must wait for the barrier
  __syncthreads();

  // ALL threads wait for TMA barrier completion using standalone wait_barrier
  // function The write to SMEM done by the TMA load is made visible to all
  // threads that invoked the mbarrier wait
  // phase_bit = 0, all threads wait
  wait_barrier(mbarrier, 0);

  // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   for(int i = 0; i < kTileM * kTileN; ++i) {
  //     printf("%.0f, ", smem[i]);

  //     if ((i + 1) % 16 == 0) printf("\n");
  //   }
  //   printf("\n");
  // }

  int tid = threadIdx.x;
  int total_elements = kTileM * kTileN;

  for (int i = tid; i < total_elements; i += blockDim.x) {
    int row = i / kTileN;
    int col = i % kTileN;

    int global_row = blockIdx.y * kTileM + row;
    int global_col = blockIdx.x * kTileN + col;
    int global_idx = global_row * 256 + global_col;  // kN = 256

    if (global_row < 128 && global_col < 256) {  // kM = 128, kN = 256
      output[global_idx] = smem[i];
    }
  }
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
    // h_src[i] = static_cast<DType>(rand_float());
    h_src[i] = static_cast<DType>(i % 2048);
    h_dst[i] = static_cast<DType>(0.);
  }
  // std::cout << "Source tensor:" << std::endl;
  // print_values<DType, kM, kN>(h_src);
  // std::cout << std::endl;

  DType *d_src = nullptr, *d_dst = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_src, kBytes));
  CHECK_CUDA(cudaMalloc((void**)&d_dst, kBytes));

  CHECK_CUDA(cudaMemcpy(d_src, h_src, kBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dst, h_dst, kBytes, cudaMemcpyHostToDevice));

  CHECK_CU(cuInit(0));

  constexpr uint32_t rank = 2;
  CUtensorMap tensorMap;

  uint64_t global_dim[rank] = {kM, kN};
  uint64_t global_stride[rank - 1] = {kN * sizeof(DType)};
  uint32_t box_dim[rank] = {kTileM, kTileN};
  uint32_t element_stride[rank] = {1, 1};

  CHECK_CU(cuTensorMapEncodeTiled(
      &tensorMap,                          // Output tensor map
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,     // Data type
      rank,                                // Tensor rank (2D)
      d_src,                               // Global address
      global_dim,                          // Global dimensions
      global_stride,                       // Global strides (in bytes)
      box_dim,                             // Box dimensions
      element_stride,                      // Element strides
      CU_TENSOR_MAP_INTERLEAVE_NONE,       // Interleave layout
      CU_TENSOR_MAP_SWIZZLE_NONE,          // Swizzle mode
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // L2 promotion
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE    // Out-of-bounds fill
      ));

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

  kernel<<<blocks, threads, smem_size>>>(tensorMap, d_dst);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_dst, d_dst, kBytes, cudaMemcpyDeviceToHost));

  std::cout << std::endl << "Destination tensor:" << std::endl;
  print_values<DType, kM, kN>(h_dst, 0, kNumel);

  // check_results(h_src, h_dst, kNumel);

  // Cleanup
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  free(h_src);
  free(h_dst);

  std::cout << "TMA example completed successfully!" << std::endl;
  return 0;
}
