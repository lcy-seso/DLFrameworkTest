#include "cuda_utils.cuh"

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

template <const int kM, const int kN, const int kTM, const int kTN,
          typename DType, typename GMemLayout, typename SMemLayout,
          typename TmaLoad>
__global__ void ke_tma_load(const DType* data,  //
                            GMemLayout g_layout, SMemLayout s_layout,
                            CUTE_GRID_CONSTANT const TmaLoad tma_load) {
  int offset = blockIdx.x * kN + blockIdx.y * kTM;

  __shared__ DType smem[kTM * kTN];

  // asynchronous transaction barrier lives in shared memory
  __shared__ uint64_t tma_load_mbar;

  auto g_tensor = make_tensor(make_gmem_ptr(data + offset), g_layout);
  auto s_tensor = make_tensor(make_smem_ptr(smem), s_layout);

  auto warp_idx = cutlass::canonical_warp_idx_sync();
  auto lane_predicate = elect_one_sync();

  if (warp_idx == 0 && lane_predicate) {
    initialize_barrier(tma_load_mbar,
                       1 /* arrival count */);  // 1 thread issues tma load
    set_barrier_transaction_bytes(tma_load_mbar, kTM * kTN * sizeof(DType));

    auto g_tensor_coord = tma_load.get_tma_tensor(shape(g_layout));
    auto g_tensor_coord_cta =
        local_tile(g_tensor_coord, shape(g_layout), make_coord(0, 0));

    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(g_tensor_coord_cta),
         tma_load_per_cta.partition_D(s_tensor));
  }
  __syncthreads();
  // the write to SMEM done by the TMA load is made visible to all threads that
  // invoked the mbarrier wait
  wait_barrier(tma_load_mbar, 0);

  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int i = 0; i < kTM * kTN; ++i) {
      printf("%.0f, ", smem[i]);

      if (i && (i + 1) % 16 == 0) printf("\n");
    }
  }
}

int main() {
  using DType = float;
  static constexpr int kM = 64 * 128;
  static constexpr int kN = 128 * 128;
  static constexpr int kNumel = kM * kN;

  static constexpr int kTM = 64;
  static constexpr int kTN = 128;

  thrust::host_vector<DType> h_data(kNumel);
  for (int i = 0; i < kNumel; ++i) {
    h_data[i] = static_cast<DType>(i);
  }
  thrust::device_vector<DType> d_data = h_data;
  cudaDeviceSynchronize();
  const DType* d_data_ptr = d_data.data().get();

  using GMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  GMemLayout g_layout;
  auto g_tensor = make_tensor(
      make_gmem_ptr(reinterpret_cast<const DType*>(d_data_ptr)), g_layout);

  using SMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  SMemLayout s_layout;

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, g_tensor, s_layout);

  int block_x = CeilDiv<kM, kTM>;
  int block_y = CeilDiv<kN, kTN>;

  dim3 blocks(block_x, block_y, 1);
  dim3 threads(256, 1, 1);

  auto kernel = &ke_tma_load<kM, kN, kTM, kTN, DType, GMemLayout, SMemLayout,
                             decltype(tma_load)>;

  kernel<<<blocks, threads>>>(d_data_ptr, g_layout, s_layout, tma_load);
  CUTE_CHECK_LAST();

  return 0;
}
