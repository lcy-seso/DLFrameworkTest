#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdio>

using namespace cute;

template <class T, class TensorX, class GmemLayout, class SmemLayout,
          class TmaLoad>
__global__ static void tma_kernel(TensorX tX, GmemLayout gmem_layout,
                                  SmemLayout smem_layout,
                                  CUTE_GRID_CONSTANT const TmaLoad tma_load) {
  __shared__ T smem[cosize_v<SmemLayout>];
  __shared__ uint64_t tma_load_mbar[1];

  auto sX = make_tensor(make_smem_ptr(smem), smem_layout);

  auto mX = tma_load.get_tma_tensor(shape(gmem_layout));
  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  auto gX = local_tile(mX, shape(smem_layout), make_coord(0, 0));

  auto cta_tma_load = tma_load.get_slice(0);

  auto tXgX = cta_tma_load.partition_S(gX);  // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  auto tXsX = cta_tma_load.partition_D(sX);  // (TMA,TMA_M,TMA_N)

  auto warp_idx = cutlass::canonical_warp_idx_sync();
  auto lane_predicate = cute::elect_one_sync();

  if (warp_idx == 0 && lane_predicate) {
    constexpr int k_tma_transaction_bytes = size(sX) * sizeof_bits_v<T> / 8;

    tma_load_mbar[0] = 0;
    cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
    cute::set_barrier_transaction_bytes(tma_load_mbar[0],
                                        k_tma_transaction_bytes);

    cute::copy(tma_load.with(tma_load_mbar[0]), tXgX, tXsX);
  }
  __syncthreads();
  constexpr int k_phase_bit = 0;
  cute::wait_barrier(tma_load_mbar[0], k_phase_bit);
}

int main() {
  using T = float;

  constexpr int m = 4;
  constexpr int n = 4;

  thrust::host_vector<T> h_data(m * n);
  for (int i = 0; i < m * n; ++i) h_data[i] = static_cast<T>(i);
  thrust::device_vector<T> d_data = h_data;
  cudaDeviceSynchronize();

  // create tensors
  auto gmem_layout = Layout<Shape<Int<n>, Int<m>>>{};
  auto smem_layout = Layout<Shape<Int<n>, Int<m>>>{};

  auto gX = make_tensor(
      make_gmem_ptr(reinterpret_cast<const T*>(d_data.data().get())),
      gmem_layout);

  // create the TMA object
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gX, smem_layout);

  // launch the kernel
  dim3 blk_dim{1, 1, 1};
  dim3 grd_dim{1, 1, 1};
  tma_kernel<T, decltype(gX), decltype(gmem_layout), decltype(smem_layout),
             decltype(tma_load)>
      <<<grd_dim, blk_dim>>>(gX, gmem_layout, smem_layout, tma_load);
  CUTE_CHECK_LAST();

  return 0;
}
