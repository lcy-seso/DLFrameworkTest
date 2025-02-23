#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

template <typename DType, typename DataLayout, typename TmaLoad,
          const int kShmSize>
__global__ void ke_tma_load(const DType* data,  //
                            DataLayout g_layout,
                            CUTE_GRID_CONSTANT const TmaLoad tma_load) {
  __shared__ DType smem[kShmSize];
  __shared__ uint64_t tma_load_mbar;

  // in this simple example global memory and shared memory have the same layout
  auto g_tensor = make_tensor(make_gmem_ptr(data), g_layout);
  auto s_tensor = make_tensor(make_smem_ptr(smem), g_layout);

  auto warp_idx = cutlass::canonical_warp_idx_sync();
  auto lane_predicate = elect_one_sync();
  if (warp_idx == 0 && lane_predicate) {
    initialize_barrier(tma_load_mbar, 1 /* arrival count */);
    set_barrier_transaction_bytes(tma_load_mbar, kShmSize * sizeof(DType));

    auto g_tensor_coord = tma_load.get_tma_tensor(shape(g_layout));
    auto g_tensor_coord_cta =
        local_tile(g_tensor_coord, shape(g_layout),
                   make_coord(0, 0) /*we use 1 block in this example*/);

    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(g_tensor_coord_cta),
         tma_load_per_cta.partition_D(s_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, 0);
}

int main() {
  using DType = float;

  constexpr int kM = 64;
  constexpr int kN = 128;
  constexpr int kNumel = kM * kN;

  thrust::host_vector<DType> h_data(kNumel);
  for (int i = 0; i < kNumel; ++i) {
    h_data[i] = static_cast<DType>(i);
  }
  thrust::device_vector<DType> d_data = h_data;
  cudaDeviceSynchronize();
  const DType* d_data_ptr = d_data.data().get();

  using DataLayout = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;
  DataLayout layout;

  auto g_tensor = make_tensor(
      make_gmem_ptr(reinterpret_cast<const DType*>(d_data_ptr)), layout);
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, g_tensor, layout);

  dim3 blocks(1, 1, 1);
  dim3 threads(1, 1, 1);
  auto kernel = &ke_tma_load<DType, DataLayout, decltype(tma_load), kM * kN>;

  kernel<<<blocks, threads>>>(d_data_ptr, layout, tma_load);
  CUTE_CHECK_LAST();

  return 0;
}
