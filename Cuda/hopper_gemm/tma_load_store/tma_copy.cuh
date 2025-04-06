#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

using namespace cute;

template <typename DType, const int kSharedMemSize,                         //
          typename ThreadLayout, typename GMemLayout, typename SMemLayout,  //
          typename TmaLoad, typename TmaStore>
__global__ void ke_tma_copy(const DType* src, DType* dst,  //
                            const GMemLayout& g_layout,
                            const SMemLayout& s_layout,
                            CUTE_GRID_CONSTANT const TmaLoad tma_load,
                            CUTE_GRID_CONSTANT const TmaStore tma_store) {
  __shared__ DType smem[kSharedMemSize];
  auto s_tensor = make_tensor(make_smem_ptr(smem), s_layout);

  // asynchronous transaction barrier lives in shared memory
  __shared__ uint64_t tma_load_mbar;

  auto blk_coord = make_coord(blockIdx.x, blockIdx.y);

  auto warp_idx = cutlass::canonical_warp_idx_sync();
  auto lane_predicate = elect_one_sync();

  if (warp_idx == 0 && lane_predicate) {
    // get the tma descriptor
    prefetch_tma_descriptor(tma_load.get_tma_descriptor());
    // 1 thread issues tma load
    initialize_barrier(tma_load_mbar, 1 /* arrival count */);
    set_barrier_transaction_bytes(tma_load_mbar,
                                  kSharedMemSize * sizeof(DType));

    // source tensor on global memory
    auto g_tensor = make_tensor(make_gmem_ptr(src), g_layout);
    auto tensor_coord = tma_load.get_tma_tensor(shape(g_tensor));
    auto gS = local_tile(tensor_coord, shape(s_layout), blk_coord);

    // get partition for the current thread
    auto tma_load_per_cta = tma_load.get_slice(Int<0>{});
    auto src_thrd = tma_load_per_cta.partition_S(gS);
    auto dst_thrd = tma_load_per_cta.partition_D(s_tensor);

    copy(tma_load.with(tma_load_mbar), src_thrd, dst_thrd);
  }
  __syncthreads();

  // the write to SMEM done by the TMA load is made visible to all threads that
  // invoked the mbarrier wait
  wait_barrier(tma_load_mbar, 0);

#if 0
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("sharedd memory: %d\n", kSharedMemSize);

    for (int i = 0; i < kSharedMemSize; ++i) {
      printf("%.0f, ", smem[i]);

      if (i && (i + 1) % 16 == 0) printf("\n");
    }
  }
#endif

  /// ==== TMA store ==== ///
  tma_store_fence();

  {  // store to destination tensor
    prefetch_tma_descriptor(tma_store.get_tma_descriptor());

    ThreadLayout thread_layout;

    // source tensor on shared memory
    auto sS = local_partition(s_tensor, thread_layout, threadIdx.x);

    auto d_tensor = make_tensor(make_gmem_ptr(dst), g_layout);
    auto tensor_coord = tma_store.get_tma_tensor(shape(d_tensor));
    auto gD = local_tile(tensor_coord, shape(s_tensor), blk_coord);

    // get partition for the current thread
    auto tma_store_per_cta = tma_store.get_slice(Int<0>{});
    auto src_thrd = tma_store_per_cta.partition_S(sS);
    auto dst_thrd = tma_store_per_cta.partition_D(gD);

    if (warp_idx == 0 and lane_predicate) {
      copy(tma_store, src_thrd, dst_thrd);
      tma_store_arrive();  // commits the TMA store operation
    }

    // waits until at most `count` many of the committed TMA store operations
    // are pending. 0 means wait for all being completed.
    tma_store_wait<0>();
  }
}

template <typename DType, const int kSharedMemSize,                         //
          typename ThreadLayout, typename GMemLayout, typename SMemLayout,  //
          typename TmaLoad, typename TmaStore>
__global__ void ke_tma_copy_multicast(
    const DType* src, DType* dst,  //
    const GMemLayout& g_layout, const SMemLayout& s_layout,
    CUTE_GRID_CONSTANT const TmaLoad tma_load,
    CUTE_GRID_CONSTANT const TmaStore tma_store) {
  __shared__ DType smem[kSharedMemSize];
  auto s_tensor = make_tensor(make_smem_ptr(smem), s_layout);

  // asynchronous transaction barrier lives in shared memory
  __shared__ uint64_t tma_load_mbar;

  auto blk_coord = make_coord(blockIdx.x, blockIdx.y);

  auto warp_idx = cutlass::canonical_warp_idx_sync();
  auto lane_predicate = elect_one_sync();

  if (warp_idx == 0 && lane_predicate) {
    // get the tma descriptor
    prefetch_tma_descriptor(tma_load.get_tma_descriptor());
    // 1 thread issues tma load
    initialize_barrier(tma_load_mbar, 1 /* arrival count */);
    set_barrier_transaction_bytes(tma_load_mbar,
                                  kSharedMemSize * sizeof(DType));

    // source tensor on global memory
    auto g_tensor = make_tensor(make_gmem_ptr(src), g_layout);
    auto tensor_coord = tma_load.get_tma_tensor(shape(g_tensor));
    auto gS = local_tile(tensor_coord, shape(s_layout), blk_coord);

    // get partition for the current thread
    auto tma_load_per_cta = tma_load.get_slice(Int<0>{});
    auto src_thrd = tma_load_per_cta.partition_S(gS);
    auto dst_thrd = tma_load_per_cta.partition_D(s_tensor);

    copy(tma_load.with(tma_load_mbar), src_thrd, dst_thrd);
  }
  __syncthreads();

  // the write to SMEM done by the TMA load is made visible to all threads that
  // invoked the mbarrier wait
  wait_barrier(tma_load_mbar, 0);

#if 0
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("sharedd memory: %d\n", kSharedMemSize);

    for (int i = 0; i < kSharedMemSize; ++i) {
      printf("%.0f, ", smem[i]);

      if (i && (i + 1) % 16 == 0) printf("\n");
    }
  }
#endif

  /// ==== TMA store ==== ///
  tma_store_fence();

  {  // store to destination tensor
    prefetch_tma_descriptor(tma_store.get_tma_descriptor());

    ThreadLayout thread_layout;

    // source tensor on shared memory
    auto sS = local_partition(s_tensor, thread_layout, threadIdx.x);

    auto d_tensor = make_tensor(make_gmem_ptr(dst), g_layout);
    auto tensor_coord = tma_store.get_tma_tensor(shape(d_tensor));
    auto gD = local_tile(tensor_coord, shape(s_tensor), blk_coord);

    // get partition for the current thread
    auto tma_store_per_cta = tma_store.get_slice(Int<0>{});
    auto src_thrd = tma_store_per_cta.partition_S(sS);
    auto dst_thrd = tma_store_per_cta.partition_D(gD);

    if (warp_idx == 0 and lane_predicate) {
      copy(tma_store, src_thrd, dst_thrd);
      tma_store_arrive();  // commits the TMA store operation
    }

    // waits until at most `count` many of the committed TMA store operations
    // are pending. 0 means wait for all being completed.
    tma_store_wait<0>();
  }
}
