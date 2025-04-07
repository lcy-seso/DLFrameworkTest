#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
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
    printf("shared memory: %d\n", kSharedMemSize);

    for (int i = 0; i < kSharedMemSize; ++i) {
      printf("%.2f, ", smem[i]);

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

template <class DType, const int kSharedMemSize_>
struct SharedStorageImpl {
  array_aligned<DType, kSharedMemSize_, 128> buf;  // 128-bits aligned

  // alignas(16) uint64_t tma_load_mbar[1];
  cutlass::arch::ClusterTransactionBarrier mbarrier;

  static constexpr int kSharedMemSize = kSharedMemSize_;
};

template <typename DType,                                //
          typename SharedStorage,                        //
          typename ClusterShape, typename ThreadLayout,  //
          typename GMemLayout, typename GMemLayoutOut,   //
          typename SMemLayout,                           //
          typename TmaLoad, typename TmaStore>
__global__ void ke_tma_copy_multicast(
    const DType* src, DType* dst,  //
    const GMemLayout& g_layout, const GMemLayoutOut& g_dst_layout,
    const SMemLayout& s_layout,  //
    CUTE_GRID_CONSTANT const TmaLoad tma_load,
    CUTE_GRID_CONSTANT const TmaStore tma_store) {
  // constants
  static constexpr int kSharedMemSize = SharedStorage::kSharedMemSize;
  static constexpr int kTmaTransactionBytes = kSharedMemSize * sizeof(DType);

  // shared memory buffer
  extern __shared__ unsigned char buf_[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(buf_);
  auto& mbarrier = smem.mbarrier;
  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;

  // intermediate tensor on shared memory
  auto s_tensor = make_tensor(make_smem_ptr(smem.buf.data()), s_layout);

  // thread block cluster related constants
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_size = size(ClusterShape{});
  uint16_t tma_mcast_mask = ((uint16_t(1) << cluster_size) - 1);

  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const bool lane_predicate = elect_one_sync();
  if (warp_idx == 0 && lane_predicate) {
    // get the tma descriptor for load and store
    prefetch_tma_descriptor(tma_load.get_tma_descriptor());
    prefetch_tma_descriptor(tma_store.get_tma_descriptor());
  }

  auto blk_coord = make_coord(blockIdx.x, blockIdx.y);
  auto tensor_coord = tma_load.get_tma_tensor(shape(g_layout));
  auto gS = local_tile(tensor_coord, shape(s_layout), blk_coord);

  auto cta_tma_src = tma_load.get_slice(block_rank_in_cluster);
  auto tSgSX = cta_tma_src.partition_S(gS);
  auto tSgS = group_modes<1, rank(tSgSX)>(tSgSX);

  auto tSsSX = cta_tma_src.partition_D(s_tensor);
  auto tSsS = group_modes<1, rank(tSsSX)>(tSsSX);

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.init(1 /* arrive count */);
  }
  __syncthreads();
  cluster_sync();
  cutlass::arch::fence_barrier_init();

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);

    copy(
        tma_load.with(reinterpret_cast<BarrierType&>(mbarrier), tma_mcast_mask),
        tSgS(_, 0), tSsS(_, 0));
  }
  __syncthreads();

  mbarrier.wait(0 /* phase */);

  cutlass::arch::fence_view_async_shared();

#if 0
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("shared memory: %d\n", kSharedMemSize);

    DType* smem_data = smem.buf.data();

    for (int i = 0; i < kSharedMemSize; ++i) {
      printf("%.3f, ", smem_data[i]);

      if (i && (i + 1) % 16 == 0) printf("\n");
    }
  }
#endif

  {
    auto blk_coord_out = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
    auto tensor_coord = tma_store.get_tma_tensor(shape(g_dst_layout));
    auto gD = local_tile(tensor_coord, shape(s_layout), blk_coord_out);

    auto tma_store_per_cta = tma_store.get_slice(Int<0>{});
    auto src_thrd = tma_store_per_cta.partition_S(s_tensor);
    auto dst_thrd = tma_store_per_cta.partition_D(gD);

    if (warp_idx == 0 and lane_predicate) {
      copy(tma_store, src_thrd, dst_thrd);
      tma_store_arrive();
    }
    tma_store_wait<0>();
    cluster_sync();
  }
}
