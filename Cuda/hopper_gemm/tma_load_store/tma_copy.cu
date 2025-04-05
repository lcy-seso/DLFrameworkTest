#include "cuda_utils.cuh"

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

template <const int kSharedMemSize, typename DType,                         //
          typename ThreadLayout, typename GMemLayout, typename SMemLayout,  //
          typename TmaLoad, typename TmaStore>
__global__ void ke_tma_copy(const DType* src, DType* dst,  //
                            const ThreadLayout& thread_layout,
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

int main() {
  using DType = float;

  static constexpr int kTM = 64;
  static constexpr int kTN = 128;

  static constexpr int kM = kTM * 128;
  static constexpr int kN = kTN * 256;

  static constexpr int kThreads = 256;

  static constexpr int kSharedMemSize = kTM * kTN;
  static constexpr int kNumel = kM * kN;

  thrust::host_vector<DType> h_src(kNumel);
  thrust::host_vector<DType> h_dst(kNumel);
  for (int i = 0; i < kNumel; ++i) {
    // h_src[i] = static_cast<DType>(i % 2048);
    h_src[i] = rand_float();
    h_dst[i] = static_cast<DType>(0);
  }

  thrust::device_vector<DType> d_src = h_src;
  thrust::device_vector<DType> d_dst = h_dst;
  cudaDeviceSynchronize();

  const DType* d_src_ptr = d_src.data().get();
  DType* d_dst_ptr = d_dst.data().get();

  // source tensor on global memory
  using GMemLayout = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;
  GMemLayout g_layout;
  auto g_src = make_tensor(
      make_gmem_ptr(reinterpret_cast<const DType*>(d_src_ptr)), g_layout);

  // intermediate tensor on shared memory
  using SMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  SMemLayout s_layout;

  // destination tensor on global memory
  auto g_dst = make_tensor(
      make_gmem_ptr(reinterpret_cast<const DType*>(d_dst_ptr)), g_layout);

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, g_src, s_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, g_dst, s_layout);

  int block_x = CeilDiv<kM, kTM>;
  int block_y = CeilDiv<kN, kTN>;

  dim3 blocks(block_x, block_y, 1);
  dim3 threads(kThreads, 1, 1);

  auto thread_layout = make_layout(Shape<_32, Int<CeilDiv<kThreads, 32>>>{});

  std::cout << "kSharedMemSize: " << kSharedMemSize << std::endl;

  auto kernel = &ke_tma_copy<kSharedMemSize, DType,  //
                             decltype(thread_layout), GMemLayout, SMemLayout,
                             decltype(tma_load), decltype(tma_store)>;

  kernel<<<blocks, threads>>>(d_src_ptr, d_dst_ptr, thread_layout, g_layout,
                              s_layout, tma_load, tma_store);
  cudaDeviceSynchronize();

  h_dst = d_dst;

  // Check if h_src and h_dst are the same
  bool match = true;
  for (int i = 0; i < kNumel; ++i) {
    if (h_src[i] != h_dst[i]) {
      std::cerr << "Verification failed: Mismatch found at index " << i
                << ": h_src[" << i << "] = " << h_src[i] << ", h_dst[" << i
                << "] = " << h_dst[i] << std::endl;
      match = false;
      break;  // Stop at the first mismatch
    }
  }

  if (match) {
    std::cout
        << "Verification successful: h_src and h_dst contain the same values."
        << std::endl;
  } else {
    std::cerr << "Verification failed: h_src and h_dst differ." << std::endl;
    return 1;
  }

#if 0
  int start = 256;
  int end = start + 64;
  printf("\nsrc:\n");
  for (int i = start; i < end; ++i) {
    printf("%.3f, ", h_src[i]);
    if (i && (i + 1) % 16 == 0) printf("\n");
  }

  printf("\n\ndst:\n");
  for (int i = start; i < end; ++i) {
    printf("%.3f, ", h_dst[i]);
    if (i && (i + 1) % 16 == 0) printf("\n");
  }
#endif

  return 0;
}
