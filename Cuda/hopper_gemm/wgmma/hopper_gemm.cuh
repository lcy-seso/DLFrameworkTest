#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <int N>
__device__ __forceinline__ void wait_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void commit_copy_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
  cute::cp_async_fence();
#endif
}

__device__ __forceinline__ void __copy_async() {
  commit_copy_group();
  wait_group<0>();
}

template <typename DType, typename KeTraits, typename CopyA, typename CopyB>
__global__ void ke_cute_wgmma(const CopyA& copy_a, const CopyB& copy_b,
                              const DType* gA_ptr, const DType* gB_ptr,
                              DType* gC_ptr) {
  using SharedStorage = typename KeTraits::SharedStorage;
  extern __shared__ char smem_[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

  typename KeTraits::TiledMma mma;

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));  // num threads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));  // num threads

  // global tensors
  auto TensorA =
      make_tensor(make_gmem_ptr(gA_ptr), typename KeTraits::LayoutGmemA{});
  auto TensorB =
      make_tensor(make_gmem_ptr(gB_ptr), typename KeTraits::LayoutGmemB{});
  auto TensorC =
      make_tensor(make_gmem_ptr(gC_ptr), typename KeTraits::LayoutGmemC{});

  auto cta_tiler = typename KeTraits::CtaTiler{};
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);

  // local tiles
  auto gA = local_tile(TensorA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  auto gB = local_tile(TensorB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  auto gC = local_tile(TensorC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  // shared memory tiles
  using SmemLayoutA = typename KeTraits::SmemLayoutA;
  using SmemLayoutB = typename KeTraits::SmemLayoutB;

  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);

  Tensor sB =
      make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);

  auto loader_a = copy_a.get_thread_slice(threadIdx.x);
  Tensor tAgA = loader_a.partition_S(gA);
  Tensor tAsA = loader_a.partition_D(sA_);

  auto loader_b = copy_b.get_thread_slice(threadIdx.x);
  Tensor tBgB = loader_b.partition_S(gB);
  Tensor tBsB = loader_b.partition_D(sB_);

  // Total number of k-tiles
  auto K_TILE_MAX = size<3>(tAgA);
  // Number of pipelined k-tiles in smem
  auto K_PIPE_MAX = size<3>(tAsA);

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // reg tile A

  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // reg tile B

  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // reg tile C

  // prefetch all but the last k-tile
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX - 1; ++k) {
    copy(copy_a, tAgA(_, _, _, k), tAsA(_, _, _, k));
    copy(copy_b, tBgB(_, _, _, k), tBsB(_, _, _, k));

    commit_copy_group();
  }
  clear(tCrC);
  __syncthreads();

  int k_pipe_read = 0;                // current pipe to read from
  int k_pipe_write = K_PIPE_MAX - 1;  // current pipe to write to

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {  // pipelined main loop
    int k_tile_next = k_tile + (K_PIPE_MAX - 1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;

    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe_write));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe_write));
    cp_async_fence();

    ++k_pipe_write;  // advance k_pipe_write
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    // compute on k_tile
    cp_async_wait<0>();  // wait on all cp.async -- optimize by pipelining to
                         // overlap GMEM reads

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();

    cute::gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read),
               tCrC);  // (V,M,K) x (V,N,K) => (V,M,N)
    warpgroup_commit_batch();
    // wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to
    // ensure smem_pipe_write is consumed
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    ++k_pipe_read;  // advance k_pipe_read
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  DType alpha = static_cast<DType>(1.0);
  DType beta = static_cast<DType>(0.0);
  axpby(alpha, tCrC, beta, tCgC);
}
