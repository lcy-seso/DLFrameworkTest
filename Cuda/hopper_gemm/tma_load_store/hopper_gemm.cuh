#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

using namespace cute;

template <typename ParamsT, typename Kernel_traits, typename TmaLoadA,
          typename TmaLoadB>
__global__ void ke_cute_hopper_gemm(
    ParamsT params, CUTE_GRID_CONSTANT TmaLoadA const tma_load_A,
    CUTE_GRID_CONSTANT TmaLoadB const tma_load_B) {
  using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
  using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kBlockK = Kernel_traits::kBlockK;

  Tensor mA = tma_load_A.get_tma_tensor(make_shape(params.M, params.K));
  Tensor mB = tma_load_B.get_tma_tensor(make_shape(params.N, params.K));
  Tensor mC =
      make_tensor(make_gmem_ptr(params.C), make_shape(params.M, params.N),
                  make_stride(_1{}, params.M));

  auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  using SharedStorage = typename Kernel_traits::SharedStorage;
  extern __shared__ char smem_[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
  Tensor sB =
      make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

  auto [tAgA, tAsA] =
      tma_partition(tma_load_A, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                    group_modes<0, 2>(gA));
  auto [tBgB, tBsB] =
      tma_partition(tma_load_B, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                    group_modes<0, 2>(gB));

  typename Kernel_traits::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;
  using T = typename Kernel_traits::DType;
  constexpr int kTmaTransactionBytesA =
      sizeof(ArrayEngine<T, size(SmemLayoutA{})>);
  constexpr int kTmaTransactionBytesB =
      sizeof(ArrayEngine<T, size(SmemLayoutB{})>);

  uint64_t& smem_A_barrier = shared_storage.smem_A_barrier;
  uint64_t& smem_B_barrier = shared_storage.smem_B_barrier;

  if (warp_idx == 0 && lane_predicate) {
    TransactionBarrier::init(&smem_A_barrier, 1);
    TransactionBarrier::init(&smem_B_barrier, 1);
  }
  __syncthreads();

  auto NUM_TILES_K = size<2>(gA);

#pragma unroll 1
  for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile) {
    if (warp_idx == 0 && lane_predicate) {
      TransactionBarrier::arrive_and_expect_tx(&smem_A_barrier,
                                               kTmaTransactionBytesA);
      TransactionBarrier::arrive_and_expect_tx(&smem_B_barrier,
                                               kTmaTransactionBytesB);
      copy(tma_load_A.with(smem_A_barrier), tAgA(_, k_tile), tAsA);
      copy(tma_load_B.with(smem_B_barrier), tBgB(_, k_tile), tBsB);
    }
    TransactionBarrier::wait(&smem_A_barrier, k_tile % 2);
    TransactionBarrier::wait(&smem_B_barrier, k_tile % 2);
    cute::warpgroup_arrive();

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
  }

  axpby(params.alpha, tCrC, params.beta, tCgC);
}
