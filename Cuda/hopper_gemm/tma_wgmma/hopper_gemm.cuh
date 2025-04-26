#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;

template <typename DType, typename KeTraits, typename TmaLoadA,
          typename TmaLoadB>
__global__ void ke_cute_tma_wgmma(CUTE_GRID_CONSTANT TmaLoadA const tma_load_A,
                                  CUTE_GRID_CONSTANT TmaLoadB const tma_load_B,
                                  DType* gC_ptr) {
  using SmemLayoutA = typename KeTraits::SmemLayoutA;
  using SmemLayoutB = typename KeTraits::SmemLayoutB;

  constexpr int kM = KeTraits::kM;
  constexpr int kN = KeTraits::kN;
  constexpr int kK = KeTraits::kK;

  constexpr int kTM = KeTraits::kTM;
  constexpr int kTN = KeTraits::kTN;
  constexpr int kTK = KeTraits::kTK;

  Tensor mA = tma_load_A.get_tma_tensor(make_shape(kM, kK));
  Tensor mB = tma_load_B.get_tma_tensor(make_shape(kN, kK));
  Tensor mC = make_tensor(make_gmem_ptr(gC_ptr), make_shape(kM, kN),
                          make_stride(_1{}, kM));

  auto cta_tiler = make_shape(Int<kTM>{}, Int<kTN>{}, Int<kTK>{});
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  using SharedStorage = typename KeTraits::SharedStorage;
  extern __shared__ char smem_[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
  Tensor sB =
      make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_load_A, Int<0>{},   //
                                    Layout<_1>{},           //
                                    group_modes<0, 2>(sA),  //
                                    group_modes<0, 2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_load_B, Int<0>{},   //
                                    Layout<_1>{},           //
                                    group_modes<0, 2>(sB),  //
                                    group_modes<0, 2>(gB));

  typename KeTraits::TiledMma tiled_mma;
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
  constexpr int kTmaTransactionBytesA =
      sizeof(ArrayEngine<DType, size(SmemLayoutA{})>);
  constexpr int kTmaTransactionBytesB =
      sizeof(ArrayEngine<DType, size(SmemLayoutB{})>);

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
    warpgroup_arrive();

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    warpgroup_commit_batch();
    warpgroup_wait<0>();
  }

  DType alpha = static_cast<DType>(1.0);
  DType beta = static_cast<DType>(0.0);
  axpby(alpha, tCrC, beta, tCgC);
}
