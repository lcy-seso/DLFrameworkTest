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
  // constants
  constexpr int kM = KeTraits::kM;
  constexpr int kN = KeTraits::kN;
  constexpr int kK = KeTraits::kK;

  constexpr int kTM = KeTraits::kTM;
  constexpr int kTN = KeTraits::kTN;
  constexpr int kTK = KeTraits::kTK;

  auto cta_tiler = make_shape(Int<kTM>{}, Int<kTN>{}, Int<kTK>{});
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);

  // shared memory buffer
  using SharedStorage = typename KeTraits::SharedStorage;
  extern __shared__ char smem_[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

  // global tile of A
  Tensor tensor_A = tma_load_A.get_tma_tensor(make_shape(kM, kK));
  Tensor gA = local_tile(tensor_A, cta_tiler, cta_coord, Step<_1, X, _1>{});

  Tensor tensor_B = tma_load_B.get_tma_tensor(make_shape(kN, kK));
  Tensor gB = local_tile(tensor_B, cta_tiler, cta_coord, Step<X, _1, _1>{});

  Tensor tensor_C =
      make_tensor(make_gmem_ptr(gC_ptr), typename KeTraits::LayoutGmemC{});
  Tensor gC = local_tile(tensor_C, cta_tiler, cta_coord, Step<_1, _1, X>{});

  // shared memory tile of A and B
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                          typename KeTraits::SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                          typename KeTraits::SmemLayoutB{});

  // partition A and B
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

  // The fragments tCrA (operand A is sourced from SMEM) and tCrB (operand B is
  // sourced from SMEM) aren’t register-backed tensors whose values are copied
  // from SMEM, but rather matrix descriptors constructed on top of SMEM.
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kTmaTransactionBytesA =
      sizeof(ArrayEngine<DType, size(typename KeTraits::SmemLayoutA{})>);
  constexpr int kTmaTransactionBytesB =
      sizeof(ArrayEngine<DType, size(typename KeTraits::SmemLayoutB{})>);

  uint64_t& smem_A_barrier = shared_storage.smem_A_barrier;
  uint64_t& smem_B_barrier = shared_storage.smem_B_barrier;

  if (warp_idx == 0 && lane_predicate) {
    TransactionBarrier::init(&smem_A_barrier, 1);
    TransactionBarrier::init(&smem_B_barrier, 1);
  }
  __syncthreads();

  auto NUM_TILES_K = size<2>(gA);
#pragma unroll 1
  for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile) {  // main loop
    if (warp_idx == 0 && lane_predicate) {
      TransactionBarrier::arrive_and_expect_tx(&smem_A_barrier,
                                               kTmaTransactionBytesA);
      TransactionBarrier::arrive_and_expect_tx(&smem_B_barrier,
                                               kTmaTransactionBytesB);
      copy(tma_load_A.with(smem_A_barrier), tAgA(_, k_tile), tAsA);
      copy(tma_load_B.with(smem_B_barrier), tBgB(_, k_tile), tBsB);
    }
    TransactionBarrier::wait(&smem_A_barrier, k_tile);
    TransactionBarrier::wait(&smem_B_barrier, k_tile);
    warpgroup_arrive();

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    warpgroup_commit_batch();
    warpgroup_wait<0>();
  }

  DType alpha = static_cast<DType>(1.0);
  DType beta = static_cast<DType>(0.0);
  axpby(alpha, tCrC, beta, tCgC);
}
