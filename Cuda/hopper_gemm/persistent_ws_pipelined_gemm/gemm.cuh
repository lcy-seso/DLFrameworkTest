#pragma once

#include "cuda_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "wgmma_utils.cuh"

template <typename DType, typename KeTraits>
__global__ void __launch_bounds__(256, 1)  // minimum 1 block per SM
    hopper_gemm(const __grid_constant__ CUtensorMap tma_desc_a,
                const __grid_constant__ CUtensorMap tma_desc_b,
                const __grid_constant__ CUtensorMap tma_desc_c) {
  static constexpr int kNumStages = KeTraits::kNumStages;
  static constexpr int kNumTmaMulticast = KeTraits::kNumTmaMulticast;
  static constexpr int kTM = KeTraits::kTM;
  static constexpr int kTN = KeTraits::kTN;
  static constexpr int kTK = KeTraits::kTK;
  extern __shared__ __align__(1024) uint8_t buf[];

  DType* smem_a[kNumStages];
  DType* smem_b[kNumStages];
  DType* smem_c = reinterpret_cast<DType*>(buf);

  uint64_t* full_barriers[kNumStages];
  uint64_t* empty_barriers[kNumStages];

  // shared memory data pointers for each operand and output
  auto a_ptr = smem_c + KeTraits::kShapeC;
  auto b_ptr = a_ptr + KeTraits::kShapeA;

  auto barrier_start_ptr =
      reinterpret_cast<uint64_t*>(buf + KeTraits::kSharedDataSize);
#pragma unroll
  for (uint32_t i = 0; i < kNumStages; ++i) {
    smem_a[i] = a_ptr + i * KeTraits::kShapeA;
    smem_b[i] = b_ptr + i * KeTraits::kShapeB;

    full_barriers[i] = barrier_start_ptr + i;
    empty_barriers[i] = barrier_start_ptr + kNumStages + i;
  }

  // prefetch tma descriptors
  if (threadIdx.x == KeTraits::kMathThreads) {
    prefetch_tma_descriptor(&tma_desc_a);
    prefetch_tma_descriptor(&tma_desc_b);
    prefetch_tma_descriptor(&tma_desc_c);
  }
  __syncwarp();

  if (threadIdx.x == KeTraits::kMathThreads) {  // the lead thread
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      // TMA thread signals `full_barriers`. since there is only one thread
      // issue TMA data transfer, the arrive count for `full_barriers` is
      // expected to be 1.
      init_barrier(full_barriers[i], 1);
      init_barrier(empty_barriers[i],
                   kNumTmaMulticast * KeTraits::kMathThreads / 128);
    }
  }
  // Synchronize all threads to make barrier visible in normal memory model
  __syncthreads();

  typename KeTraits::Scheduler_ scheduler;
  uint32_t m_block_idx, n_block_idx, idx;

  if (threadIdx.x >= KeTraits::kMathThreads) {  // Producer, TMA copy
    warpgroup_reg_dealloc<KeTraits::kNumTMARegisters>();

    if (threadIdx.x == KeTraits::kMathThreads) {  // lead thread issues TMA
      while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        for (uint32_t k = 0; k < KeTraits::kKNumIterations; ++k) {
          idx = scheduler.current_iter * KeTraits::kKNumIterations + k + 1;

#pragma unroll
          for (uint32_t s = 0; s < kNumStages; ++s) {
            wait(empty_barriers[s], idx & 1);  // phase bit 1, 0, 1, 0, ...

            const uint32_t k_idx = k * KeTraits::kKShapeAllStages + s * kTK;

            if (k_idx >= KeTraits::kK) {
              // signal shared memory buffer as "full" to unblock the consumer
              // for the pipeline drain.
              arrive(full_barriers[s]);
              continue;
            }

            arrive_and_expect_tx(full_barriers[s], KeTraits::kExpectedTmaBytes);
            tma_load(&tma_desc_a, full_barriers[s], smem_a[s], k_idx,
                     m_block_idx * kTM);
            tma_load(&tma_desc_b, full_barriers[s], smem_b[s], k_idx,
                     n_block_idx * kTN);
          }  // end of stages
        }  // end of tile scheduler
      }  // end of lead thread
    }  // end of producer
  } else {  // Consumer, WGMMA
    uint32_t warp_idx = get_warp_idx();
    uint32_t lane_idx = get_lane_id();
    uint32_t warp_group_idx = get_warp_group_idx();
    const uint32_t in_group_idx = threadIdx.x % 128;

    warpgroup_reg_alloc<KeTraits::kNumMathRegisters>();
    float accum[WGMMA::kNumAccums];

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      idx = scheduler.current_iter * KeTraits::kKNumIterations;

      wait(full_barriers[0], idx & 1);  // phase bit 0, 1, 0, 1, ...
      compute_wgmma_stage(0, accum, smem_a, smem_b, kTK, false);
      empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[0]);

#pragma unroll
      for (uint32_t s = 1; s < kNumStages; ++s) {
        wait(full_barriers[s], idx & 1);

        const uint32_t k_idx = s * KeTraits::kTK;
        if (k_idx >= KeTraits::kK) {
          empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s]);
          continue;
        }

        compute_wgmma_stage(s, accum, smem_a, smem_b, kTK);
        empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s]);
      }

      for (uint32_t k = 1; k < KeTraits::kKNumIterations; ++k) {
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
          wait(full_barriers[s], (idx + k) & 1);

          const uint32_t k_idx = k * KeTraits::kKShapeAllStages + s * kTK;
          if (k_idx >= KeTraits::kK) {
            empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s]);
            continue;
          }

          compute_wgmma_stage(s, accum, smem_a, smem_b, kTK);
          empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s]);
        }
      }
      tma_store_wait<0>();
      asm volatile("bar.sync %0, 128;\n" ::"r"(warp_group_idx + 8) : "memory");

      uint32_t smem_store_offset =
          (warp_idx * 16 + lane_idx % 16) * 32 + 8 * (lane_idx / 16);

      uint32_t tma_store_smem_offset = warp_group_idx * WGMMA::kM * 32;
      uint32_t tma_store_gmem_n = n_block_idx * KeTraits::kTN;
      uint32_t tma_store_gmem_m =
          m_block_idx * KeTraits::kTM + warp_group_idx * WGMMA::kM;

#pragma unroll
      for (auto j = 0; j < WGMMA::kNumAccums / 16; ++j) {
        const auto i0 = j * 2 + 0;
        StoreMatrixU32x4<__nv_bfloat162>::copy(
            __float22bfloat162_rn({accum[i0 * 8 + 0], accum[i0 * 8 + 1]}),
            __float22bfloat162_rn({accum[i0 * 8 + 2], accum[i0 * 8 + 3]}),
            __float22bfloat162_rn({accum[i0 * 8 + 4], accum[i0 * 8 + 5]}),
            __float22bfloat162_rn({accum[i0 * 8 + 6], accum[i0 * 8 + 7]}),
            smem_c + smem_store_offset);

        const auto i1 = j * 2 + 1;
        StoreMatrixU32x4<__nv_bfloat162>::copy(
            __float22bfloat162_rn({accum[i1 * 8 + 0], accum[i1 * 8 + 1]}),
            __float22bfloat162_rn({accum[i1 * 8 + 2], accum[i1 * 8 + 3]}),
            __float22bfloat162_rn({accum[i1 * 8 + 4], accum[i1 * 8 + 5]}),
            __float22bfloat162_rn({accum[i1 * 8 + 6], accum[i1 * 8 + 7]}),
            smem_c + smem_store_offset + 16);

        smem_store_offset += KeTraits::kTM * 32;

        // force warp group synchronization
        tma_store_fence();
        asm volatile("bar.sync %0, 128;\n" ::"r"(warp_group_idx + 8)
                     : "memory");

        if (in_group_idx == 0) {
          tma_store(&tma_desc_c, smem_c + tma_store_smem_offset,
                    tma_store_gmem_n, tma_store_gmem_m);

          tma_store_arrive();
        }
        __syncwarp();

        tma_store_smem_offset += KeTraits::kTM * 32;
        tma_store_gmem_n += 32;
      }
    }
  }
}
