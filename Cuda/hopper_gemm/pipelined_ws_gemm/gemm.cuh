#pragma once

#include "cuda_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"

template <const int kNumTmaMulticast>
__device__ void empty_barrier_arrive(uint64_t* empty_barrier,
                                     uint32_t in_group_idx) {
  if constexpr (kNumTmaMulticast == 1) {
    if (in_group_idx == 0) {
      arrive(empty_barrier);
    }
  } else {
    if (in_group_idx < kNumTmaMulticast) {
      arrive_cluster(empty_barrier, in_group_idx);
    }
  }
}

template <typename DType, const int kM_, const int kN_, const int kK_,
          const int kTM_, const int kTN_, const int kTK_,
          const int kNumTmaMulticast_ = 1, const int kNumStages_ = 2>
struct GemmTraits {
  static constexpr int kM = kM_;
  static constexpr int kN = kN_;
  static constexpr int kK = kK_;

  static constexpr int kTM = kTM_;
  static constexpr int kTN = kTN_;
  static constexpr int kTK = kTK_;

  static constexpr int kNumTmaMulticast = kNumTmaMulticast_;

  static constexpr int kNumStages = kNumStages_;

  // the size of shared memory for each operand and result
  static constexpr int kSizeA = kTM * kTK * sizeof(DType);
  static constexpr int kSizeB = kTN * kTK * sizeof(DType);
  static constexpr int kSizeC = kTM * kTN * sizeof(DType);

  static_assert(kSizeC % 1024 == 0,
                "Shared memory of output tensor must be aligned to 1024 bytes");
  static_assert(kSizeA % 1024 == 0,
                "Shared memory of operand A must be aligned to 1024 bytes");
  static_assert(kSizeB % 1024 == 0,
                "Shared memory of operand B must be aligned to 1024 bytes");

  static constexpr int kSharedDataSize =
      (kSizeA + kSizeB) * kNumStages + kSizeC;
  static constexpr int kSharedMemSize =  // data + barriers
      kSharedDataSize + kNumStages * sizeof(uint64_t) * 2;

  static constexpr int kExpectedTmaBytes = kSizeA + kSizeB;

  static constexpr uint32_t kKShapeAllStages = kNumStages * kTK;
  static constexpr uint32_t kKNumIterations = CeilDiv<kK, kKShapeAllStages>;

  static constexpr uint32_t kNumWarpGroup = 2;
  static constexpr uint32_t kThreads = 128 * kNumWarpGroup;
  // thread 0 ~ kMathThreads - 1: consumer
  // thread kMathThreads ~ kThreads - 1: producer
  static constexpr uint32_t kMathThreads = 128;

  // register reconfigurations
  static constexpr uint32_t kNumTMARegisters = 40;
  static constexpr uint32_t kNumMathRegisters = 232;

  // tile scheduler
  using Scheduler_ = Scheduler<kM, kN, kTM, kTN, kNumTmaMulticast>;
};

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
  auto barrier_start_ptr =
      reinterpret_cast<uint64_t*>(buf + KeTraits::kSharedDataSize);
  auto a_ptr = smem_c + KeTraits::kSizeC;
  auto b_ptr = a_ptr + KeTraits::kSizeA * kNumStages;

#pragma unroll
  for (uint32_t i = 0; i < kNumStages; ++i) {
    smem_a[i] = a_ptr + i * KeTraits::kSizeA;
    smem_b[i] = b_ptr + i * KeTraits::kSizeB;

    full_barriers[i] = barrier_start_ptr + i;
    empty_barriers[i] = barrier_start_ptr + i;
  }

  const uint32_t warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  const uint32_t in_group_idx = threadIdx.x % 128;
  const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane_idx = get_lane_id();

  // prefetch tma descriptors
  if (threadIdx.x == KeTraits::kMathThreads) {
    prefetch_tma_descriptor(&tma_desc_a);
    prefetch_tma_descriptor(&tma_desc_b);
    prefetch_tma_descriptor(&tma_desc_c);
  }
  __syncwarp();

  if (threadIdx.x == KeTraits::kMathThreads) {  // the leader thread
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      // TMA thread signals `full_barriers`. since there is only one thread
      // issue TMA data transfer, the arrive count for `full_barriers` is
      // expected to be 1.
      init_barrier(full_barriers[i], 1);
      init_barrier(empty_barriers[i], KeTraits::kMathThreads / 128);
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
            wait(empty_barriers[s], idx & 1);

            const uint32_t k_idx = k * KeTraits::kKShapeAllStages + s * kTK;
            if (k_idx >= KeTraits::kK) {
              // signal shared memory buffer as "full" to unblock the consumer
              // for the pipeline drain.
              arrive(full_barriers[s]);
              continue;
            }

            tma_load(&tma_desc_a, full_barriers[s], smem_a[s], k_idx,
                     m_block_idx * kTM);
            tma_load(&tma_desc_b, full_barriers[s], smem_b[s], k_idx,
                     n_block_idx * kTN);
            arrive_and_expect_tx(full_barriers[s], KeTraits::kExpectedTmaBytes);
          }  // end of stages
        }  // end of tile scheduler
      }  // end of lead thread
    }  // end of producer
  } else {  // Consumer, WGMMA
    warpgroup_reg_alloc<KeTraits::kNumMathRegisters>();
    float accum[WGMMA::kNumAccums];

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      idx = scheduler.current_iter * KeTraits::kKNumIterations;

      wait(full_barriers[0], idx & 1);
      // TO ADD, compute wgmma stage 0 ...
      empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[0], in_group_idx);

#pragma unroll
      for (uint32_t s = 1; s < kNumStages; ++s) {
        wait(full_barriers[s], idx & 1);

        const uint32_t k_idx = s * KeTraits::kTK;
        if (k_idx >= KeTraits::kK) {
          empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s],
                                                 in_group_idx);
          continue;
        }

        // compute wgmma stage s
        empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s], in_group_idx);
      }

      for (uint32_t k = 1; k < KeTraits::kKNumIterations; ++k) {
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
          wait(full_barriers[s], (idx + k) & 1);

          const uint32_t k_idx = k * KeTraits::kKShapeAllStages + s * kTK;
          if (k_idx >= KeTraits::kK) {
            empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s],
                                                   in_group_idx);
            continue;
          }

          // compute wgmma stage s
          empty_barrier_arrive<kNumTmaMulticast>(empty_barriers[s],
                                                 in_group_idx);
        }
      }

      // store results
    }
  }
}
