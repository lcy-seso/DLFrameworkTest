#pragma once

#include <cstdint>

template <uint32_t kM, uint32_t kN, uint32_t kTM, uint32_t kTN,
          uint32_t kNumTmaMulticast, uint32_t kNumBlocksPerGroup = 16>
struct Scheduler {
  int current_iter = -1;
  uint32_t num_aligned_m_blocks;
  uint32_t num_aligned_n_blocks;
  uint32_t num_blocks;

  __host__ __device__ explicit Scheduler() {
    num_aligned_m_blocks = CEIL_DIV(kM, kTM);
    num_aligned_n_blocks = CEIL_DIV(kN, kTN);
    num_blocks = num_aligned_m_blocks * num_aligned_n_blocks;
  }

  __host__ __device__ void get_swizzled_block_idx(int block_idx,
                                                  uint32_t& m_block_idx,
                                                  uint32_t& n_block_idx) {
    static_assert(kNumBlocksPerGroup % kNumTmaMulticast == 0,
                  "Invalid group size");

    const auto num_blocks_per_group = num_aligned_n_blocks * kNumBlocksPerGroup;
    const auto group_idx = block_idx / num_blocks_per_group;
    const auto in_group_idx = block_idx % num_blocks_per_group;

    const auto first_m_block_idx = group_idx * kNumBlocksPerGroup;
    const auto num_m_blocks_in_group =
        min(kNumBlocksPerGroup, num_aligned_m_blocks - first_m_block_idx);
    m_block_idx = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
    n_block_idx = in_group_idx / num_m_blocks_in_group;
  }

  __device__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const auto next_block_idx = (++current_iter) * gridDim.x + blockIdx.x;
    if (next_block_idx >= num_blocks) return false;

    get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    return true;
  }
};
