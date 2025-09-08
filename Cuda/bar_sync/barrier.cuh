#pragma once

#include <cstdint>

namespace barrier {

__device__ __forceinline__ void init_barrier(std::uint64_t* barrier,
                                             int arrive_count) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n\t"
      "mbarrier.init.shared::cta.b64 [%1], %0; \n"
      "}"
      :
      : "r"(arrive_count), "r"(barrier_ptr));
}

}  // namespace barrier
