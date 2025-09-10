#pragma once

namespace block {

__device__ __forceinline__ void sync() { __syncthreads(); }

template <int N>
__device__ __forceinline__ void wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void commit_copy_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void copy_async() {
  commit_copy_group();
  wait_group<0>();
}

__device__ __forceinline__ void bar_sync(const unsigned barrier_id,
                                         const unsigned thread_count) {
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrier_id), "r"(thread_count)
               : "memory");
}

}  // namespace block
