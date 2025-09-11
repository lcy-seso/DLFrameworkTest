#include "barrier.cuh"
#include "copy.cuh"
#include "cuda_utils.cuh"
#include "sync.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <iostream>

using namespace copy;
using namespace barrier;
namespace {

__forceinline__ __device__ uint32_t smem_ptr_to_uint(void const* const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__forceinline__ __device__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : :);
}

__forceinline__ __device__ void mbarrier_cp_async_arrive(
    uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_ptr));
}

__device__ __forceinline__ void init_barrier(uint64_t* barrier,
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

__device__ __forceinline__ void arrive(uint64_t const* barrier) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"  // "_" sink symbol is used
      "}"
      :
      : "r"(barrier_ptr));
}

__device__ __forceinline__ void wait(uint64_t* barrier, int phase) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  constexpr uint32_t ticks = 0x989680;  // timeout
  asm volatile(
      "{\n\t"
      ".reg .pred       P1; \n\t"  // predicate register
      "LAB_WAIT: \n\t"             // spin-wait loop
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
      "@P1 bra DONE; \n\t"
      "bra     LAB_WAIT; \n\t"
      "DONE: \n\t"
      "}"
      :
      : "r"(barrier_ptr), "r"(phase), "r"(ticks));
}

template <const int kNumel>
__global__ void test_bar_sync(const bfloat16* scores, const bfloat16* bias,
                              bfloat16* output) {
  if (blockIdx.x) return;

  using DType = bfloat16;
  extern __shared__ __align__(1024) uint8_t buf_[];
  auto sptr_uint = [](auto* ptr) { return __cvta_generic_to_shared(ptr); };
  int tid = threadIdx.x;

  DType* scores_s = reinterpret_cast<DType*>(buf_);
  DType* bias_s = scores_s + kNumel;
  DType* biased_scores_s = bias_s + kNumel;

  // for barrier
  auto barriers = reinterpret_cast<uint64_t*>(biased_scores_s + kNumel);
  if (threadIdx.x == 0) {
    init_barrier(&barriers[0], 1);
    init_barrier(&barriers[1], 4);
  }
  __syncthreads();

  static constexpr int kBytesPerAccess = 128 / 8;
  static constexpr int kNumPerAccess = kBytesPerAccess / sizeof(DType);

  // step1: load scores and bias to shared memory
  int num_load_threads = kNumel / kNumPerAccess;
  int offset = tid * kNumPerAccess;

  if (tid < num_load_threads) {
    ld_global_st_shared<kBytesPerAccess>(sptr_uint(scores_s + offset),
                                         scores + offset);
    ld_global_st_shared<kBytesPerAccess>(sptr_uint(bias_s + offset),
                                         bias + offset);
  }

  // block::copy_async();
  wait(&barriers[0], 0);

  // step2 : add bias to scores using vectorized addition(128 threads)
  int compute_threads = kNumel / 2;
  if (tid < compute_threads) {
    biased_scores_s[tid] = scores_s[tid] + bias_s[tid];
    bfloat162 v_scores = *reinterpret_cast<bfloat162*>(scores_s + tid * 2);
    bfloat162 v_bias = *reinterpret_cast<bfloat162*>(bias_s + tid * 2);
    bfloat162 biased_score = __hadd2(v_scores, v_bias);

    *reinterpret_cast<bfloat162*>(biased_scores_s + tid * 2) = biased_score;

    if (tid % 32 == 0) {
      arrive(&barriers[1]);
    }
  }
  wait(&barriers[1], 0);

  // step3: store biased scores to global memory
  if (tid < num_load_threads) {
    ld_shared_st_global<kBytesPerAccess>(output + offset,
                                         sptr_uint(biased_scores_s + offset));
  }
}
}  // namespace

int main(int argc, char** argv) {
  printf("Starting program...\n");
  using DType = bfloat16;
  static constexpr int kN = 256;

  thrust::host_vector<DType> h_a(kN);
  thrust::host_vector<DType> h_b(kN);
  thrust::host_vector<DType> h_c_ref(kN);

  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = rand_bfloat16();
    h_b[i] = rand_bfloat16();
    h_c_ref[i] = h_a[i] + h_b[i];
  }
  thrust::device_vector<DType> d_a = h_a;
  thrust::device_vector<DType> d_b = h_b;

  thrust::host_vector<DType> h_c(kN);
  thrust::fill(h_c.begin(), h_c.end(), static_cast<DType>(0));
  thrust::device_vector<DType> d_c = h_c;
  CudaCheck(cudaDeviceSynchronize());

  static constexpr int num_sms = 128;
  static constexpr int kThreads = 512;

  dim3 blocks(num_sms, 1);
  dim3 threads(kThreads, 1, 1);

  static constexpr int kSharedMemSize =
      kN * sizeof(DType) * 3 + 2 * sizeof(std::uint64_t) /*mbarrier*/;

  test_bar_sync<kN><<<blocks, threads, kSharedMemSize>>>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()));
  CudaCheck(cudaDeviceSynchronize());

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return -1;
  }

  printf("\n\nKernel execution completed successfully\n");
  h_c = d_c;
  for (int i = 0; i < h_c.size(); ++i) {
    printf("%.2f, ", ToFloat(h_c[i]));

    if ((i + 1) % 8 == 0) printf("\n");
  }

  // printf("\n\nh_c_ref: \n");
  // for (int i = 0; i < h_c.size(); ++i) {
  //   printf("%.2f, ", ToFloat(h_c_ref[i]));

  //   if ((i + 1) % 8 == 0) printf("\n");
  // }
  // printf("\n");

  return 0;
}
