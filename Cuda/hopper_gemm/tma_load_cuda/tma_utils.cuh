#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>

typedef __nv_bfloat16 __bfloat16;
typedef __nv_fp8_e4m3 __fp8_e4m3;
typedef __nv_fp8_e5m2 __fp8_e5m2;

// Helper template to map C++ types to CUDA tensor map data types
template <typename T>
struct TMADataTypeTraits;

template <>
struct TMADataTypeTraits<uint8_t> {  // 1 byte
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <>
struct TMADataTypeTraits<uint16_t> {  // 2 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT16;
};

template <>
struct TMADataTypeTraits<int> {  // 4 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_INT32;
};

template <>
struct TMADataTypeTraits<uint32_t> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT32;
};

template <>
struct TMADataTypeTraits<int64_t> {  // 8 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_INT64;
};

template <>
struct TMADataTypeTraits<uint64_t> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT64;
};

template <>
struct TMADataTypeTraits<__fp8_e4m3> {  // 1 byte
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <>
struct TMADataTypeTraits<__fp8_e5m2> {  // 1 byte
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <>
struct TMADataTypeTraits<__half> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
};

template <>
struct TMADataTypeTraits<__bfloat16> {  // 2 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};

template <>
struct TMADataTypeTraits<float> {  // 4 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
};

template <>
struct TMADataTypeTraits<double> {  // 8 bytes
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
};

__device__ __forceinline__ uint32_t get_lane_id() {
  uint32_t lane_id;
  asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

template <typename DType>
class TMADescriptor {
public:
  TMADescriptor() : is_initialized(false) {}
  ~TMADescriptor() = default;

  void create_tma_2d_desc(
      void* global_address,
      uint64_t global_dim[2],  // 2 stands for rank
      uint32_t shared_dim[2], uint64_t global_stride,
      CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    assert(strides[0] % 16 == 0 && "stride must be multiples of 16 bytes");

    if (!is_driver_initialized) {
      CHECK_CU(cuInit(0));
      is_driver_initialized = true;
    }
    static constexpr uint32_t rank = 2;
    // stride in bytes
    uint64_t strides[rank - 1] = {global_stride * sizeof(DType)};
    uint32_t element_stride[rank] = {1, 1};

    CHECK_CU(cuTensorMapEncodeTiled(
        &tensor_map,                         // Output tensor map
        data_type,                           // Data type
        rank,                                // Tensor rank (2D)
        global_address,                      // Global address
        global_dim,                          // Global dimensions
        strides,                             // Global strides (in elements)
        shared_dim,                          // Box dimensions
        element_stride,                      // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,       // Interleave layout
        swizzle,                             // Swizzle mode
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // L2 promotion
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE    // Out-of-bounds fill
        ));

    is_initialized = true;
  }

  const CUtensorMap& get_tma_desc() const {
    if (!is_initialized) {
      throw std::runtime_error("TMA descriptor not initialized");
    }
    return tensor_map;
  }

  void print_tma_desc_info() const {
    if (!is_initialized) {
      std::cout << "TMA Descriptor not initialized" << std::endl;
      return;
    }

    std::cout << "TMA Descriptor Info:" << std::endl;
    std::cout << "Address: " << std::hex << &tensor_map << std::dec
              << std::endl;
  }

private:
  CUtensorMap tensor_map;
  bool is_initialized;
  bool is_driver_initialized;

  static constexpr CUtensorMapDataType data_type =
      TMADataTypeTraits<DType>::value;
};

__host__ __device__ __forceinline__ void prefetch_tma_descriptor(
    const void* desc_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
#else
  // For host code or older architectures, this is a no-op
  (void)desc_ptr;
#endif
}

// Barrier management functions
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

/// @brief Implements a barrier synchronization for threads within a single
///        thread block.
///
///        The core instruction is `mbarrier.try_wait` for barrier
///        synchronization, which checks if all threads in the scope (::cta, the
///        entire thread block) have arrived at the barrier with the correct
///        phase.
///
///        "try wait" is a potentially blocking instruction which tests for the
///        completion of the phase. If the phase is not complete, the executing
///        thread may be suspended. P1 becomes true if the barrier is completed
///        (all threads have arrived), and false otherwise.
///
///        The `wait` function itself doesn't decide what the state should be;
///        It just uses the phase value you pass in to check against the
///        hardware barrier's current state. This mechanism is crucial for
///        safely reusing the same barrier object multiple times.
///
/// @param barrier Pointer to the barrier synchronization object.
/// @param phase The phase of the barrier.
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

/// @brief The `mbarrier.arrive` instruction increments the arrival counter for
///        the barrier. When the final thread arrives at the barrier (meaning
///        the arrival count reaches the expected number of threads), the
///        hardware automatically flips the phase bit of the barrier.
__device__ __forceinline__ void arrive(uint64_t const* barrier) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  // The sink symbol "_" is used to inform the compiler that the 64-bit
  // phase token can be discarded. The caller is only interested in the side
  // effect of the arrive operation, which increases the arrival count of the
  // mbarrier, and does not need the phase token for any future wait operations,
  // so it can be safely discarded.

  // when the sink symbol "_" is used, a thread's only job is to signal its
  // arrival but it will not wait for other threads at that same synchronization
  // point. This is common in producer-consumer patterns.

  // Producer Threads might be responsible for fetching data from global memory
  // into shared memory using TMA (Tensor Memory Access). Once a producer has
  // finished its data transfer, it needs to signal to the consumer threads that
  // the data is ready. It does this by arriving at a barrier. However, the
  // producer thread itself might not need to wait; it might go on to fetch the
  // next chunk of data or simply terminate.

  // Consumer Threads: These threads (e.g., the threads performing the math
  // operations in a GEMM) will wait on that same barrier to ensure the data is
  // ready before they start consuming it.

  // In this scenario, the producer thread's arrive is a "fire-and-forget"
  // signal. It doesn't need a phase token because it's not going to wait.

  asm volatile(
      "{\n\t"
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"  // "_" sink symbol is used
      "}"
      :
      : "r"(barrier_ptr));
}

__device__ __forceinline__ void arrive_and_expect_tx(uint64_t* mbar,
                                                     uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"l"(mbar),
               "r"(bytes)
               : "memory");
}

/// @brief Waits until at most `N` many of the committed TMA store operations
///        are pending. (e.g., set `N` to be 0 means wait for all TMA store
///        operations to complete.)
template <int N>
__device__ __forceinline__ void tma_store_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void tma_store_wait_group(uint32_t n) {
  // Runtime version: can only wait for all outstanding operations
  // For compile-time constants, use the template version above
  (void)n;  // Suppress unused parameter warning
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <uint32_t kRegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kRegCount));
}

template <uint32_t kRegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kRegCount));
}

__device__ __forceinline__ void tma_load(void const* desc, uint64_t* barrier,
                                         void* smem, int32_t const& crd_0,
                                         int32_t const& crd_1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr uint64_t CACHE_HINT = 0x1000000000000000ull;

  uint64_t tma_ptr = reinterpret_cast<uint64_t>(desc);
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_ptr), "l"(tma_ptr), "r"(barrier_ptr), "r"(crd_0), "r"(crd_1),
        "l"(CACHE_HINT)
      : "memory");

#else
  // For host code or older architectures, this is a no-op
  (void)desc;
  (void)smem;
  (void)crd_0;
  (void)crd_1;
#endif
}

__device__ __forceinline__ void tma_store(void const* desc, void* smem,
                                          int32_t const& crd_0,
                                          int32_t const& crd_1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(desc);
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
      "[%1];"
      :
      : "l"(tma_ptr), "r"(smem_ptr), "r"(crd_0), "r"(crd_1)
      : "memory");
#else
  // For host code or older architectures, this is a no-op
  (void)desc;
  (void)smem;
  (void)crd_0;
  (void)crd_1;
#endif
}
