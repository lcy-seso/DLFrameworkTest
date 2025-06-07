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

// Error checking macro for CUDA driver API
#define CHECK_CU(call)                                                \
  do {                                                                \
    CUresult err = call;                                              \
    if (err != CUDA_SUCCESS) {                                        \
      const char* error_str;                                          \
      cuGetErrorString(err, &error_str);                              \
      fprintf(stderr, "CUDA Driver API error in %s at line %d: %s\n", \
              __FILE__, __LINE__, error_str);                         \
      throw std::runtime_error(error_str);                            \
    }                                                                 \
  } while (0)

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
    if (!is_driver_initialized) {
      CHECK_CU(cuInit(0));
      is_driver_initialized = true;
    }
    static constexpr uint32_t rank = 2;
    uint64_t global_stride_[rank - 1] = {global_stride};
    uint32_t element_stride[rank] = {1, 1};

    CHECK_CU(cuTensorMapEncodeTiled(
        &tensor_map,                         // Output tensor map
        data_type,                           // Data type
        rank,                                // Tensor rank (2D)
        global_address,                      // Global address
        global_dim,                          // Global dimensions
        global_stride_,                      // Global strides (in elements)
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
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"l"(mbar), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"l"(mbar),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
  uint32_t result;
  do {
    asm volatile("mbarrier.try_wait.shared.b64 %0, [%1], %2;"
                 : "=r"(result)
                 : "l"(mbar), "r"(phase)
                 : "memory");
  } while (result == 0);
}

// Proper barrier wait implementation using CUTE's PTX syntax
__device__ __forceinline__ void wait_barrier(uint64_t& smem_barrier,
                                             int phase_bit) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem_barrier));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra DONE;\n"
      "bra                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

__device__ __forceinline__ void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

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

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store(void const* desc_ptr, void* smem_ptr,
                                          int32_t const& crd_0,
                                          int32_t const& crd_1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
      "[%1];"
      :
      : "l"(tma_ptr), "r"(smem_int_ptr), "r"(crd_0), "r"(crd_1)
      : "memory");
#else
  // For host code or older architectures, this is a no-op
  (void)desc_ptr;
  (void)smem_ptr;
  (void)crd_0;
  (void)crd_1;
#endif
}

__device__ __forceinline__ void tma_copy(void const* desc_ptr,
                                         uint64_t* barrier_ptr, void* smem_ptr,
                                         int32_t const& crd_0,
                                         int32_t const& crd_1,
                                         uint32_t num_tma_multicast) {
  constexpr auto cache_hint =
      static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  if (num_tma_multicast == 1) {
    cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr,
                                 crd_0, crd_1);
  } else if (cute::block_rank_in_cluster() == 0) {
    cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr,
                                           (1 << num_tma_multicast) - 1,
                                           cache_hint, smem_ptr, crd_0, crd_1);
  }
}
