#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

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

// TMA descriptor creation API
class TMADescriptor {
private:
  CUtensorMap tensorMap;
  bool isInitialized;

public:
  TMADescriptor() : isInitialized(false) {}

  ~TMADescriptor() = default;

  // Create 2D TMA Load descriptor for row-major layout
  void createTMALoad2D(
      void* globalAddress,       // Global memory pointer
      CUdeviceptr baseAddress,   // Base address for bounds checking
      uint64_t globalDim[2],     // Global tensor dimensions [height, width]
      uint64_t globalStride[2],  // Global tensor strides [height_stride,
                                 // width_stride]
      uint32_t boxDim[2],  // Tile dimensions to load [tile_height, tile_width]
      uint32_t elementStride[2],  // Element strides [1, 1] for contiguous
      CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
    // Initialize CUDA driver API if not already done
    static bool driverInit = false;
    if (!driverInit) {
      CHECK_CU(cuInit(0));
      driverInit = true;
    }

    // Data type: assuming float (32-bit)
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;

    CHECK_CU(cuTensorMapEncodeTiled(
        &tensorMap,     // Output tensor map
        dataType,       // Data type
        2,              // Tensor rank (2D)
        globalAddress,  // Global address
        globalDim,      // Global dimensions
        globalStride,   // Global strides (in elements)
        boxDim,         // Box dimensions
        elementStride,  // Element strides
        interleave,     // Interleave layout
        swizzle,        // Swizzle mode
        l2Promotion,    // L2 promotion
        oobFill         // Out-of-bounds fill
        ));

    isInitialized = true;
  }

  // Create 2D TMA Store descriptor
  void createTMAStore2D(
      void* globalAddress, CUdeviceptr baseAddress, uint64_t globalDim[2],
      uint64_t globalStride[2], uint32_t boxDim[2], uint32_t elementStride[2],
      CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
    createTMALoad2D(globalAddress, baseAddress, globalDim, globalStride, boxDim,
                    elementStride, interleave, swizzle, l2Promotion, oobFill);
  }

  // Get the raw tensor map pointer for use in kernels
  const CUtensorMap* getTensorMap() const {
    if (!isInitialized) {
      throw std::runtime_error("TMA descriptor not initialized");
    }
    return &tensorMap;
  }

  // Helper function to print tensor map info
  void printInfo() const {
    if (!isInitialized) {
      std::cout << "TMA Descriptor not initialized" << std::endl;
      return;
    }

    std::cout << "TMA Descriptor Info:" << std::endl;
    std::cout << "Address: " << std::hex << &tensorMap << std::dec << std::endl;
  }
};

// Convenience functions for common use cases

// Create TMA Load descriptor for row-major 2D tensor
inline TMADescriptor createTMALoad2DRowMajor(
    float* globalPtr,  // Global memory pointer
    int height,        // Tensor height
    int width,         // Tensor width
    int tileHeight,    // Tile height to load
    int tileWidth      // Tile width to load
) {
  TMADescriptor desc;

  uint64_t globalDim[2] = {static_cast<uint64_t>(height),
                           static_cast<uint64_t>(width)};
  uint64_t globalStride[2] = {static_cast<uint64_t>(width),
                              1};  // Row-major: [width, 1]
  uint32_t boxDim[2] = {static_cast<uint32_t>(tileHeight),
                        static_cast<uint32_t>(tileWidth)};
  uint32_t elementStride[2] = {1, 1};  // Contiguous access

  desc.createTMALoad2D(
      globalPtr,                                 // Global address
      reinterpret_cast<CUdeviceptr>(globalPtr),  // Base address
      globalDim,                                 // Global dimensions
      globalStride,                              // Global strides
      boxDim,                                    // Box dimensions
      elementStride                              // Element strides
  );

  return desc;
}

// Create TMA Store descriptor for row-major 2D tensor
inline TMADescriptor createTMAStore2DRowMajor(float* globalPtr, int height,
                                              int width, int tileHeight,
                                              int tileWidth) {
  TMADescriptor desc;

  uint64_t globalDim[2] = {static_cast<uint64_t>(height),
                           static_cast<uint64_t>(width)};
  uint64_t globalStride[2] = {static_cast<uint64_t>(width), 1};
  uint32_t boxDim[2] = {static_cast<uint32_t>(tileHeight),
                        static_cast<uint32_t>(tileWidth)};
  uint32_t elementStride[2] = {1, 1};

  desc.createTMAStore2D(globalPtr, reinterpret_cast<CUdeviceptr>(globalPtr),
                        globalDim, globalStride, boxDim, elementStride);

  return desc;
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
