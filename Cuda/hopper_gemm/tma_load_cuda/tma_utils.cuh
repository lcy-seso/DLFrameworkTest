#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>

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
      uint32_t shared_dim[2],  //
      uint64_t global_stride,
      CUtensorMapSwizzle swizzle =
          CU_TENSOR_MAP_SWIZZLE_NONE,  // CU_TENSOR_MAP_SWIZZLE_128B,
      bool enable_l2_promotion = true) {
    if (!is_driver_initialized) {
      CHECK_CU(cuInit(0));
      is_driver_initialized = true;
    }

    static constexpr uint32_t rank = 2;
    // stride in bytes
    uint64_t strides[rank - 1] = {global_stride * sizeof(DType)};
    assert(strides[0] % 16 == 0 && "stride must be multiples of 16 bytes");
    uint32_t element_stride[rank] = {1, 1};

    CHECK_CU(cuTensorMapEncodeTiled(
        &tensor_map,                    // Output tensor map
        data_type,                      // Data type
        rank,                           // Tensor rank (2D)
        global_address,                 // Global address
        global_dim,                     // Global dimensions
        strides,                        // Global strides (in elements)
        shared_dim,                     // Box dimensions
        element_stride,                 // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,  // Interleave layout
        swizzle,                        // Swizzle mode
        enable_l2_promotion ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                            : CU_TENSOR_MAP_L2_PROMOTION_NONE,  // L2 promotion
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE  // Out-of-bounds fill
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

__device__ __forceinline__ void arrive_cluster(uint64_t* barrier,
                                               uint32_t cta_id) {
  uint32_t barrier_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

  asm volatile(
      "{\n"
      ".reg .b32 remAddr32;\n"
      "mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n"
      "}"
      :
      : "r"(barrier_ptr), "r"(cta_id));
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

template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
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

union GmmaDescriptor {
  __host__ __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}

  __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept
      : desc_(desc) {}

  __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const& t) noexcept
      : desc_(t.desc_) {}

  __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor&& t) noexcept
      : desc_(t.desc_) {}

  __host__ __device__ constexpr GmmaDescriptor& operator=(
      GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  __host__ __device__ constexpr GmmaDescriptor& operator=(
      GmmaDescriptor&& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, : 2;
    uint8_t : 1, base_offset_ : 3, : 4;
    uint8_t : 6, layout_type_ : 2;
  } bitfield;

  // Decay to an `uint64_t`
  __host__ __device__ constexpr operator uint64_t() const noexcept {
    return desc_;
  }
};

template <class PointerType>
__device__ GmmaDescriptor make_k_major_smem_desc(
    PointerType smem_ptr, int layout_type, int leading_byte_offset = 0,
    int stride_byte_offset = 1024) {
  GmmaDescriptor desc;
  auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = layout_type;
  desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
  desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.bitfield.base_offset_ = 0;
  return desc;
}

struct WGMMA {
  __device__ static void wgmma(
      uint64_t const& desc_a, uint64_t const& desc_b, float& d000, float& d001,
      float& d002, float& d003, float& d004, float& d005, float& d006,
      float& d007, float& d008, float& d009, float& d010, float& d011,
      float& d012, float& d013, float& d014, float& d015, float& d016,
      float& d017, float& d018, float& d019, float& d020, float& d021,
      float& d022, float& d023, float& d024, float& d025, float& d026,
      float& d027, float& d028, float& d029, float& d030, float& d031,
      float& d032, float& d033, float& d034, float& d035, float& d036,
      float& d037, float& d038, float& d039, float& d040, float& d041,
      float& d042, float& d043, float& d044, float& d045, float& d046,
      float& d047, float& d048, float& d049, float& d050, float& d051,
      float& d052, float& d053, float& d054, float& d055, float& d056,
      float& d057, float& d058, float& d059, float& d060, float& d061,
      float& d062, float& d063, float& d064, float& d065, float& d066,
      float& d067, float& d068, float& d069, float& d070, float& d071,
      float& d072, float& d073, float& d074, float& d075, float& d076,
      float& d077, float& d078, float& d079, float& d080, float& d081,
      float& d082, float& d083, float& d084, float& d085, float& d086,
      float& d087, float& d088, float& d089, float& d090, float& d091,
      float& d092, float& d093, float& d094, float& d095, float& d096,
      float& d097, float& d098, float& d099, float& d100, float& d101,
      float& d102, float& d103, float& d104, float& d105, float& d106,
      float& d107, float& d108, float& d109, float& d110, float& d111,
      float& d112, float& d113, float& d114, float& d115, float& d116,
      float& d117, float& d118, float& d119, float& d120, float& d121,
      float& d122, float& d123, float& d124, float& d125, float& d126,
      float& d127, bool scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " p,   1,   1,   0,   0;\n"
        "}\n"
        : "+f"(d000), "+f"(d001), "+f"(d002), "+f"(d003), "+f"(d004),
          "+f"(d005), "+f"(d006), "+f"(d007), "+f"(d008), "+f"(d009),
          "+f"(d010), "+f"(d011), "+f"(d012), "+f"(d013), "+f"(d014),
          "+f"(d015), "+f"(d016), "+f"(d017), "+f"(d018), "+f"(d019),
          "+f"(d020), "+f"(d021), "+f"(d022), "+f"(d023), "+f"(d024),
          "+f"(d025), "+f"(d026), "+f"(d027), "+f"(d028), "+f"(d029),
          "+f"(d030), "+f"(d031), "+f"(d032), "+f"(d033), "+f"(d034),
          "+f"(d035), "+f"(d036), "+f"(d037), "+f"(d038), "+f"(d039),
          "+f"(d040), "+f"(d041), "+f"(d042), "+f"(d043), "+f"(d044),
          "+f"(d045), "+f"(d046), "+f"(d047), "+f"(d048), "+f"(d049),
          "+f"(d050), "+f"(d051), "+f"(d052), "+f"(d053), "+f"(d054),
          "+f"(d055), "+f"(d056), "+f"(d057), "+f"(d058), "+f"(d059),
          "+f"(d060), "+f"(d061), "+f"(d062), "+f"(d063), "+f"(d064),
          "+f"(d065), "+f"(d066), "+f"(d067), "+f"(d068), "+f"(d069),
          "+f"(d070), "+f"(d071), "+f"(d072), "+f"(d073), "+f"(d074),
          "+f"(d075), "+f"(d076), "+f"(d077), "+f"(d078), "+f"(d079),
          "+f"(d080), "+f"(d081), "+f"(d082), "+f"(d083), "+f"(d084),
          "+f"(d085), "+f"(d086), "+f"(d087), "+f"(d088), "+f"(d089),
          "+f"(d090), "+f"(d091), "+f"(d092), "+f"(d093), "+f"(d094),
          "+f"(d095), "+f"(d096), "+f"(d097), "+f"(d098), "+f"(d099),
          "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104),
          "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109),
          "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114),
          "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119),
          "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124),
          "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
  }

  __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                               float* d, bool scale_d) {
    wgmma(desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
          d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18],
          d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28],
          d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38],
          d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48],
          d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58],
          d[59], d[60], d[61], d[62], d[63], d[64], d[65], d[66], d[67], d[68],
          d[69], d[70], d[71], d[72], d[73], d[74], d[75], d[76], d[77], d[78],
          d[79], d[80], d[81], d[82], d[83], d[84], d[85], d[86], d[87], d[88],
          d[89], d[90], d[91], d[92], d[93], d[94], d[95], d[96], d[97], d[98],
          d[99], d[100], d[101], d[102], d[103], d[104], d[105], d[106], d[107],
          d[108], d[109], d[110], d[111], d[112], d[113], d[114], d[115],
          d[116], d[117], d[118], d[119], d[120], d[121], d[122], d[123],
          d[124], d[125], d[126], d[127], scale_d);
  }

  static constexpr int kM = 64;
  static constexpr int kN = 256;
  static constexpr int kK = 16;
  static constexpr int kNumAccums = kM * kN / 128;
};

__device__ __forceinline__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg)::"memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
  static_assert(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
}
