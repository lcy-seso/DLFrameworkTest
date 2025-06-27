#pragma once

enum class LayoutType : uint8_t {
  INTERLEAVE = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

__host__ __device__ __forceinline__ char const* to_string(LayoutType const& t) {
  switch (t) {
    case LayoutType::INTERLEAVE:
      return "INTERLEAVE";
    case LayoutType::B128:
      return "B128";
    case LayoutType::B64:
      return "B64";
    case LayoutType::B32:
      return "B32";
  }
  return nullptr;
}

// cutlass implementation for reference:
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_desc.hpp#L107
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

  // this is c++ biefield.
  // The useage is like this: ^type    ^field_name ^bit_count
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;  // 14 bits [0,14), 2 bits unused

    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, : 2;

    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    // 1 - An  unnamed field using i bit (reserved)
    // base_offset_ : 3 - A named fielf using 3 bits [0,3)
    // : 4 - Another unnamed field using 4 bits (reserved)
    uint8_t : 1, base_offset_ : 3, : 4;

    // layout type, bit [61,64)
    // SWIZZLE_NONE matrix descriptor = 0,
    // SWIZZLE_128B_BASE32B = 1,
    // SWIZZLE_128B matrix descriptor = 2,
    // SWIZZLE_128B_BASE32B = 1,
    // SWIZZLE_64B descriptor = 4,
    // SWIZZLE_32B descriptor = 6,
    // N/A = 3, N/A = 5, N/A = 7
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

__host__ __device__ void print(GmmaDescriptor const& t) {
#if !defined(__CUDACC_RTC__)
  printf("GmmaDescriptor: 0x%016llx\n",
         static_cast<unsigned long long>(t.desc_));
  printf("  start_addr :  0x%04x\n", t.bitfield.start_address_);
  printf("  leading_off:  0x%04x (%d)\n", t.bitfield.leading_byte_offset_,
         t.bitfield.leading_byte_offset_);
  printf("  stride_off :  0x%04x (%d)\n", t.bitfield.stride_byte_offset_,
         t.bitfield.stride_byte_offset_);
  printf("  base_offset:  0x%01x\n", t.bitfield.base_offset_);
  printf("  layout_type:  0x%01x (%s)\n", t.bitfield.layout_type_,
         to_string(static_cast<LayoutType>(t.bitfield.layout_type_)));
#endif
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

template <typename DType>
__device__ void compute_wgmma_stage(uint32_t s, float* accum, DType* smem_a[],
                                    DType* smem_b[], uint32_t kTK,
                                    bool scale_d = true) {
  const uint32_t warp_group_idx = get_warp_group_idx();

#pragma unroll
  for (int i = 0; i < WGMMA::kNumAccums; ++i) {
    warpgroup_fence_operand(accum[i]);
  }
  warpgroup_arrive();

  const auto smem_a_warp_group_offset = warp_group_idx * WGMMA::kM * kTK;
  auto desc_a = make_k_major_smem_desc(smem_a[s] + smem_a_warp_group_offset, 1);
  auto desc_b = make_k_major_smem_desc(smem_b[s], 1);
  WGMMA::wgmma(desc_a, desc_b, accum, scale_d);

#pragma unroll
  for (int k = 1; k < kTK / WGMMA::kK; ++k) {
    auto desc_a = make_k_major_smem_desc(
        smem_a[s] + k * WGMMA::kK + smem_a_warp_group_offset, 1);
    auto desc_b = make_k_major_smem_desc(smem_b[s] + k * WGMMA::kK, 1);
    WGMMA::wgmma(desc_a, desc_b, accum, true);
  }
  warpgroup_commit_batch();

#pragma unroll
  for (int i = 0; i < WGMMA::kNumAccums; ++i) {
    warpgroup_fence_operand(accum[i]);
  }
  warpgroup_wait<0>();
}
