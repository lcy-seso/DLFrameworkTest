#pragma once

#include <iostream>

template <int N, int D>
constexpr int CeilDiv = (N + D - 1) / D;

float rand_float(float a = 1e-3, float b = 1) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

__forceinline__ unsigned int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename DType>
bool check_results(const DType* h_src, const DType* h_dst, int kNumel) {
  // Check if h_src and h_dst are the same
  bool match = true;
  for (int i = 0; i < kNumel; ++i) {
    if (h_src[i] != h_dst[i]) {
      std::cerr << "Verification failed: Mismatch found at index " << i
                << ": h_src[" << i << "] = " << h_src[i] << ", h_dst[" << i
                << "] = " << h_dst[i] << std::endl;
      match = false;
      break;  // Stop at the first mismatch
    }
  }

  if (match) {
    std::cout
        << "Verification successful: h_src and h_dst contain the same values."
        << std::endl;
  } else {
    std::cerr << "Verification failed: h_src and h_dst differ." << std::endl;
    return 1;
  }

#if 0
  int start = 256;
  int end = start + 64;
  printf("\nsrc:\n");
  for (int i = start; i < end; ++i) {
    printf("%.3f, ", h_src[i]);
    if (i && (i + 1) % 16 == 0) printf("\n");
  }

  printf("\n\ndst:\n");
  for (int i = start; i < end; ++i) {
    printf("%.3f, ", h_dst[i]);
    if (i && (i + 1) % 16 == 0) printf("\n");
  }
#endif
}
