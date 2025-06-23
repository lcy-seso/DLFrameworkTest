#pragma once

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

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

int GetMaxSharedMemoryPerBlock() {
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
  return prop.sharedMemPerBlock;
}

template <int N, int D>
constexpr int CeilDiv = (N + D - 1) / D;

__forceinline__ unsigned int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename DType, const int kM, const int kN>
__forceinline__ __host__ __device__ void print_values(const DType* tensor,
                                                      int start = 256,
                                                      int cutoff = 128) {
  printf("\n");
  for (int i = start; i < kM * kN; ++i) {
    printf("%.0f, ", static_cast<float>(tensor[i]));
    if ((i + 1) % 16 == 0) printf("\n");
    if (i == (start + cutoff - 1)) break;
  }
  printf("\n");
}

float rand_float(float a = 1e-3, float b = 1) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

float rand_normal(float mean = 0.0f, float stddev = 1.0f) {
  // Box-Muller transform to generate random numbers with Normal distribution
  float u1 = ((float)rand()) / (float)RAND_MAX;
  float u2 = ((float)rand()) / (float)RAND_MAX;

  // Avoid log(0) by ensuring u1 is not zero
  if (u1 < 1e-10f) u1 = 1e-10f;

  // Compute Gaussian random value
  float r = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);

  // Scale and shift to get desired mean and standard deviation
  return mean + stddev * r;
}

template <typename DType>
int check_results(const DType* value1, const DType* value2, int kNumel) {
  // Check if h_src and h_dst are the same
  bool match = true;
  for (int i = 0; i < kNumel; ++i) {
    if (value1[i] != value2[i]) {
      std::cerr << "Verification failed: Mismatch found at index " << i
                << ": value1[" << i << "] = " << value1[i] << ", value2[" << i
                << "] = " << value2[i] << std::endl;
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
  std::cout << "\nsrc:\n";
  for (int i = start; i < end; ++i) {
    std::cout << std::fixed << std::setprecision(3) << value1[i] << ", ";
    if (i && (i + 1) % 16 == 0) std::cout << "\n";
  }

  std::cout << "\n\ndst:\n";
  for (int i = start; i < end; ++i) {
    std::cout << std::fixed << std::setprecision(3) << value2[i] << ", ";
    if (i && (i + 1) % 16 == 0) std::cout << "\n";
  }
#endif

  return 0;
}
