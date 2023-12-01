#pragma once

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

template <typename T>
__global__ void PrintValueHost(const T* data, int rows, int cols) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.3f, ", data[i * cols + j]);
      }
      printf("\n");
    }
  }
}

template <>
__global__ void PrintValueHost(const cutlass::half_t* data, int rows,
                               int cols) {
  __half* tmp = reinterpret_cast<__half*>(const_cast<cutlass::half_t*>(data));

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("Input:\n");
    for (int i = 0; i < rows; ++i) {
      printf("[%d]:\t", i);
      for (int j = 0; j < cols - 1; ++j)
        printf("%.3f, ", __half2float(tmp[i * cols + j]));

      printf("%.3f\n", __half2float(tmp[(i + 1) * cols - 1]));
    }
    printf("\n\n");
  }
}

bool CheckResult(const __half* v1, const __half* v2, int64_t numel) {
  float epsilon = 1e-5;

  for (int i = 0; i < numel; ++i) {
    if (abs(__half2float(v1[i]) - __half2float(v2[i])) > epsilon) {
      std::cout << "v1[" << i << "] vs. v2[" << i
                << "] = " << __half2float(v1[i]) << " vs. "
                << __half2float(v2[i]) << std::endl;
      return false;
    }
  }
  return true;
}
