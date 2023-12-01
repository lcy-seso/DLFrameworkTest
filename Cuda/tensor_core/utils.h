#pragma once

#include <iomanip>

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

#define RM(i, j, row, col) i* col + j
#define CM(i, j, row, col) i + j* row

template <typename T>
void PrintMatrix(const T* data, size_t rows, size_t cols, bool row_major) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "MM ref:" << std::endl;
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "[" << i << "]:\t";
    for (size_t j = 0; j < cols - 1; ++j) {
      if (row_major) {
        std::cout << data[RM(i, j, rows, cols)] << ", ";
      } else {
        std::cout << data[CM(i, j, rows, cols)] << ", ";
      }
    }
    if (row_major) {
      std::cout << data[RM(i, cols - 1, rows, cols)] << std::endl;
    } else {
      std::cout << data[CM(i, cols - 1, rows, cols)] << std::endl;
    }
  }
}

template <>
void PrintMatrix(const __half* data, size_t rows, size_t cols, bool row_major) {
  std::cout << std::fixed << std::setprecision(2);
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "[" << i << "]:\t";
    for (size_t j = 0; j < cols - 1; ++j) {
      if (row_major) {
        std::cout << __half2float(data[RM(i, j, rows, cols)]) << ", ";
      } else {
        std::cout << __half2float(data[CM(i, j, rows, cols)]) << ", ";
      }
    }
    if (row_major) {
      std::cout << __half2float(data[RM(i, cols - 1, rows, cols)]) << std::endl;
    } else {
      std::cout << __half2float(data[CM(i, cols - 1, rows, cols)]) << std::endl;
    }
  }
}

void MmaRef(__half* hA, __half* hB, float* hC, int M, int N, int K) {
  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        hC[RM(i, j, M, N)] +=
            __half2float(hA[RM(i, k, M, K)]) * __half2float(hB[CM(k, j, K, N)]);
      }
    }
  }

  PrintMatrix(hC, M, N, true);
}
