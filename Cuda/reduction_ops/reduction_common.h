#ifndef _REDUCE_COMMON_H_
#define _REDUCE_COMMON_H_

#include <assert.h>
#include <algorithm>
#include <stdexcept>
#include "stdio.h"

#include "reduction_kernel.cuh"

namespace {
inline bool IsPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

inline unsigned int NextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

inline unsigned int Log2Floor(unsigned int x) {
  if (x == 0) return -1U;
  int log = 0;
  unsigned int value = x;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    unsigned int n = value >> shift;
    if (n != 0) {
      value = n;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

template <typename T, typename X, typename Y>
inline T divup(const X x, const Y y) {
  return static_cast<T>((x + y - 1) / y);
}
}

template <typename T, typename Reducer>
void LaunchColumnReduction_LE16Cols(const T* I, T* O, int height, int width,
                                    Reducer reducer, T init_val, T scale) {
  int rows_per_warp = 32 / width;
  int threads = std::min(divup<int, int, int>(height, rows_per_warp), 32);

  // 32 blocks at most.
  int blocks =
      std::min(divup<int, int, int>(height, rows_per_warp * threads), 32);
  if (blocks > 2 && blocks < 32) {  // Aligns to the nearest power2 value.
    int log2 = Log2Floor(blocks);
    blocks = 1 << log2;
  }

  dim3 dimBlock(32, threads, 1);
  dim3 dimGrid(1, blocks, 1);

  printf("threads = %d, blocks = %d\n", threads, blocks);
  MultiBlockColumnReduceMax16ColumnsKernel<<<dimGrid, dimBlock>>>(
      I, O, height, width, reducer, init_val, scale);
}

template <typename T, typename Reducer>
void LaunchColumnReduction(const T* I, T* O, int height, int width,
                           Reducer reducer, int max_threads, T init_val,
                           T scale) {
  // This kernel is ONLY for reducing a 2-D matrix along column.
  // TODO(Ying) Optimized implementation is not finished yet.
  if (width <= 16)
    LaunchColumnReduction_LE16Cols(I, O, height, width, reducer, init_val,
                                   scale);
  else {
    // TODO(Ying) A simple kernel for column reduction.
    int threads = 128;  // This lanuch configuration is not optimal.
    int blocks = divup<int, int, int>(width, threads);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    ColumnReduceSimpleKernel<<<dimGrid, dimBlock, 0>>>(I, O, height, width,
                                                       reducer, scale);
  }
}

template <typename T, typename Reducer>
void LaunchRowReduction(const T* I, T* O, int height, int width,
                        Reducer reducer, int max_threads, T init_val, T scale) {
  // This kernel is ONLY for reduce a 2-D matrix along row.
  int threads = width < max_threads ? 1 << Log2Floor(width) : max_threads;
  threads = threads < 32 ? 32 : threads;
  int blocks = height;

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  printf("threads = %d, blocks = %d\n", threads, blocks);
  RowReduceKernel<<<dimGrid, dimBlock, 0>>>(I, O, height, width, threads,
                                            reducer, init_val, scale);
}

template <typename T, typename Reducer>
void LaunchScalarReduction(const T* I, T* O, int num_elements, Reducer reducer,
                           int max_threads, int max_blocks, T init_val,
                           T scale) {
  int threads = (num_elements < max_threads * 2) ? NextPow2(num_elements / 2)
                                                 : max_threads;
  int blocks = std::max(1, num_elements / (threads * 2));
  blocks = std::min(max_blocks, blocks);

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  MultiBlockScalarReduceKernel<<<dimGrid, dimBlock, 0>>>(
      num_elements, IsPow2(num_elements), I, O, reducer, threads, init_val,
      scale);
}

template <typename T, typename Reducer>
void ReduceImpl(const T* I, T* O, const std::vector<int>& axes, int in_rank,
                int in_dim0, int in_dim1, int in_dim2, int out_rank,
                Reducer reducer, int max_threads, int max_blocks, T init_val,
                T scale = 1) {
  if (out_rank == 0) {
    // reduction to a scalar
    LaunchScalarReduction(I, O, in_dim0 * in_dim1 * in_dim2, reducer,
                          max_threads, max_blocks, init_val, scale);
  } else if (in_rank == 2UL && out_rank == 1 && axes[0] == 0) {
    // Column reduction. Sums over all rows.
    LaunchColumnReduction(I, O, in_dim0, in_dim1, reducer, max_threads,
                          init_val, scale);
  } else if (in_rank == 2UL && out_rank == 1 && axes[0] == 1) {
    // Row reduction. Sums over all columns.
    LaunchRowReduction(I, O, in_dim0, in_dim1, reducer, max_threads, init_val,
                       scale);
  } else {
    throw std::invalid_argument("Not implemented yet.");
  }
}

#endif
