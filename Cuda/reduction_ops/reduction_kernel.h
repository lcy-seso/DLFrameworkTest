#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <algorithm>
#include <stdexcept>
#include "stdio.h"

#define CHECK(call)                                                        \
  {                                                                        \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess) {                                            \
      printf("Error: %s: %d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason : %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                             \
    }                                                                      \
  }

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__device__ unsigned int count = 0;

inline bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

template <typename T>
__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, T val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

template <typename T, typename Reducer>
__device__ float power2Reduce(T val, int tid, T* shm, Reducer reducer,
                              int blockSize) {
  // This kernel function only works for power-of-2 arrays.
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = reducer(val, CudaShuffleDownSync(mask, val, 16));
  val = reducer(val, CudaShuffleDownSync(mask, val, 8));
  val = reducer(val, CudaShuffleDownSync(mask, val, 4));
  val = reducer(val, CudaShuffleDownSync(mask, val, 2));
  val = reducer(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);
  if (tid < warpSize) {
    val = shm[tid];
    val = reducer(val, CudaShuffleDownSync(mask, val, 16));
    val = reducer(val, CudaShuffleDownSync(mask, val, 8));
    val = reducer(val, CudaShuffleDownSync(mask, val, 4));
    val = reducer(val, CudaShuffleDownSync(mask, val, 2));
    val = reducer(val, CudaShuffleDownSync(mask, val, 1));
  }
  return val;
}

template <typename T, typename Reducer>
__device__ void blockReduce(unsigned int numElements, bool nIsPow2, const T* I,
                            T* O, Reducer reducer, int blockSize) {
  // This reduction kernel reduces all inputs with an arbitrary size in shared
  // memory.
  const int warpSize = 32;
  __shared__ T shm[warpSize];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T val = 0.;
  while (i < numElements) {
    val = reducer(val, I[i]);
    if (nIsPow2 || i + blockSize < numElements)
      val = reducer(val, I[i + blockSize]);
    i += gridSize;
  }
  __syncthreads();

  val = power2Reduce(val, tid, shm, reducer, blockSize);
  if (tid == 0) O[0] = val;
}

template <typename T, typename Reducer>
__global__ void multiBlockReduce(unsigned int numElements, bool nIsPow2,
                                 const T* I, T* O, Reducer reducer,
                                 int blockSize) {
  __shared__ float partialSum;
  blockReduce(numElements, nIsPow2, I, &partialSum, reducer, blockSize);
  if (threadIdx.x == 0) {
    // TODO(Ying) Current implementation only hard code for sum reduce. Need
    // re-implementation.
    atomicAdd(O, partialSum);
  }
}

template <typename T, typename Reducer>
void rowReduction(const T* I, T* O, int width, int height, Reducer reducer,
                  int maxThreads) {
  // This kernel is ONLY for reduce a 2-D matrix along row.
  int threads =
      (width * height < maxThreads * 2) ? nextPow2(width / 2) : maxThreads;
  int blocks = height;
}

template <typename T, typename Reducer>
void columnReduction(const T* I, T* O, int width, int height, Reducer reducer,
                     int maxThreads) {
  int threads =
      (width * height < maxThreads * 2) ? nextPow2(height / 2) : maxThreads;
  int blocks = width;
  // This kernel is ONLY for reduce a 2-D matrix along row.
}

template <typename T, typename Reducer>
void reduceToScalar(const T* I, T* O, int numElements, Reducer reducer,
                    int maxThreads, int maxBlocks) {
  int threads =
      (numElements < maxThreads * 2) ? nextPow2(numElements / 2) : maxThreads;
  int blocks = std::max(1, numElements / (threads * 2));
  blocks = min(maxBlocks, blocks);

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  multiBlockReduce<<<dimGrid, dimBlock, 0>>>(numElements, isPow2(numElements),
                                             I, O, reducer, threads);
}

template <typename T>
struct Sum {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a > b ? a : b;
  }
};

template <typename T, typename Reducer>
void ReduceImpl(const T* I, T* O, const std::vector<int>& axes, int in_rank,
                int in_dim0, int in_dim1, int in_dim2, int out_rank,
                Reducer reducer, int maxThreads, int maxBlocks) {
  if (out_rank == 0) {
    // reduction to a scalar
    reduceToScalar(I, O, in_dim0 * in_dim1 * in_dim2, reducer, maxThreads,
                   maxBlocks);
  } else if (in_rank == 2UL && out_rank == 1 && axes[0] == 0) {
    // row reduction.
    rowReduction(I, O, in_dim0, in_dim1, reducer, maxThreads);
  } else if (in_rank == 2UL && out_rank == 1 && axes[0] == 1) {
    // column reduction.
    columnReduction(I, O, in_dim0, in_dim1, reducer, maxThreads);
  } else {
    throw std::invalid_argument("Not implemented yet.");
  }
}

#endif
