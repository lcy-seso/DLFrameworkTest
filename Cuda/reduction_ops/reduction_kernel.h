#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

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

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

template <unsigned int blockSize>
__device__ float reduceSum(float val, int tid, float* shm) {
  // This kernel function only works for power-of-2 arrays.
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);
  if (tid < warpSize) {
    val = shm[tid];
    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void blockReduceSum(unsigned int numElements, const float* I,
                               float* O) {
  // This reduction kernel reduces all inputs with an arbitrary size in shared
  // memory.
  const int warpSize = 32;
  __shared__ float shm[warpSize];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  float val = 0.;
  while (i < numElements) {
    val += I[i];
    if (nIsPow2 || i + blockSize < numElements) val += I[i + blockSize];
    i += gridSize;
  }
  __syncthreads();

  val = reduceSum<blockSize>(val, tid, shm);
  if (tid == 0) O[0] = val;
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void multiBlockReduceSum(unsigned int numElements, const float* I,
                                    float* O) {
  __shared__ float partialSum;
  blockReduceSum<blockSize, nIsPow2>(numElements, I, &partialSum);
  if (threadIdx.x == 0) atomicAdd(O, partialSum);
}

void reduceToScalar(int threads, int blocks, int numElements, float* I,
                    float* O) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(numElements)) {
    switch (threads) {
      case 512:
        multiBlockReduceSum<512, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 256:
        multiBlockReduceSum<256, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 128:
        multiBlockReduceSum<128, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 64:
        multiBlockReduceSum<64, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 32:
        multiBlockReduceSum<32, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 16:
        multiBlockReduceSum<16, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 8:
        multiBlockReduceSum<8, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                               O);
        break;
      case 4:
        multiBlockReduceSum<4, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                               O);
        break;
      case 2:
        multiBlockReduceSum<2, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                               O);
        break;
      case 1:
        multiBlockReduceSum<1, true><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                               O);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        multiBlockReduceSum<512, false><<<dimGrid, dimBlock, 0>>>(numElements,
                                                                  I, O);
        break;
      case 256:
        multiBlockReduceSum<256, false><<<dimGrid, dimBlock, 0>>>(numElements,
                                                                  I, O);
        break;
      case 128:
        multiBlockReduceSum<128, false><<<dimGrid, dimBlock, 0>>>(numElements,
                                                                  I, O);
        break;
      case 64:
        multiBlockReduceSum<64, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 32:
        multiBlockReduceSum<32, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 16:
        multiBlockReduceSum<16, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                 O);
        break;
      case 8:
        multiBlockReduceSum<8, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 4:
        multiBlockReduceSum<4, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 2:
        multiBlockReduceSum<2, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
      case 1:
        multiBlockReduceSum<1, false><<<dimGrid, dimBlock, 0>>>(numElements, I,
                                                                O);
        break;
    }
  }
}

#endif
