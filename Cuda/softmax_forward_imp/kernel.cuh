#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

#define THRESHOLD 64
#define CUDA_VERSION 9000

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s: %d, ", __FILE__, __LINE__);                           \
      printf("code: %d, reason : %s\n", error, cudaGetErrorString(error));     \
      exit(1);                                                                 \
    }                                                                          \
  }

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate)                                      \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float
CudaShuffleDownSync(unsigned mask, float val, int delta, int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ float reduceMax(float val, int tid, int blockSize, float *shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize)
    shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0)
    shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ __forceinline__ void findMax(float *I, float *shm, int blockSize,
                                        int base, int curIdx, int nextIdx,
                                        int matWidth) {
  float val = -1.e20;

  while (curIdx < matWidth) {
    if (val < I[nextIdx])
      val = I[nextIdx];
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  val = reduceMax(val, base, blockSize, shm);

  if (0 == base)
    shm[0] = val;

  __syncthreads();
}

__device__ __forceinline__ void subMaxAndExp(float *I, float *O, int curIdx,
                                             int nextIdx, int blockSize,
                                             int matWidth, float max) {
  float val = 0.;
  while (curIdx < matWidth) {
    val = I[nextIdx] - max;
    if (val < -THRESHOLD)
      val = -THRESHOLD;
    I[nextIdx] = val;
    O[nextIdx] = exp(val);

    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();
}

__device__ float reduceSum(float val, int tid, int blockSize, float *shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize)
    shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0)
    shm[tid / warpSize] = val;

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

__device__ __forceinline__ void valueSum(float *O, float *shm, int blockSize,
                                         int base, int curIdx, int nextIdx,
                                         int matWidth) {
  float val = 0.;
  while (curIdx < matWidth) {
    val += O[nextIdx];
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  val = reduceSum(val, base, blockSize, shm);
  if (base == 0)
    shm[0] = val;

  __syncthreads();
}

__device__ __forceinline__ void divSum(float *O, float sum, int curIdx,
                                       int nextIdx, int blockSize,
                                       int matWidth) {
  while (curIdx < matWidth) {
    O[nextIdx] /= sum;
    nextIdx += blockSize;
    curIdx += blockSize;
  }
}

__device__ __forceinline__ void softmax(float *I, float *O, int blockSize,
                                        int base, int curIdx, int nextIdx,
                                        int matWidth) {
  const int warpSize = 32;
  __shared__ float shm[warpSize];

  // find the max number, max value is stored in shm[0]
  findMax(I, shm, blockSize, base, curIdx, nextIdx, matWidth);

  // sub max Value and do Exp operation
  subMaxAndExp(I, O, base, nextIdx, blockSize, matWidth, shm[0]);

  // add matWidth values into blockDim.x buffer, sum is in shm[0]
  valueSum(O, shm, blockSize, base, curIdx, nextIdx, matWidth);

  // divided by sum
  divSum(O, shm[0], curIdx, nextIdx, blockSize, matWidth);
}

__global__ void KeMatrixSoftMax(float *O, float *I, const size_t matWidth) {
  int blockSize = blockDim.x;
  int base = threadIdx.x;
  int nextIdx = blockIdx.x * matWidth + base;
  int curIdx = base;

  softmax(I, O, blockSize, base, curIdx, nextIdx, matWidth);
}

