#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

#define THRESHOLD 64

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s: %d, ", __FILE__, __LINE__);                           \
      printf("code: %d, reason : %s\n", error, cudaGetErrorString(error));     \
      exit(1);                                                                 \
    }                                                                          \
  }

__device__ __forceinline__ float sumSingleWarp(float val) {
  unsigned mask = 0xffffffff;
  val += __shfl_down_sync(mask, val, 16);
  val += __shfl_down_sync(mask, val, 8);
  val += __shfl_down_sync(mask, val, 4);
  val += __shfl_down_sync(mask, val, 2);
  val += __shfl_down_sync(mask, val, 1);
  return val;
}

__device__ __forceinline__ float maxSingleWarp(float val) {
  unsigned mask = 0xffffffff;
  val = max(__shfl_down_sync(mask, val, 16), val);
  val = max(__shfl_down_sync(mask, val, 8), val);
  val = max(__shfl_down_sync(mask, val, 4), val);
  val = max(__shfl_down_sync(mask, val, 2), val);
  return max(__shfl_down_sync(mask, val, 1), val);
}

__device__ __forceinline__ void findMax(float *I, float *dfMax_s, int blockSize,
                                        int base, int curIdx, int nextIdx,
                                        int dimN, float *max) {
  dfMax_s[base] = -1.0e20;
  while (curIdx < dimN) {
    if (dfMax_s[base] < I[nextIdx]) {
      dfMax_s[base] = I[nextIdx];
    }
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  for (int stride = blockSize >> 1; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (base < stride) {
      nextIdx = base + stride;
      if (dfMax_s[base] < dfMax_s[nextIdx]) {
        dfMax_s[base] = dfMax_s[nextIdx];
      }
    }
  }

  float val = dfMax_s[base];
  val = maxSingleWarp(val);
  if (0 == base)
    max[0] = dfMax_s[0];

  __syncthreads();
}

__device__ __forceinline__ void subMaxAndExp(float *I, float *O, int curIdx,
                                             int nextIdx, int blockSize,
                                             int dimN, float max) {
  float val;
  while (curIdx < dimN) {
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

__device__ __forceinline__ void valueSum(float *O, float *dfMax_s,
                                         int blockSize, int base, int curIdx,
                                         int nextIdx, int dimN) {
  dfMax_s[base] = 0.;
  while (curIdx < dimN) {
    dfMax_s[base] += O[nextIdx];
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  for (int stride = blockSize >> 1; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (base < stride) {
      nextIdx = base + stride;
      dfMax_s[base] += dfMax_s[nextIdx];
    }
    __syncthreads();
  }

  float val = dfMax_s[base];
  val = sumSingleWarp(val);
  if (base == 0)
    dfMax_s[0] = val;

  __syncthreads();
}

__device__ __forceinline__ void divSum(float *O, float sum, int curIdx,
                                       int nextIdx, int blockSize, int dimN) {
  while (curIdx < dimN) {
    O[nextIdx] /= sum;
    nextIdx += blockSize;
    curIdx += blockSize;
  }
}

__device__ __forceinline__ void softmax(float *I, float *O, float *dfMax_s,
                                        int blockSize, int base, int curIdx,
                                        int nextIdx, int dimN) {
  __shared__ float max;

  // find the max number
  findMax(I, dfMax_s, blockSize, base, curIdx, nextIdx, dimN, &max);

  // sub max Value and do Exp operation
  subMaxAndExp(I, O, base, nextIdx, blockSize, dimN, max);

  // add dimN values into blockDim.x buffer
  // sum is in dfMax_s[0]
  valueSum(O, dfMax_s, blockSize, base, curIdx, nextIdx, dimN);

  // divided by sum
  divSum(O, dfMax_s[0], curIdx, nextIdx, blockSize, dimN);
}

__global__ void KeMatrixSoftMax(float *O, float *I, const size_t dimN) {
  extern __shared__ float dfMax_s[];

  int blockSize = blockDim.x;
  int base = threadIdx.x;
  int nextIdx = blockIdx.x * dimN + base;
  int curIdx = base;

  softmax(I, O, dfMax_s, blockSize, base, curIdx, nextIdx, dimN);
}

void printResult(const float *mat, size_t height, size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j)
      printf("%d ", mat[i * width + j]);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  const size_t kMatHeight = 2;
  const size_t kMatWidth = 16;

  const size_t mat_size = kMatHeight * kMatWidth;

  srand(0);

  float *h_a, *h_b;

  cudaMallocHost((void **)&h_a, sizeof(float) * mat_size);
  cudaMallocHost((void **)&h_b, sizeof(float) * mat_size);

  // random initialization of matrix A.
  for (size_t i = 0; i < mat_size; ++i)
    h_a[i] = ((float)rand()) / (float)RAND_MAX;
  // initialize memory that stores computation result to all zeros;
  memset(h_b, 0., sizeof(float) * mat_size);

  // events to count the execution time.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate memory space on the device.
  float *d_a, *d_b;
  cudaMalloc((void **)&d_a, sizeof(float) * mat_size);
  cudaMalloc((void **)&d_b, sizeof(float) * mat_size);

  // copy matrix A from host to device memory
  CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * mat_size, cudaMemcpyHostToDevice));

  // start to count execution time. use the default stream.
  cudaEventRecord(start);
  int block_num =
      kMatWidth > 512
          ? 512
          : pow(2, static_cast<int>(log2(static_cast<float>(kMatWidth))));

  dim3 block(block_num, 1);
  dim3 grid(kMatHeight, 1);
  KeMatrixSoftMax<<<grid, block, block_num * sizeof(float)>>>(d_b, d_a,
                                                              kMatWidth);

  cudaEventRecord(stop);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaMemcpy(h_b, d_b, sizeof(float) * mat_size, cudaMemcpyDeviceToHost));

  float kernel_elapsed_time;
  cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
  printf("kernel execution time elapse : %d\n", kernel_elapsed_time);

  printResult(h_b, kMatHeight, kMatWidth);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);

  return 0;
}
