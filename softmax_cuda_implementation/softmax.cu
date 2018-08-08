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

void printResult(const float *mat, size_t height, size_t width) {
  printf("matrix : (%d, %d)\n", height, width);
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j)
      printf("%f ", mat[i * width + j]);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  const size_t kMatHeight = 2;
  const size_t kMatWidth = 317;
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

  KeMatrixSoftMax<<<grid, block, 0>>>(d_b, d_a, kMatWidth);

  cudaEventRecord(stop);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaMemcpy(h_b, d_b, sizeof(float) * mat_size, cudaMemcpyDeviceToHost));

  float kernel_elapsed_time;
  cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
  printf("kernel execution time elapse : %f\n", kernel_elapsed_time);

  printResult(h_b, kMatHeight, kMatWidth);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);

  return 0;
}
