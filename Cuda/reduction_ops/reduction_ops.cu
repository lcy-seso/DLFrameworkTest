#include "reduction_kernel.h"
#include "reduction_ops.h"

int main(int argc, char* argv[]) {
  std::vector<size_t> kTensorShape = {23, 13, 11};
  size_t product = std::accumulate(kTensorShape.begin(), kTensorShape.end(), 1,
                                   std::multiplies<size_t>());
  printf("reduce %d numbers.\n", static_cast<int>(product));

  const int kMaxThreads = 512;
  const int kMaxBlocks = 64;

  srand(0);

  float *h_a, *h_b;

  cudaMallocHost((void**)&h_a, sizeof(float) * product);
  cudaMallocHost((void**)&h_b, sizeof(float) * product);

  // random initialization of matrix A.
  for (size_t i = 0; i < product; ++i) h_a[i] = static_cast<float>(i + 1);

  // initialize memory that stores computation result to all zeros;
  memset(h_b, 0., sizeof(float) * product);

  // events to count the execution time.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate memory space on the device.
  float *d_a, *d_b;
  cudaMalloc((void**)&d_a, sizeof(float) * product);
  cudaMalloc((void**)&d_b, sizeof(float) * product);

  // copy matrix A from host to device memory
  CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * product, cudaMemcpyHostToDevice));

  // start to count execution time. use the default stream.
  cudaEventRecord(start);

  // lanuch kernel.
  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(product, kMaxBlocks, kMaxThreads, numBlocks,
                         numThreads);
  printf("numThreads = %d, numBlocks = %d\n", numThreads, numBlocks);

  reduceToScalar(numThreads, numBlocks, product, d_a, d_b);

  cudaEventRecord(stop);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaMemcpy(h_b, d_b, sizeof(float) * product, cudaMemcpyDeviceToHost));

  float kernel_elapsed_time;
  cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
  printf("kernel execution time elapse : %f\n", kernel_elapsed_time);

  printf("reduced sum = %.2f\n", h_b[0]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);

  return 0;
}
