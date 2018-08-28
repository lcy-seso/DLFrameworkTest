#include "reduction_kernel.h"

int main(int argc, char* argv[]) {
  std::vector<int> kTensorShape = {13, 33, 11};
  std::vector<int> axes = {1};
  int out_rank = 0;

  int product = std::accumulate(kTensorShape.begin(), kTensorShape.end(), 1,
      std::multiplies<int>());
  printf("reduce %d numbers.\n", product);

  const int kMaxThreads = 512;
  const int kMaxBlocks = 64;

  srand(0);
  float *h_a, *h_b;

  cudaMallocHost((void**)&h_a, sizeof(float) * product);
  // TODO(ying) The current output size is the same as the input which is not
  // correct.
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
  int in_dim0 = kTensorShape[0];
  int in_dim1 = kTensorShape.size() > 1 ? kTensorShape[1] : 1;
  int in_dim2 = kTensorShape.size() > 2 ? kTensorShape[2] : 1;
  // ReduceImpl<float, Sum<float>>(d_a, d_b, axes, kTensorShape.size(), in_dim0,
  //                               in_dim1, in_dim2, out_rank, Sum<float>(),
  //                               kMaxThreads, kMaxBlocks);
  ReduceImpl<float, Max<float>>(d_a, d_b, axes, kTensorShape.size(), in_dim0,
      in_dim1, in_dim2, out_rank, Max<float>(),
      kMaxThreads, kMaxBlocks);

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
