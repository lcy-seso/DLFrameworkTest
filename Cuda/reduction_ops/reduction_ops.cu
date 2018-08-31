#include <limits>

#include "reduction_common.h"

int getOutputSize(const std::vector<int>& tensor_shape,
                  const std::vector<int>& axes) {
  if (axes.empty()) {
    return 1;
  } else if (tensor_shape.size() == 2 && axes[0] == 0U) {
    return tensor_shape[1];
  } else if (tensor_shape.size() == 2 && axes[0] == 1U) {
    return tensor_shape[0];
  } else
    throw std::invalid_argument("Not implemented yet.");
}

int main(int argc, char* argv[]) {
  std::vector<int> kTensorShape = {757, 3};
  std::vector<int> axes = {0};
  int out_rank = 1;

  int product = std::accumulate(kTensorShape.begin(), kTensorShape.end(), 1,
                                std::multiplies<int>());
  printf("reduce %d numbers.\n", product);

  const int kMaxThreads = 512;
  const int kMaxBlocks = 64;

  srand(0);
  float *h_a, *h_b;

  cudaMallocHost((void**)&h_a, sizeof(float) * product);
  int out_size = getOutputSize(kTensorShape, axes);

  int out_size_tmp = out_size;
  out_size = 32 * 16;  // hard code for tests.
  cudaMallocHost((void**)&h_b, sizeof(float) * out_size);
  out_size = out_size_tmp;

  // random initialization of matrix A.
  for (size_t i = 0; i < product; ++i) h_a[i] = static_cast<float>(i + 1);

  // initialize memory that stores computation result to all zeros;
  memset(h_b, 0., sizeof(float) * out_size);

  // events to count the execution time.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate memory space on the device.
  float *d_a, *d_b;
  cudaMalloc((void**)&d_a, sizeof(float) * product);
  cudaMalloc((void**)&d_b, sizeof(float) * out_size);

  // copy matrix A from host to device memory
  CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * product, cudaMemcpyHostToDevice));

  // start to count execution time. use the default stream.
  cudaEventRecord(start);

  // lanuch kernel.
  int in_dim0 = kTensorShape[0];
  int in_dim1 = kTensorShape.size() > 1 ? kTensorShape[1] : 1;
  int in_dim2 = kTensorShape.size() > 2 ? kTensorShape[2] : 1;

  float init_val = 0.;
  float scale = 1. / in_dim0;
  ReduceImpl<float, Sum<float>>(d_a, d_b, axes, kTensorShape.size(), in_dim0,
                                in_dim1, in_dim2, out_rank, Sum<float>(),
                                kMaxThreads, kMaxBlocks, init_val, scale);

  // float init_val = std::numeric_limits<float>::min();
  // ReduceImpl<float, Max<float>>(d_a, d_b, axes, kTensorShape.size(), in_dim0,
  //     in_dim1, in_dim2, out_rank, Max<float>(),
  //     kMaxThreads, kMaxBlocks, init_val);

  // float init_val = 1.;
  // ReduceImpl<float, Prod<float>>(d_a, d_b, axes, kTensorShape.size(),
  // in_dim0,
  //     in_dim1, in_dim2, out_rank, Prod<float>(),
  //     kMaxThreads, kMaxBlocks, init_val);

  // float init_val = std::numeric_limits<float>::max();
  // ReduceImpl<float, Min<float>>(d_a, d_b, axes, kTensorShape.size(), in_dim0,
  //                               in_dim1, in_dim2, out_rank, Min<float>(),
  //                               kMaxThreads, kMaxBlocks, init_val);

  cudaEventRecord(stop);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaMemcpy(h_b, d_b, sizeof(float) * out_size, cudaMemcpyDeviceToHost));

  float kernel_elapsed_time;
  cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
  printf("kernel execution time elapse : %f\n", kernel_elapsed_time);

  for (size_t i = 0; i < out_size; ++i) printf("[%d] :\t%.4f\n", i, h_b[i]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);

  return 0;
}
