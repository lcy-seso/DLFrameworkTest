#include "kernel.cuh"

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
