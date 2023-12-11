#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define N1 8 * 8

__global__ void LdmatrixX1(bool trans) {
  __shared__ half data[N1];

  int tid = threadIdx.x;

  if (threadIdx.x == 0) {
    for (size_t i = 0; i < N1; ++i) {
      data[i] = __float2half(float(i));
    }
  }
  __syncthreads();

  uint32_t r;
  uint32_t addr = __cvta_generic_to_shared((char*)data + (tid % 8) * 16);
  if (tid > 8) addr = __cvta_generic_to_shared(data);

  if (trans) {
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr));
  }
  __syncthreads();

  half* data_ptr = (half*)(&r);
  printf("[%d], %.0f, %.0f\n", tid, __half2float(data_ptr[0]),
         __half2float(data_ptr[1]));

  __shared__ half data_out[N1];
  data_out[tid * 2] = data_ptr[0];
  data_out[tid * 2 + 1] = data_ptr[1];

  if (tid == 0) {
    printf("\n");
    for (size_t i = 0; i < 8; ++i) {
      for (size_t j = 0; j < 7; ++j) {
        printf("%.0f, ", __half2float(data_out[i * 8 + j]));
      }
      printf("%.0f\n", __half2float(data_out[(i + 1) * 8 - 1]));
    }
  }
}

#define N2 8 * 16
__global__ void LdmatrixX2(bool trans) {  // column major
  __shared__ half data[N2];

  int tid = threadIdx.x;

  if (threadIdx.x == 0) {
    for (size_t i = 0; i < N2; ++i) {
      data[i] = __float2half(float(i));
    }
  }
  __syncthreads();

  uint32_t r;
  uint32_t addr = __cvta_generic_to_shared((char*)data + (tid % 8) * 16);
  if (tid > 8) addr = __cvta_generic_to_shared(data);

  if (trans) {
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr));
  }
  __syncthreads();

  half* data_ptr = (half*)(&r);
  printf("[%d], %.0f, %.0f\n", tid, __half2float(data_ptr[0]),
         __half2float(data_ptr[1]));

  __shared__ half data_out[N2];
  data_out[tid * 2] = data_ptr[0];
  data_out[tid * 2 + 1] = data_ptr[1];

  if (tid == 0) {
    printf("\n");
    for (size_t i = 0; i < 8; ++i) {
      for (size_t j = 0; j < 7; ++j) {
        printf("%.0f, ", __half2float(data_out[i * 8 + j]));
      }
      printf("%.0f\n", __half2float(data_out[(i + 1) * 8 - 1]));
    }
  }
}

int main() {
  // LdmatrixX1<<<1, 32>>>(false);
  LdmatrixX2<<<1, 32>>>(false);

  cudaDeviceSynchronize();
}
