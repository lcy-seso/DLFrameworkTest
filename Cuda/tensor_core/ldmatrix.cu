#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

template <bool transpose = false>
__device__ int ldsm(void* addr) {
  unsigned int addr_int = __cvta_generic_to_shared(addr);
  int r;

  if (!transpose) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr_int));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                 : "=r"(r)
                 : "r"(addr_int));
  }

  return r;
}

__global__ void foo() {
  __shared__ half data[8 * 8];

  int idx = threadIdx.x;
  if (idx == 0) {
    for (int i = 0; i < 8 * 8; ++i) {
      data[i] = i;
    }
  }
  __syncthreads();

  void* addr = (char*)data + (idx % 8) * 16;
  // for .x1 only T0-T7 is response for the address
  // but the other thread must specify a readable address
  if (idx >= 8) {
    addr = data;
  }

  {
    int r = ldsm<false>(addr);

    __shared__ half data_out[8 * 8];

    data_out[idx * 2] = ((half*)(&r))[0];
    data_out[idx * 2 + 1] = ((half*)(&r))[1];

    __syncthreads();

    if (idx == 0) {
      printf("non-transpose:\n");
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          printf("%.1f ", __half2float(data_out[i * 8 + j]));
        }
        printf("\n");
      }
    }
  }

  {
    int r = ldsm<true>(addr);

    __shared__ half data_out[8 * 8];

    data_out[idx * 2] = ((half*)(&r))[0];
    data_out[idx * 2 + 1] = ((half*)(&r))[1];

    __syncthreads();

    if (idx == 0) {
      printf("transpose:\n");
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          printf("%.1f ", __half2float(data_out[i * 8 + j]));
        }
        printf("\n");
      }
    }
  }
}

int main() {
  foo<<<1, 32>>>();

  cudaDeviceSynchronize();
}
