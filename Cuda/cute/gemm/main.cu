#include <cublas_v2.h>
#include <cuda.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "gemm.cuh"

void Compare(const __half* data1, const __half* data2, int m, int n) {
  float threshold = 1e-3;
  for (int i = 0; i < m * n; ++i) {
    float v1 = __half2float(data1[i]);
    float v2 = __half2float(data2[i]);
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }
}

int main() {
  srand(10086);

  int m = 81920;
  int n = 256;
  int k = 256;

  using Element = __half;

  thrust::host_vector<Element> h_A(m * k);
  thrust::host_vector<Element> h_B(k * n);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    // h_A[i] = __float2half(i);
  }
  for (int i = 0; i < h_B.size(); ++i) {
    h_B[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    // h_A[i] = __float2half(i);
  }
  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B = h_B;

  thrust::device_vector<Element> d_C(m * n);
  thrust::fill(d_C.begin(), d_C.end(), static_cast<Element>(0.));

  thrust::device_vector<Element> d_C2(m * n);
  thrust::fill(d_C2.begin(), d_C2.end(), static_cast<Element>(0.));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}),
                              make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 32;

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);

  for (int i = 0; i < 100; ++i) {
    gemm_simple<Element, kTileM, kTileN, kTileK, MMA>
        <<<grid, block>>>(thrust::raw_pointer_cast(d_C.data()),
                          thrust::raw_pointer_cast(d_A.data()),
                          thrust::raw_pointer_cast(d_B.data()), m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret =
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                    thrust::raw_pointer_cast(d_B.data()), k,
                    thrust::raw_pointer_cast(d_A.data()), k, &beta,
                    thrust::raw_pointer_cast(d_C2.data()), n);

    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  thrust::host_vector<Element> h_C = d_C;
  thrust::host_vector<Element> h_C2 = d_C2;

  Compare(reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_C.data())),
          reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_C2.data())), m,
          n);

  using namespace cute;
  Tensor tensor_C = make_tensor(
      reinterpret_cast<cutlass::half_t*>(thrust::raw_pointer_cast(h_C.data())),
      make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(
      reinterpret_cast<cutlass::half_t*>(thrust::raw_pointer_cast(h_C2.data())),
      make_shape(m, n), make_stride(n, 1));

  // debug
  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}
