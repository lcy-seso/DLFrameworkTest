
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.h"

__global__ void TestMmaInstruction(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   half* __restrict__ C, size_t M, size_t N,
                                   size_t K) {
  const size_t K_tiles = CEIL_DIV(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  __shared__ half A_smem[MMA_M][MMA_K];
  __shared__ half B_smem[MMA_N][MMA_K];
  __shared__ half C_smem[MMA_M][MMA_N];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RA[4];
  uint32_t RB[2];
  uint32_t RC[2] = {0, 0};

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    *((int4*)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
        *((int4*)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

    if (lane_id < MMA_N * 2) {
      *((int4*)(&B_smem[lane_id / 2][0]) + lane_id % 2) = *(
          (int4*)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
    }
    __syncthreads();

    uint32_t A_smem_lane_addr =
        __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);

    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
        : "r"(A_smem_lane_addr));

    // half* rA = (half*)(RA);
    // printf("[%d]: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",
    //        threadIdx.x, __half2float(rA[0]), __half2float(rA[1]),
    //        __half2float(rA[2]), __half2float(rA[3]), __half2float(rA[4]),
    //        __half2float(rA[5]), __half2float(rA[6]), __half2float(rA[7]));

    uint32_t B_smem_lane_addr =
        __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(RB[0]), "=r"(RB[1])
                 : "r"(B_smem_lane_addr));

    // half* rB = (half*)(RB);
    // printf("[%d]: %.0f, %.0f, %.0f, %.0f\n", threadIdx.x,
    // __half2float(rB[0]),
    //        __half2float(rB[1]), __half2float(rB[2]), __half2float(rB[3]));

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "
        "%4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(RC[0]), "=r"(RC[1])
        : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]),
          "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));

    __syncthreads();
  }

  half* rC = (half*)(RC);
  printf("[%d]: %.2f, %.2f, %.2f, %.2f\n", threadIdx.x, __half2float(rC[0]),
         __half2float(rC[1]), __half2float(rC[2]), __half2float(rC[3]));

  *((uint32_t*)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t*)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4*)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4*)(&C_smem[lane_id][0]));
  }
}

void MmaNaive(__half* A, __half* B, __half* C, size_t M, size_t N, size_t K) {
  dim3 block(WARP_SIZE);
  dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));

  TestMmaInstruction<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
  using Element = __half;

  thrust::host_vector<Element> hA(MMA_M * MMA_K);
  for (int i = 0; i < MMA_M * MMA_K; ++i) {
    hA[i] = __float2half(i / 100.);
    // hA[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
  }
  // PrintMatrix(thrust::raw_pointer_cast(hA.data()), MMA_M, MMA_K, true);
  // std::cout << std::endl;

  thrust::host_vector<Element> hB(MMA_K * MMA_N);
  for (int i = 0; i < MMA_K * MMA_N; ++i) {
    // hB[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    hB[i] = __float2half(i / 100.);
  }
  // PrintMatrix(thrust::raw_pointer_cast(hB.data()), MMA_K, MMA_N, false);
  // std::cout << std::endl;

  thrust::device_vector<Element> A = hA;
  thrust::device_vector<Element> B = hB;

  thrust::device_vector<Element> C(MMA_M * MMA_N);
  thrust::fill(C.begin(), C.end(), static_cast<Element>(0.));

  MmaNaive(thrust::raw_pointer_cast(A.data()),
           thrust::raw_pointer_cast(B.data()),
           thrust::raw_pointer_cast(C.data()), MMA_M, MMA_N, MMA_K);

  thrust::host_vector<Element> hC1(MMA_M * MMA_N);
  hC1 = C;

  // std::cout << "mma results:" << std::endl;
  // PrintMatrix(thrust::raw_pointer_cast(hC1.data()), MMA_M, MMA_N, true);
  // std::cout << std::endl;

  thrust::host_vector<float> hC2(MMA_M * MMA_N);
  MmaRef(thrust::raw_pointer_cast(hA.data()),
         thrust::raw_pointer_cast(hB.data()),
         thrust::raw_pointer_cast(hC2.data()), MMA_M, MMA_N, MMA_K);

  return 0;
}
