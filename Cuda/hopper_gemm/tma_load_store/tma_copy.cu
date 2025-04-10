#include "cuda_utils.cuh"
#include "tma_copy.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

int main() {
  using DType = float;

  static constexpr int kTM = 64;
  static constexpr int kTN = 128;

  static constexpr int kM = kTM * 128;
  static constexpr int kN = kTN * 256;

  static constexpr int kNumel = kM * kN;

  thrust::host_vector<DType> h_src(kNumel);
  thrust::host_vector<DType> h_dst(kNumel);
  for (int i = 0; i < kNumel; ++i) {
    // h_src[i] = static_cast<DType>(i % 2048);
    h_src[i] = rand_float();
    h_dst[i] = static_cast<DType>(0);
  }

  thrust::device_vector<DType> d_src = h_src;
  thrust::device_vector<DType> d_dst = h_dst;
  cudaDeviceSynchronize();

  const DType* d_src_ptr = d_src.data().get();
  DType* d_dst_ptr = d_dst.data().get();

  static constexpr int kSharedMemSize = kTM * kTN;
  static constexpr int kThreads = 256;

  // source tensor on global memory
  using GMemLayout = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;
  GMemLayout g_layout;
  auto g_src = make_tensor(make_gmem_ptr(d_src_ptr), g_layout);

  // intermediate tensor on shared memory
  using SMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  SMemLayout s_layout;

  // destination tensor on global memory
  auto g_dst = make_tensor(make_gmem_ptr(d_dst_ptr), g_layout);

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, g_src, s_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, g_dst, s_layout);

  int block_x = CeilDiv<kM, kTM>;
  int block_y = CeilDiv<kN, kTN>;

  dim3 blocks(block_x, block_y, 1);
  dim3 threads(kThreads, 1, 1);

  std::cout << "kSharedMemSize: " << kSharedMemSize << std::endl;

  static constexpr int kThreadCols = CeilDiv<kThreads, 32>;
  using ThreadLayout =
      Layout<Shape<_32, Int<kThreadCols>>, Stride<Int<kThreadCols>, _1>>;

  auto kernel = &ke_tma_copy<DType, kSharedMemSize,  //
                             ThreadLayout, GMemLayout, SMemLayout,
                             decltype(tma_load), decltype(tma_store)>;

  kernel<<<blocks, threads>>>(d_src_ptr, d_dst_ptr, g_layout, s_layout,
                              tma_load, tma_store);

  h_dst = d_dst;

  check_results(h_src.data(), h_dst.data(), kNumel);
  return 0;
}
