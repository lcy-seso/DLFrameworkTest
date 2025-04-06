#include "cuda_utils.cuh"
#include "tma_copy.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename DType, const int kThreads,  //
          const int kM, const int kN, const int kTM, const int kTN>
int test_tma_copy(const DType* src, DType* dst) {
  static constexpr int kSharedMemSize = kTM * kTN;

  // source tensor on global memory
  using GMemLayout = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;
  GMemLayout g_layout;
  auto g_src = make_tensor(make_gmem_ptr(src), g_layout);

  // intermediate tensor on shared memory
  using SMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  SMemLayout s_layout;

  // destination tensor on global memory
  auto g_dst = make_tensor(make_gmem_ptr(dst), g_layout);

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, g_src, s_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, g_dst, s_layout);

  int block_x = CeilDiv<kM, kTM>;
  int block_y = CeilDiv<kN, kTN>;

  dim3 blocks(block_x, block_y, 1);
  dim3 threads(kThreads, 1, 1);

  auto thread_layout = make_layout(Shape<_32, Int<CeilDiv<kThreads, 32>>>{});

  std::cout << "kSharedMemSize: " << kSharedMemSize << std::endl;

  auto kernel = &ke_tma_copy<kSharedMemSize, DType,  //
                             decltype(thread_layout), GMemLayout, SMemLayout,
                             decltype(tma_load), decltype(tma_store)>;

  kernel<<<blocks, threads>>>(src, dst, thread_layout, g_layout, s_layout,
                              tma_load, tma_store);
  cudaDeviceSynchronize();
}

// int test_tma_copy_multicast() {
//   using DType = float;

//   static constexpr int kTM = 64;
//   static constexpr int kTN = 128;
// }

int main() {
  using DType = float;

  static constexpr int kTM = 64;
  static constexpr int kTN = 128;

  static constexpr int kM = kTM * 1;
  static constexpr int kN = kTN * 1;

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

  test_tma_copy<DType, 256 /*kThreads*/, kM, kN, kTM, kTN>(d_src.data().get(),
                                                           d_dst.data().get());

  h_dst = d_dst;
  cudaDeviceSynchronize();
  check_results(h_src.data(), h_dst.data(), kNumel);

  return 0;
}
