#include "cuda_utils.cuh"
#include "tma_copy.cuh"

#include <cutlass/cluster_launch.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

int main() {
  using DType = float;

  static constexpr int kTM = 64;
  static constexpr int kTN = 128;
  static constexpr int kSharedMemSize = kTM * kTN;

  static constexpr int kM = kTM * 128;
  static constexpr int kN = kTN * 128;

  static constexpr int kNumel = kM * kN;

  static constexpr int kThreads = 256;

  // threads block cluster
  static constexpr int kCopyN = 2;
  using ClusterShape = Shape<_1, _1, Int<kCopyN>>;
  ClusterShape cluster_shape;

  thrust::host_vector<DType> h_src(kNumel);  // kM * kN
  for (int i = 0; i < kNumel; ++i) {
    // h_src[i] = static_cast<DType>(i % 2048);
    h_src[i] = rand_float();
  }

  thrust::host_vector<DType> h_dst(kNumel * kCopyN);  // kM * kN * kCopyN
  for (int i = 0; i < kNumel * kCopyN; ++i) {
    h_dst[i] = static_cast<DType>(0.);
  }

  thrust::device_vector<DType> d_src = h_src;
  thrust::device_vector<DType> d_dst = h_dst;
  cudaDeviceSynchronize();

  const DType* d_src_ptr = thrust::raw_pointer_cast(d_src.data());
  DType* d_dst_ptr = thrust::raw_pointer_cast(d_dst.data());

  // source tensor on global memory
  using GMemLayout = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;
  GMemLayout g_layout;
  auto g_tensor_src = make_tensor(make_gmem_ptr(d_src_ptr), g_layout);

  // intermediate tensor on shared memory
  // TODO: no swizzling is applied at the moment
  using SMemLayout = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;
  SMemLayout s_layout;

  using TileShape = Shape<Int<kTM>, Int<kTN>>;
  auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, g_tensor_src,
                                s_layout, TileShape{}, size(cluster_shape));

  // destination tensor on global memory
  using TensorShapeOut = Shape<Int<kM>, Int<kN>, Int<kCopyN>>;
  auto g_dst_layout = make_ordered_layout(TensorShapeOut{}, Step<_1, _0, _2>{});

  auto g_tensor_dst = make_tensor(make_gmem_ptr(d_dst_ptr), g_dst_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, g_tensor_dst, s_layout,
                                 TileShape{}, Int<1>{});

  static constexpr int kThreadCols = CeilDiv<kThreads, 32>;
  using ThreadLayout =
      Layout<Shape<_32, Int<kThreadCols>>, Stride<Int<kThreadCols>, _1>>;

  int block_x = CeilDiv<kM, kTM>;
  int block_y = CeilDiv<kN, kTN>;
  dim3 blocks(block_x, block_y, kCopyN);
  dim3 threads(kThreads, 1, 1);
  dim3 cluster_dims(size<0>(cluster_shape), size<1>(cluster_shape),
                    size<2>(cluster_shape));

  using SharedStorage = SharedStorageImpl<DType, kSharedMemSize>;
  int smem_size = int(sizeof(SharedStorage));
  cutlass::ClusterLaunchParams launch_params{blocks, threads, cluster_dims,
                                             smem_size};

  void const* kernel = reinterpret_cast<void const*>(
      &ke_tma_copy_multicast<DType, SharedStorage,        //
                             ClusterShape, ThreadLayout,  //
                             GMemLayout, decltype(g_dst_layout), SMemLayout,
                             decltype(tma_load), decltype(tma_store)>);

  cutlass::Status status =
      cutlass::launch_kernel_on_cluster(launch_params, kernel,  //
                                        d_src_ptr, d_dst_ptr,   //
                                        g_layout, g_dst_layout,
                                        s_layout,  //
                                        tma_load, tma_store);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel launch failed with status: " << std::endl;
  }

  h_dst = d_dst;
  check_results(thrust::raw_pointer_cast(h_src.data()),
                thrust::raw_pointer_cast(h_dst.data()), kNumel);
  return 0;
}
