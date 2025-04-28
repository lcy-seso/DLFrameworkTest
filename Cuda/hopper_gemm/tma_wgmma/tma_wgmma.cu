#include "cuda_utils.cuh"
#include "hopper_gemm.cuh"
#include "utils.hpp"

#include <cute/arch/copy_sm90.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

template <typename DType, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  // data storage
  array_aligned<DType, cosize_v<SmemLayoutA>, 128> smem_A;
  array_aligned<DType, cosize_v<SmemLayoutB>, 128> smem_B;

  // barrier
  uint64_t smem_A_barrier;
  uint64_t smem_B_barrier;
};

// kernel traits
template <typename DType, const int kM_, const int kN_, const int kK_,
          const int kTM_, const int kTN_, const int kTK_>
struct KeTraits {
  static constexpr int kM = kM_;
  static constexpr int kN = kN_;
  static constexpr int kK = kK_;

  static constexpr int kTM = kTM_;
  static constexpr int kTN = kTN_;
  static constexpr int kTK = kTK_;

  // A is laid out as a row-major tensor
  using LayoutGmemA = Layout<Shape<Int<kM>, Int<kK>>, Stride<Int<kK>, _1>>;
  // interpret B as a row-major tensor
  using LayoutGmemB = Layout<Shape<Int<kN>, Int<kK>>, Stride<Int<kK>, _1>>;
  // C is laid out as a row-major tensor
  using LayoutGmemC = Layout<Shape<Int<kM>, Int<kN>>, Stride<Int<kN>, _1>>;

  // 1. 16-bit data type for inputs and output, accumulator in 16 bit.
  // 2. operand A and B are sourced from shared memory.
  // 3. operand A and B are both memory-contiguous in the K mode, that means
  //    A is row-major and B is column-major.
  // about the permutation layout (the third parameter of `TiledMMA`) see this
  // issue: https://github.com/NVIDIA/cutlass/discussions/1345
  using TiledMma = TiledMMA<
      MMA_Atom<SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>>,
      Layout<Shape<_1, _1, _1>>,  // use a single warp group
      Tile<_64, _64, _16>>;

  // swizzle function: <3, 4, 3>, shape: ((8, 64), (64, 1))
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<DType>;
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTM>{}, Int<kTK>{})));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTN>{}, Int<kTK>{})));

  // SharedStorage
  using SharedStorage = SharedStorage<DType, SmemLayoutA, SmemLayoutB>;

  // smem_size
  static constexpr int kShmSize = sizeof(SharedStorage);
};

template <typename DType, const int kM, const int kN, const int kK>
void hopper_gemm(const DType* gA_ptr, const DType* gB_ptr, DType* gC_ptr) {
  // Block shape and cta tiler
  constexpr int kTM = 64;
  constexpr int kTN = 64;
  constexpr int kTK = 64;

  using Traits = KeTraits<DType, kM, kN, kK, kTM, kTN, kTK>;

  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;

  using TiledMma = typename Traits::TiledMma;

#if 0
  std::cout << "TiledMMA" << std::endl;
  print(TiledMma{});
  std::cout << std::endl;

  using SmemLayoutAtom = typename Traits::SmemLayoutAtom;
  std::cout << std::endl << "SmemLayoutAtom" << std::endl;
  print(SmemLayoutAtom{});
  std::cout << std::endl;
#endif

  Tensor gA =
      make_tensor(make_gmem_ptr(gA_ptr), typename Traits::LayoutGmemA{});
  Tensor gB =
      make_tensor(make_gmem_ptr(gB_ptr), typename Traits::LayoutGmemB{});

  // create tma_load
  auto tma_load_A = make_tma_copy(SM90_TMA_LOAD{}, gA, SmemLayoutA{});
  auto tma_load_B = make_tma_copy(SM90_TMA_LOAD{}, gB, SmemLayoutB{});

  void const* kernel = reinterpret_cast<void const*>(
      &ke_cute_tma_wgmma<DType, Traits, decltype(tma_load_A),
                         decltype(tma_load_B)>);

  constexpr int kShmSize = Traits::kShmSize;
  if (kShmSize >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize));
  }

  dim3 grid{ceil_div(kN, kTN), ceil_div(kM, kTM), 1U};
  dim3 block{cute::size(TiledMma{}), 1U, 1U};
  dim3 cluster{1, 1, 1};

  cutlass::ClusterLaunchParams launch_params{grid, block, cluster, kShmSize};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params, kernel, tma_load_A, tma_load_B, gC_ptr);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel launch failed with status: " << std::endl;
  }
}

int main() {
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);

  std::cout << "CUDA Device: " << props.name << std::endl;
  std::cout << "CUDA Compute Capability: " << props.major << ".";

  if (props.major != 9 || props.minor != 0) {
    std::cerr << std::endl
              << "This example requires a GPU of NVIDIA's Hopper Architecture "
                 "(compute capability 90).\n";
    return 0;
  } else {
    std::cout << props.minor << std::endl;
  }

  using DType = cutlass::half_t;
  static constexpr int kM = 1024;
  static constexpr int kN = 1024;
  static constexpr int kK = 1024;

  // initialize data
  thrust::host_vector<DType> h_a(kM * kK);  // 64 * 16
  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<DType>(rand_normal(0.05f, 1e-2f));
    // h_a[i] = static_cast<DType>(i % 2048);
  }

  thrust::host_vector<DType> h_b(kK * kN);  // 16 * 64
  for (int i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<DType>(rand_normal(0.03f, 5e-2f));
    // h_b[i] = static_cast<DType>(i % 2048);
  }

  thrust::host_vector<DType> h_c(kM * kN);  // 64 * 64
  thrust::fill(h_c.begin(), h_c.end(), 0.);

  thrust::host_vector<DType> h_c_ref(kM * kN);  // 64 * 64
  thrust::fill(h_c_ref.begin(), h_c_ref.end(), 0.);

  thrust::device_vector<DType> d_a = h_a;
  thrust::device_vector<DType> d_b = h_b;
  thrust::device_vector<DType> d_c = h_c;
  thrust::device_vector<DType> d_c_ref = h_c_ref;
  hopper_gemm<DType, kM, kN, kK>(thrust::raw_pointer_cast(d_a.data()),
                                 thrust::raw_pointer_cast(d_b.data()),
                                 thrust::raw_pointer_cast(d_c.data()));
  h_c = d_c;
  cudaDeviceSynchronize();

  // ground truth
  const __half* A =
      reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_a.data()));
  const __half* B =
      reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_b.data()));
  __half* C =
      reinterpret_cast<__half*>(thrust::raw_pointer_cast(d_c_ref.data()));
  cublas_hgemm(kM, kN, kK, A, B, C);
  h_c_ref = d_c_ref;

  {  // check result
    const __half* data =
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(h_c.data()));

    const __half* data_ref = reinterpret_cast<const __half*>(
        thrust::raw_pointer_cast(h_c_ref.data()));

#if 1
    // debug print
    print_matrix(data, kM, kN, 64);
    print_matrix(data_ref, kM, kN, 64);

    check_result(data, data_ref, kM * kN);
#endif
  }

  return 0;
}
