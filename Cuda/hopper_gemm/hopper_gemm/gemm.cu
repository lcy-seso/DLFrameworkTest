#include "cuda_utils.cuh"
#include "hopper_gemm.cuh"
#include "utils.hpp"

#include <cute/arch/copy_sm90.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

template <typename T, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  // data storage
  array_aligned<T, cosize_v<SmemLayoutA>, 128> smem_A;
  array_aligned<T, cosize_v<SmemLayoutB>, 128> smem_B;

  // barrier
  uint64_t smem_A_barrier;
  uint64_t smem_B_barrier;
};

// kernel traits
template <typename T, const int kM_, const int kN_, const int kK_,
          const int kTM_, const int kTN_, const int kTK_>
struct KernelTraits {
  static constexpr int kM = kM_;
  static constexpr int kN = kN_;
  static constexpr int kK = kK_;

  static constexpr int kTM = kTM_;
  static constexpr int kTN = kTN_;
  static constexpr int kTK = kTK_;

  // 1. 16-bit data type for inputs and output, accumulator in 16 bit.
  // 2. operand A and B are sourced from shared memory.
  // 3. operand A and B are both memory-contiguous in the K mode, that means
  //    A is row-major and B is column-major.
  // about the permutation layout (the third parameter of `TiledMMA`) see this
  // issue: https://github.com/NVIDIA/cutlass/discussions/1345
  using TiledMMA = decltype(make_tiled_mma(
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},  // wgmma
      Layout<Shape<_1, _1, _1>>{},  // atom layout
      Tile<_64, _64, _16>{}));      // tiler for the MNK mode

  // swizzle function: <3, 4, 3>, shape: ((8, 64), (64, 1))
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<T>;
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTM>{}, Int<kTK>{})));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTN>{}, Int<kTK>{})));

  // SharedStorage
  using SharedStorage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;

  // smem_size
  static constexpr int smem_size = sizeof(SharedStorage);
};

template <typename T, const int kM, const int kN, const int kK>
void hopper_gemm(const T* gA, const T* gB, T* gC) {
  // Block shape and cta tiler
  constexpr int kTM = 64;
  constexpr int kTN = 64;
  constexpr int kTK = 64;

  using Traits = KernelTraits<T, kM, kN, kK, kTM, kTN, kTK>;

  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;

  using TiledMMA = typename Traits::TiledMMA;
  // print_latex(TiledMMA{});

#if 0
  std::cout << "TiledMMA" << std::endl;
  print(TiledMMA{});
  std::cout << std::endl;

  using SmemLayoutAtom = typename Traits::SmemLayoutAtom;
  std::cout << std::endl << "SmemLayoutAtom" << std::endl;
  print(SmemLayoutAtom{});
  std::cout << std::endl;
#endif

  using LayoutA = Layout<Shape<Int<kM>, Int<kK>>, Stride<Int<kK>, _1>>;
  Tensor mA = make_tensor(make_gmem_ptr(gA), LayoutA{});

  // Interpret the column-major matrix B with shape [kK, kN] as a row-major
  // matrix B with shape [kN, kK]
  using LayoutB = Layout<Shape<Int<kN>, Int<kK>>, Stride<Int<kK>, _1>>;
  Tensor mB = make_tensor(make_gmem_ptr(gB), LayoutB{});

  // create tma_load
  auto tma_load_A = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{});
  auto tma_load_B = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{});

  void const* kernel = reinterpret_cast<void const*>(
      &ke_cute_hopper_gemm<T, Traits, decltype(tma_load_A),
                           decltype(tma_load_B)>);

  constexpr int smem_size = Traits::smem_size;
  if (smem_size >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  // Launch parameter setup
  dim3 block{cute::size(TiledMMA{}), 1U, 1U};
  dim3 cluster{1, 1, 1};
  dim3 grid{ceil_div(kN, kTN), ceil_div(kM, kTM), 1U};

  cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params, kernel, tma_load_A, tma_load_B, gC);
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
  static constexpr int kM = 64;
  static constexpr int kN = 64;
  static constexpr int kK = 64;

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
    print_matrix(data, kM, kN, 32);
    print_matrix(data_ref, kM, kN, 32);

    check_result(data, data_ref, kM * kN);
#endif
  }

  return 0;
}
