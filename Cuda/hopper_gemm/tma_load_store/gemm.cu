#include "cuda_utils.cuh"
#include "hopper_gemm.cuh"

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

  // TiledMMA
  using mma_op =
      decltype(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // thread repetition
  static constexpr int kMmaEURepeatM = 1;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int MmaVN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));

  using MMA_V_RepeatT =
      decltype(make_shape(Int<MmaVM>{}, Int<MmaVN>{}, Int<MmaVK>{}));

  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));

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
void hopper_gemm(const T* A, const T* B, T* C) {
  // Block shape and cta tiler
  constexpr int kTM = 256;
  constexpr int kTN = 128;
  constexpr int kTK = 64;

  using Traits = KernelTraits<T, kM, kN, kK, kTM, kTN, kTK>;

  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;

  using TiledMMA = typename Traits::TiledMMA;
  Tensor mA = make_tensor(make_gmem_ptr(A), Shape<Int<kM>, Int<kK>>{},
                          Stride<Int<kK>, _1>{});

  // Interpret the column-major matrix B with shape [kK, kN] as a row-major
  // matrix B with shape [kN, kK]
  Tensor mB = make_tensor(make_gmem_ptr(B), Shape<Int<kN>, Int<kK>>{},
                          Stride<Int<kK>, _1>{});

  // create tma_load
  auto tma_load_A = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{});
  auto tma_load_B = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{});

  // Launch parameter setup
  constexpr int smem_size = Traits::smem_size;
  dim3 block{cute::size(TiledMMA{}), 1U, 1U};
  dim3 cluster{1, 1, 1};
  dim3 grid{ceil_div(kN, kTN), ceil_div(kM, kTM), 1U};

  void const* kernel = reinterpret_cast<void const*>(
      &ke_cute_hopper_gemm<T, Traits, decltype(tma_load_A),
                           decltype(tma_load_B)>);

  if (smem_size >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params, kernel, tma_load_A, tma_load_B);
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
  } else
    std::cout << props.minor << std::endl;

  using DType = cutlass::half_t;
  static constexpr int kM = 8192;
  static constexpr int kN = 8192;
  static constexpr int kK = 8192;

  // initialize data
  thrust::host_vector<DType> h_a(kM * kK);
  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<DType>(rand_float());
  }

  thrust::host_vector<DType> h_b(kK * kN);
  for (int i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<DType>(rand_float());
  }

  thrust::host_vector<DType> h_c(kM * kN);
  thrust::fill(h_c.begin(), h_c.end(), 0.);

  thrust::device_vector<DType> d_a = h_a;
  thrust::device_vector<DType> d_b = h_b;
  thrust::device_vector<DType> d_c = h_c;

  hopper_gemm<DType, kM, kN, kK>(thrust::raw_pointer_cast(d_a.data()),
                                 thrust::raw_pointer_cast(d_b.data()),
                                 thrust::raw_pointer_cast(d_c.data()));
  return 0;
}
