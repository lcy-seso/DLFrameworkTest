#include "cuda_utils.cuh"
#include "hopper_gemm.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cute;

namespace {
template <typename T>
struct Params {
  int M, N, K;
  T* C;
  const T alpha;
  const T beta;
};
}  // namespace

// shared storage
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
template <typename T, int kBlockM_, int kBlockN_, int kBlockK_>
struct KernelTraits {
  using DType = T;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kBlockK_;

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
      SmemLayoutAtom{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{})));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kBlockN>{}, Int<kBlockK>{})));

  // SharedStorage
  using SharedStorage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;

  // smem_size
  static constexpr int smem_size = sizeof(SharedStorage);
};

template <typename T>
void hopper_gemm(int m, int n, int k, const T* alpha, const T* A, int lda,
                 const T* B, int ldb, const T* beta, T* C, int ldc,
                 cudaStream_t stream) {
  using ParamsT = Params<T>;
  ParamsT params = {m, n, k, C, *alpha, *beta};

  // Block shape and cta tiler
  constexpr int kBlockM_ = 256;
  constexpr int kBlockN_ = 128;
  constexpr int kBlockK_ = 64;

  using Kernel_traits = KernelTraits<T, kBlockM_, kBlockN_, kBlockK_>;

  using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
  using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
  using TiledMMA = typename Kernel_traits::TiledMMA;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(params.M, params.K),
                          make_stride(lda, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(params.N, params.K),
                          make_stride(ldb, _1{}));

  // Finally we create tma_load
  auto tma_load_A = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{});

  auto tma_load_B = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{});

  // Launch parameter setup
  constexpr int smem_size = Kernel_traits::smem_size;
  dim3 block{cute::size(TiledMMA{}), 1U, 1U};
  dim3 cluster{1, 1, 1};
  dim3 grid{utils::ceil_div(params.N, kBlockN_),
            utils::ceil_div(params.M, kBlockM_), 1U};

  cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size,
                                             stream};

  void const* kernel = reinterpret_cast<void const*>(
      &ke_cute_hopper_gemm<ParamsT, Kernel_traits, decltype(tma_load_A),
                           decltype(tma_load_B)>);

  if (smem_size >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params, kernel, params, tma_load_A, tma_load_B);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel launch failed with status: " << std::endl;
  }
}

int main() {
  using DType = cutlass::half_t;
  static constexpr int kM = 8192;
  static constexpr int kN = 8192;
  static constexpr int kK = 8192;

  // initialize data
  thrust::host_vector<DType> h_a(kM * kK);
  for (int i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<DType>(utils::rand_float());
  }

  thrust::host_vector<DType> h_b(kK * kN);
  for (int i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<DType>(utils::rand_float());
  }

  thrust::host_vector<DType> h_c(kM * kN);
  thrust::fill(h_c.begin(), h_c.end(), 0.);

  thrust::device_vector<DType> d_a = h_a;
  thrust::device_vector<DType> d_b = h_b;
  thrust::device_vector<DType> d_c = h_c;

  DType alpha = static_cast<DType>(1.0);
  DType beta = static_cast<DType>(0.0);

  int lda = kK;  // row major
  int ldb = kK;  // column major
  int ldc = kN;  // row major

  cudaStream_t stream = 0;
  hopper_gemm<DType>(kM, kN, kK, &alpha, thrust::raw_pointer_cast(d_a.data()),
                     lda, thrust::raw_pointer_cast(d_b.data()), ldb, &beta,
                     thrust::raw_pointer_cast(d_c.data()), ldc, stream);
  return 0;
}
