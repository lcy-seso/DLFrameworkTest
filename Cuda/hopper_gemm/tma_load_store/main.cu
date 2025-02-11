#include <cutlass/arch/barrier.h>

#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "cuda_gemm.hpp"

using namespace cute;

template <typename T>
struct Params {
  int M, N, K;
  T* C;
  const T alpha;
  const T beta;
};

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
  using Element = T;
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

int main() { return 0; }
