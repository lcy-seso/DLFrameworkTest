#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <iomanip>

#include "utils.h"

using namespace cute;

template <typename Element_, const int kRows_, const int kCols_,
          const int kShmRows_, const int kShmCols_, const int kThreads>
struct KeTraits {
  using Element = Element_;

  static constexpr int kRows = kRows_;
  static constexpr int kCols = kCols_;

  static constexpr int kShmRows = kShmRows_;
  static constexpr int kShmCols = kShmCols_;

  static const int kAccessInBits = 128;
  static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
  static const int kNumPerAccess = kAccessInBits / kElmentBits;

  static constexpr int kThreadsPerRow = kShmCols / kNumPerAccess;
  using GmemLayoutAtom =
      Layout<Shape<Int<kThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
             Stride<Int<kThreadsPerRow>, _1>>;
  using TiledCopy =
      decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                               GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

  // FIXME(ying): still has bug
  // using TiledCopy = decltype(make_tiled_copy(
  //     Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
  //     GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

  using GmemLayout =
      Layout<Shape<Int<kShmRows>, Int<kShmCols>>, Stride<Int<kCols>, _1>>;
  using SmemLayout =
      Layout<Shape<Int<kShmRows>, Int<kShmCols>>, Stride<Int<kShmCols>, _1>>;
};

namespace {
__device__ void DebugPrint(const cutlass::half_t* input, int row, int col) {
  auto* data = reinterpret_cast<const __half*>(input);

  for (int i = 0; i < row; ++i) {
    printf("[%d]:\t", i);
    for (int j = 0; j < col - 1; ++j) {
      printf("%.0f,", __half2float(data[i * col + j]));
    }
    printf("%.0f\n", __half2float(data[(i + 1) * col - 1]));
  }
  printf("\n");
}

}  // namespace

/*
  Each CTA loads a tile of data with a shape of `ShmShape` into shared memory
  from a larger data `src` in the global memory with a shape of `Shape`, and
  then stores it in the global memory as `trg`.

  The data is stored in row-major order in `src`.
  The data is stored in row-major order in `trg`.
*/

template <typename Element, typename KeTraits>
__global__ void Copy(const Element* src, Element* trg) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* smem_buf = reinterpret_cast<Element*>(shared_buf);

  int rows = KeTraits::kRows;
  int cols = KeTraits::kCols;

  int shm_rows = KeTraits::kShmRows;
  int shm_cols = KeTraits::kShmCols;

  const int x_block = blockIdx.x;
  const int y_block = blockIdx.y;

  // advance the pointer to the input data to the current CTA
  const int offset = x_block * (shm_rows * cols) + y_block * shm_cols;

  // Interpret the buffer as a tensor using the pointer to the starting address
  // in the global memory.
  typename KeTraits::GmemLayout g_layout;
  auto g_tile = make_tensor(make_gmem_ptr(src + offset), g_layout);

  // shared memory is interpreted as a row major matrix
  typename KeTraits::SmemLayout s_layout;
  auto s_tile = make_tensor(make_smem_ptr(smem_buf), s_layout);

  typename KeTraits::TiledCopy tiled_copy;
  auto loader = tiled_copy.get_thread_slice(threadIdx.x);

  auto thrd_gmem = loader.partition_S(g_tile);
  auto thrd_shmem = loader.partition_D(s_tile);
  copy(tiled_copy, thrd_gmem, thrd_shmem);
  cp_async_fence();
  cp_async_wait<0>();

  // store shared memory tile into global memory
  auto g_tile2 = make_tensor(make_gmem_ptr(trg + offset), g_layout);
  auto thrd_shmem2 = loader.partition_S(s_tile);
  auto thrd_gmem2 = loader.partition_D(g_tile2);
  copy(tiled_copy, thrd_shmem2, thrd_gmem2);
  cute::cp_async_fence();
}

int main() {
  using Element = cutlass::half_t;
  static constexpr int kRows = 16 * 2;
  static constexpr int kCols = 16 * 3;

  static constexpr int kShmRows = 16;
  static constexpr int kShmCols = 16;

  static constexpr int kThreads = 32;

  int numel = kRows * kCols;
  thrust::host_vector<Element> h_A(numel);
  srand(42);
  for (int i = 0; i < h_A.size(); ++i) {
    // h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    h_A[i] = __float2half(i);
  }

  // copy data from host to device
  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B(numel);
  thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

  dim3 dim_grid(CEIL_DIV(kRows, kShmRows), CEIL_DIV(kCols, kShmCols));
  dim3 dim_block(kThreads);

  using KeTraits_ =
      KeTraits<Element, kRows, kCols, kShmRows, kShmCols, kThreads>;

  Copy<Element, KeTraits_><<<dim_grid, dim_block, kShmRows * kShmCols>>>(
      thrust::raw_pointer_cast(d_A.data()),
      thrust::raw_pointer_cast(d_B.data()));
  cudaDeviceSynchronize();

  // unittest
  thrust::host_vector<Element> h_B(numel);
  h_B = d_B;
  assert(CheckResult(
      reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_A.data())),
      reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_B.data())),
      h_A.size()));

  // std::cout << std::endl;
  // int blocks = CEIL_DIV(numel, kThreads);
  // PrintValueHost<Element><<<blocks, kThreads>>>(
  //     thrust::raw_pointer_cast(d_B.data()), kRows, kCols);

  return 0;
}
