#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <iomanip>

#include "utils.h"

using namespace cute;

/*
  Each CTA loads a tile of data with a shape of `ShmShape` into shared memory
  from a larger data `src` in the global memory with a shape of `Shape`, and
  then stores it in the global memory as `trg`.

  The data is stored in row-major order in `src`.
  The data is stored in row-major order in `trg`.
*/
template <typename Shape, typename ShmShape, typename TiledCopy,
          typename Element>
__global__ void copy(Shape problem_shape, ShmShape shm_shape,
                     TiledCopy tiled_copy, const Element* src, Element* trg) {
  int rows = size<0>(problem_shape);
  int cols = size<1>(problem_shape);

  auto shm_rows = size<0>(shm_shape);
  auto shm_cols = size<1>(shm_shape);

  const int shm_size = decltype(size(shm_shape))::value;

  const int x_block = blockIdx.x;
  const int y_block = blockIdx.y;

  // advance the pointer to the input data to the current CTA
  const int offset = x_block * (shm_rows * cols) + y_block * shm_cols;

  // Interpret the buffer as a tensor using the pointer to the starting address
  // in the global memory.
  Layout row_major =
      make_layout(make_shape(shm_rows, shm_cols), make_stride(cols, 1));
  auto gmem_tile = make_tensor(make_gmem_ptr(src + offset), row_major);

  __shared__ Element smem_buf[shm_size];
  // shared memory is interpreted as a row major matrix
  auto shmem_tile = make_tensor(
      make_smem_ptr(smem_buf),
      make_layout(make_shape(shm_rows, shm_cols), make_stride(shm_cols, 1)));

  auto loader = tiled_copy.get_thread_slice(threadIdx.x);

  auto thrd_gmem = loader.partition_S(gmem_tile);
  auto thrd_shmem = loader.partition_D(shmem_tile);
  copy(tiled_copy, thrd_gmem, thrd_shmem);
  __syncthreads();

  // store shared memory tile into global memory
  auto gmem_tile_trg = make_tensor(make_gmem_ptr(trg + offset), row_major);

  auto thrd_shmem2 = loader.partition_S(shmem_tile);
  auto thrd_gmem2 = loader.partition_D(gmem_tile_trg);
  copy(tiled_copy, thrd_shmem2, thrd_gmem2);
}

int main() {
  using Element = cutlass::half_t;
  const int kRows = 32 * 3;
  const int kCols = 128 * 7;
  int numel = kRows * kCols;

  thrust::host_vector<Element> h_A(kRows * kCols);
  srand(42);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    // h_A[i] = __float2half(i);
  }

  // copy data from host to device
  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B(numel);
  thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

  const int kThreads = 64;
  auto shm_row = _32{};
  auto shm_col = _128{};

  // threads are laid out as a row major matrix
  Layout thread_layout = Layout<Shape<_16, _4>, Stride<_4, _1>>{};
  // values are laid out as a row vector, 8 values per access.
  Layout value_layout = Layout<Shape<_1, _8>, Stride<_0, _1>>{};

  auto shm_shape = make_shape(shm_row, shm_col);
  auto problem_shape = make_shape(kRows, kCols);

  // const bool Has_cp_async = true;
  // using CopyStruct = std::conditional_t<
  //     Has_cp_async, SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
  //     DefaultCopy>;

  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                    thread_layout, value_layout);

  dim3 dim_grid(CEIL_DIV(kRows, shm_row), CEIL_DIV(kCols, shm_col));
  dim3 dim_block(kThreads);
  copy<<<dim_grid, dim_block>>>(problem_shape, shm_shape, tiled_copy,
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

  // int blocks = CEIL_DIV(numel, kThreads);
  // PrintValueHost<Element><<<blocks, kThreads>>>(
  //     thrust::raw_pointer_cast(d_B.data()), kRows, kCols);

  return 0;
}
