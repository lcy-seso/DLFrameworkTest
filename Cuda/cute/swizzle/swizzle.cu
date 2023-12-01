#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <iomanip>

using namespace cute;

void test() {
  auto row_major = Layout<Shape<_4, _8>, Stride<_8, _1>>{};
  using SwizzledColumnLayout =
      decltype(composition(Swizzle<2, 0, 3>{}, row_major));
  SwizzledColumnLayout swizzled;

  std::cout << std::endl << "row major: " << std::endl;
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      std::cout << row_major(i, j) << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl << "swizzled layout: " << std::endl;
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      std::cout << swizzled(i, j) << ", ";
    }
    std::cout << std::endl;
  }
}

template <typename Shape, typename ShmShape, typename TiledCopy,
          typename Element>
__global__ void copy(Shape problem_shape, ShmShape shm_shape,
                     TiledCopy tiled_copy, const Element* src) {
  int rows = size<0>(problem_shape);
  int cols = size<1>(problem_shape);

  auto shm_rows = size<0>(shm_shape);
  auto shm_cols = size<1>(shm_shape);

  const int shm_size = decltype(size(shm_shape))::value;

  const int x_block = blockIdx.x;
  const int y_block = blockIdx.y;

  const int offset = x_block * (shm_rows * cols) + y_block * shm_cols;

  // Interpret the buffer as a tensor using the pointer to the starting address
  // in the global memory.
  Layout row_major =
      make_layout(make_shape(shm_rows, shm_cols), make_stride(cols, 1));
  auto gmem_tile = make_tensor(make_gmem_ptr(src + offset), row_major);

  __shared__ Element smem_buf[shm_size];

  auto shmem_tile = make_tensor(
      make_smem_ptr(smem_buf),
      make_layout(make_shape(shm_rows, shm_cols), make_stride(shm_cols, 1)));

  auto loader = tiled_copy.get_thread_slice(threadIdx.x);

  auto thrd_gmem = loader.partition_S(gmem_tile);
  auto thrd_shmem = loader.partition_D(shmem_tile);
  copy(tiled_copy, thrd_gmem, thrd_shmem);
  __syncthreads();
}

int main() {
  using Element = cutlass::half_t;
  const int kRows = 4;
  const int kCols = 8;
  int numel = kRows * kCols;

  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
      Layout<Shape<_16, _8>, Stride<_1, _16>>{}, Layout<Shape<_8, _1>>{}));
}
