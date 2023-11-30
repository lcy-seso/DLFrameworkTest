
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

using namespace cute;
#include <iomanip>

template <typename T>
__global__ void PrintValueHost(const T* data, int rows, int cols) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.3f, ", data[i * cols + j]);
      }
      printf("\n");
    }
  }
}

template <>
__global__ void PrintValueHost(const cutlass::half_t* data, int rows,
                               int cols) {
  __half* tmp = reinterpret_cast<__half*>(const_cast<cutlass::half_t*>(data));

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("Input:\n");
    for (int i = 0; i < rows; ++i) {
      printf("[%d]: ", i);
      for (int j = 0; j < cols - 1; ++j)
        printf("%.0f, ", __half2float(tmp[i * cols + j]));

      printf("%.0f\n", __half2float(tmp[(i + 1) * cols - 1]));
    }
    printf("\n\n");
  }
}

template <typename Element>
bool vec_eq(const Element* v1, const Element* v2, int64_t numel) {
  float e = 1e-5;

  for (int i = 0; i < numel; ++i) {
    if (abs(float(v1[i]) - float(v2[i])) > e) {
      return false;
    }
  }
  return true;
}

template <typename Shape, typename ShmShape, typename TiledCopy,
          typename Element>
__global__ void copy(Shape problem_shape, ShmShape shm_shape,
                     TiledCopy tiled_copy, const Element* src, Element* trg) {
  int rows = size<0>(problem_shape);
  int cols = size<1>(problem_shape);

  auto shm_row = size<0>(shm_shape);
  auto shm_col = size<1>(shm_shape);

  const int x_block = blockIdx.x;
  const int y_block = blockIdx.y;
  // advance the pointer to the input data to the current CTA
  const int offset = x_block * shm_row * cols + y_block * shm_col;

  auto gmem_tile =
      make_tensor(make_gmem_ptr(src + offset),
                  make_layout(make_shape(shm_row, shm_col),
                              make_stride(cols, 1)));  // laid out as row major

  __shared__ Element smem_buf[shm_row * shm_col];
  auto shmem_tile = make_tensor(
      make_smem_ptr(smem_buf),
      make_layout(make_shape(shm_row, shm_col), make_stride(shm_col, _1{})));

  // load from global memory to shared memory
  auto loader = tiled_copy.get_thread_slice(threadIdx.x);
  auto thrd_gmem = loader.partition_S(gmem_tile);
  auto thrd_shmem = loader.partition_D(shmem_tile);
  copy(tiled_copy, thrd_gmem, thrd_shmem);
  __syncthreads();

  //   store from shared memory to global memory
  auto gmem_tile_trg =
      make_tensor(make_gmem_ptr(trg + offset),
                  make_layout(make_shape(shm_row, shm_col),
                              make_stride(cols, 1)));  // laid out as row major
  auto thrd_shmem2 = loader.partition_S(shmem_tile);
  auto thrd_gmem2 = loader.partition_D(gmem_tile_trg);
  copy(tiled_copy, thrd_shmem2, thrd_gmem2);
  __syncthreads();
}

int main() {
  using Element = cutlass::half_t;

  const int kRows = 16;
  const int kCols = 64;
  int numel = kRows * kCols;

  thrust::host_vector<Element> h_A(kRows * kCols);
  srand(42);
  for (int i = 0; i < h_A.size(); ++i) {
    // h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    h_A[i] = __float2half(i);
  }

  // copy data from host to device
  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B(numel);
  thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

  const int kThreads = 32;

  Layout thread_layout = Layout<Shape<_8, _4>, Stride<_8, _1>>{};
  Layout value_layout = Layout<Shape<_1, _8>>{};

  //   const bool Has_cp_async = true;
  //   using CopyStruct = std::conditional_t<
  //       Has_cp_async, SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
  //       DefaultCopy>;
  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                    thread_layout, value_layout);

  auto shm_row = _8{};
  auto shm_col = _32{};

  dim3 dim_grid(CEIL_DIV(kRows, shm_row), CEIL_DIV(kCols, shm_col));
  dim3 dim_block(kThreads);
  copy<<<dim_grid, dim_block>>>(
      make_shape(kRows, kCols), make_shape(shm_row, shm_col), tiled_copy,
      reinterpret_cast<Element*>(thrust::raw_pointer_cast(d_A.data())),
      reinterpret_cast<Element*>(thrust::raw_pointer_cast(d_B.data())));
  cudaDeviceSynchronize();

  int blocks = CEIL_DIV(numel, kThreads);
  PrintValueHost<Element><<<blocks, kThreads>>>(
      thrust::raw_pointer_cast(d_B.data()), kRows, kCols);
  return 0;
}
