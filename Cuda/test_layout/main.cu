
#include <iomanip>
#include <iostream>

#include "cutlass/aligned_buffer.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"
#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "tile_loader.h"

template <typename Element, typename LOAD>
__global__ void TestTileLoader(LOAD load, Element* src, int ld_size,
                               int stride) {
  extern __shared__ Element shared_storage[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  load.template load(src, shared_storage, ld_size, stride, tid);
}

int main() {
  // Row-major has a layout of [strided x lda]-shaped matrix.
  const int row = 32;
  const int col = 8;

  // const int row = 8;
  // const int col = 32;

  using Element = cutlass::half_t;

  // ld for leading dimension:
  // ld = 1: row-major, ld = 0: column-major
  // const int ld = 1;
  // using Layout = cutlass::layout::RowMajor;
  using Layout = cutlass::layout::ColumnMajor;
  const int ld = 0;
  cutlass::HostTensor<Element, Layout> matrix({row /*ld*/, col /*strided*/});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());
  // Dump the matrix.
  //   std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  int smem_size = int(sizeof(Element) * row * col);
  const int kAccessInBits = 128;
  const int kThreads = 32;
  const int kElementPerAccess =
      kAccessInBits / cutlass::sizeof_bits<Element>::value;

  const int ld_size = (ld ? col : row);
  const int stride = (ld ? row : col);
  std::cout << "leading dimension size: " << ld_size << "; stride: " << stride
            << std::endl;

  // column-major
  TileLoader<row, col, ld, Element, kThreads> load(row, col);
  TestTileLoader<Element, decltype(load)><<<grid, block, smem_size, 0>>>(
      load, matrix.device_ref().data(), ld_size, stride);

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}
