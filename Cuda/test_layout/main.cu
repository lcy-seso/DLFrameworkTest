
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
__global__ void TestTileLoader(LOAD load, Element* src) {
  extern __shared__ Element shared_storage[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  load.template load(src, shared_storage, tid);
}

int TestRowMajor() {
  using Element = cutlass::half_t;
  const int row = 8;
  const int col = 32;
  using Layout = cutlass::layout::RowMajor;

  cutlass::HostTensor<Element, Layout> matrix({row /*ld*/, col /*strided*/});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());
  // Dump the matrix.
  // std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  int smem_size = int(sizeof(Element) * row * col);
  const int kThreads = 32;
  dim3 grid(1, 1);
  dim3 block(kThreads, 1, 1);

  // row-major to column-major
  R2CTileLoader<row, col, Element, kThreads> load1(row, col);
  TestTileLoader<Element, decltype(load1)>
      <<<grid, block, smem_size, 0>>>(load1, matrix.device_ref().data());

  // row-major to row-major
  R2RTileLoader<row, col, Element, kThreads> load2(row, col);
  TestTileLoader<Element, decltype(load2)>
      <<<grid, block, smem_size, 0>>>(load2, matrix.device_ref().data());

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}

int TestColumnMajor() {
  using Element = cutlass::half_t;

  const int row = 32;
  const int col = 8;
  using Layout = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<Element, Layout> matrix({row, col});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());
  // Dump the matrix.
  // std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  int smem_size = int(sizeof(Element) * row * col);
  const int kThreads = 32;
  dim3 grid(1, 1);
  dim3 block(kThreads, 1, 1);

  // column-major to column-major
  C2CTileLoader<row, col, Element, kThreads> load1(row, col);
  TestTileLoader<Element, decltype(load1)>
      <<<grid, block, smem_size, 0>>>(load1, matrix.device_ref().data());

  // column-major to column-major
  C2RTileLoader<row, col, Element, kThreads> load2(row, col);
  TestTileLoader<Element, decltype(load2)>
      <<<grid, block, smem_size, 0>>>(load2, matrix.device_ref().data());

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}

int main() {
  TestRowMajor();
  TestColumnMajor();
}
