
#include <cutlass/aligned_buffer.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>
#include <cutlass/util/debug.h>
#include <cutlass/util/device_dump.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <iomanip>
#include <iostream>

#include "cuda_utils.cuh"
#include "tile_loader.h"

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

template <typename Element, typename LOAD>
__global__ void TestTileLoader(LOAD load, Element* src) {
  extern __shared__ Element shared_storage[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  load.template load(src, shared_storage, tid);
}

int TestRowMajor() {
  // const int M = 1024;
  // const int N = 512;

  const int row = 64;
  const int col = 64;
  int numel = row * col;

  using Element = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;

  int threads = 128;
  int blocks = CEIL_DIV(numel, threads);
  __half* src;
  cudaMalloc(&src, numel * sizeof(__half));
  InitHalfs<<<blocks, threads>>>(src, numel);
  // PrintHalfs(src, numel);

  // using Element = float;
  // Element* src;
  // CudaCheck(cudaMalloc(&src, numel * sizeof(Element)));

  int smem_size = int(sizeof(Element) * row * col);
  const int kThreads = 32;
  dim3 grid(1, 1);
  dim3 block(kThreads, 1, 1);

  // FillRandomFloats(src, numel);
  // PrintFloats(src, numel);

  // return 0;

  // row-major to column-major
  TileLoader<row, col, Element, kThreads, TileLayout::RowMajor,
             TileLayout::SwizzledColumnMajor>
      load1(row, col);
  TestTileLoader<Element, decltype(load1)>
      <<<grid, block, smem_size, 0>>>(load1, reinterpret_cast<Element*>(src));

  // row-major to row-major
  TileLoader<row, col, Element, kThreads, TileLayout::RowMajor,
             TileLayout::SwizzledRowMajor>
      load2(row, col);
  TestTileLoader<Element, decltype(load2)>
      <<<grid, block, smem_size, 0>>>(load2, reinterpret_cast<Element*>(src));

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
  TileLoader<row, col, Element, kThreads, TileLayout::ColumnMajor,
             TileLayout::SwizzledColumnMajor>
      load1(row, col);
  TestTileLoader<Element, decltype(load1)>
      <<<grid, block, smem_size, 0>>>(load1, matrix.device_ref().data());

  // column-major to column-major
  TileLoader<row, col, Element, kThreads, TileLayout::ColumnMajor,
             TileLayout::SwizzledRowMajor>
      load2(row, col);
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
