
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

template <typename Element, typename GmemIterator, typename SmemIterator,
          const int row, const int col>
__global__ void load(typename GmemIterator::Params params, Element* data) {
  extern __shared__ Element shared_storage[];

  // Construct the global iterator and load the data to the fragments.
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  GmemIterator gmem_iterator(params, data, {row, col}, tb_thread_id);

  typename GmemIterator::Fragment frag;
  frag.clear();
  gmem_iterator.load(frag);

  SmemIterator smem_iterator(
      typename SmemIterator::TensorRef(
          {shared_storage, SmemIterator::Layout::packed({row, col})}),
      tb_thread_id);

  smem_iterator.store(frag);

  // Call dump_shmem() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
  cutlass::debug::dump_shmem(shared_storage, row * col);
}

int main() {
  // Row-major has a layout of [strided x lda]-shaped matrix.
  const int row = 4;
  const int col = 64;

  int numel = row * col;

  using Element = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;
  // using Layout = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<Element, Layout> matrix({row /*ld*/, col /*strided*/});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());
  // Dump the matrix.
  //   std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  int smem_size = int(sizeof(Element) * numel);
  const int kAccessInBits = 128;
  const int num_threads = 32;
  const int element_per_access =
      kAccessInBits / cutlass::sizeof_bits<Element>::value;

  const int lda = col;
  const int strided = row;
  // Define a global iterator, a shared iterator and their thread map.
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<lda /*contiguous dim*/, strided>,
      num_threads /*threads*/,
      cutlass::layout::PitchLinearShape<8 /*contiguous dim*/,
                                        4> /*warp arrangement*/,
      element_per_access /*ElementPerAccess*/>;

  // Row-Major
  using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<row /*row*/, col /*col*/>, Element, Layout,
      1 /*AdvanceRank*/, ThreadMap>;

  typename GmemIterator::Params params(Layout::packed({strided, lda}));
  // make crosswise be equal to the number of elements that occupy a single
  // shared memory cache line
  const int SharedMemoryCacheLineWidth = 128;  // 128B
  const int crosswise =
      SharedMemoryCacheLineWidth / (cutlass::sizeof_bits<Element>::value / 8);
  std::cout << crosswise << std::endl;
  using SLayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;
  // using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
  //     cutlass::sizeof_bits<Element>::value, crosswise>;
  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<strided /*row*/, lda /*col*/>, Element, SLayout, 1,
      ThreadMap>;

  load<Element /*element type*/, GmemIterator /*source iterator*/,
       SmemIterator /*target iterator*/, row, col>
      <<<grid, block, smem_size, 0>>>(params, matrix.device_ref().data());

  cudaError_t result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);

  return 0;
}
