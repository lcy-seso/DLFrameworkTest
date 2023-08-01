
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
          const int m, const int n>
__global__ void test_kernel_dump(typename GmemIterator::Params params,
                                 typename GmemIterator::TensorRef ref) {
  extern __shared__ Element shared_storage[];

  // Construct the global iterator and load the data to the fragments.
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  GmemIterator gmem_iterator(params, ref.data(), {m, n}, tb_thread_id);

  typename GmemIterator::Fragment frag;

  frag.clear();
  gmem_iterator.load(frag);

  // // Call dump_fragment() with different parameters.
  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("\nAll threads dump all the elements:\n");
  // cutlass::debug::dump_fragment(frag);

  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("\nFirst thread dumps all the elements:\n");
  // cutlass::debug::dump_fragment(frag, /*N = */ 1);

  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("\nFirst thread dumps first 16 elements:\n");
  // cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16);

  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
  // cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16, /*S = */ 8);

  // Construct the shared iterator and store the data to the shared memory.
  SmemIterator smem_iterator(
      typename SmemIterator::TensorRef(
          {shared_storage, SmemIterator::Layout::packed({m, n})}),
      tb_thread_id);

  smem_iterator.store(frag);

  // Call dump_shmem() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
  cutlass::debug::dump_shmem(shared_storage, m * n);

  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("\nDump all the elements with a stride of 8:\n");
  // cutlass::debug::dump_shmem(shared_storage, m * n, /*S = */ 8);
}

int main() {
  const int m = 64;
  const int n = 4;

  using Element = cutlass::half_t;
  // using Layout = cutlass::layout::RowMajor;
  using Layout = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<Element, Layout> matrix({m /*lda*/, n /*strided*/});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());
  // Dump the matrix.
  // std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  // Define a global iterator, a shared iterator and their thread map.
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<m, n>, 32 /*threads*/,
      cutlass::layout::PitchLinearShape<8, 4> /*warp arrangement*/,
      8 /*ElementPerAccess*/>;

  using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<m, n>, Element, Layout, 1 /*AdvanceRank*/,
      ThreadMap>;

  typename GmemIterator::Params params(matrix.layout());

  const int crosswise = 64;
  // using SLayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
  //         cutlass::sizeof_bits<Element>::value, crosswise>;
  using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;

  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<m, n>, Element, SLayout, 1, ThreadMap>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  int smem_size = int(sizeof(Element) * m * n);

  test_kernel_dump<Element, GmemIterator, SmemIterator, m, n>
      <<<grid, block, smem_size, 0>>>(params, matrix.device_ref());

  cudaError_t result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);

  return 0;
}
