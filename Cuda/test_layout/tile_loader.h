#pragma once

#include <iostream>

#include "cutlass/aligned_buffer.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

// Load a tile from global memory to shared memory.
// Note: the shared memory layout is swizzled.
template <const int kRow, const int kCol,
          const int kLd /*leading dimension, 0: colum-major, 1: row-major*/,
          typename Element_, /*Element Type*/
          const int kThreads /*number of threads in a CTA*/
          >
struct TileLoader {
  static_assert(kLd == 0 or kLd == 1,
                "Wrong value for ld. The leading dimension should be 0 "
                "(column-major) or 1(row-major).");
  // TODO(ying): The current implementation requires the source element type is
  // the same as the target element type.
  using Element = Element_;

  static const int kAccessInBits = 128;
  static const int element_per_access =
      kAccessInBits / cutlass::sizeof_bits<Element>::value;

  int warp_count = kThreads / 32;

  // FIXME(ying): warp arrangement should be modified according to the original
  // problem size.
  static const int kWarpArrangeContiguous = 8;
  static const int kWarpArrangeStrided = 4;

  static const int ld_size = (kLd ? kCol : kRow);
  static const int stride = (kLd ? kRow : kCol);

  // Define a global iterator, a shared iterator and their thread map.
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ld_size, stride>, kThreads,
      cutlass::layout::PitchLinearShape<kWarpArrangeContiguous,
                                        kWarpArrangeStrided>,
      element_per_access>;

  using GTileShape = cutlass::layout::PitchLinearShape<ld_size, stride>;
  using GLayout = cutlass::layout::PitchLinear;
  using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      GTileShape, Element, GLayout, 1 /*AdvanceRank*/, ThreadMap>;

  // make crosswise be equal to the number of elements that occupy a single
  // shared memory cache line
  static const int SharedMemoryCacheLineWidth = 128;  // 128B
  static const int crosswise =
      SharedMemoryCacheLineWidth / (cutlass::sizeof_bits<Element>::value / 8);

  //   using SLayout =
  //   cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
  //       cutlass::sizeof_bits<Element>::value, crosswise>;
  using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;
  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 1, ThreadMap>;

  // Ctor
  TileLoader(int64_t row_size, int64_t col_size)
      : row_size(row_size), col_size(col_size) {}

  __device__ void load(Element* src, Element* trg, int tid) const {
    typename GmemIterator::Params params(GLayout::packed({col_size, row_size}));
    // Construct the global iterator and load the data to the fragments.
    GmemIterator gmem_iterator(params, src, {row_size, col_size}, tid);

    typename GmemIterator::Fragment frag;
    frag.clear();
    gmem_iterator.load(frag);

    SmemIterator smem_iterator(
        typename SmemIterator::TensorRef(
            {trg, SmemIterator::Layout::packed({row_size, col_size})}),
        tid);

    smem_iterator.store(frag);

    // Call dump_shmem() with different parameters.
    if (threadIdx.x == 0 && blockIdx.x == 0)
      printf("\nDump all the elements:\n");
    cutlass::debug::dump_shmem(trg, row_size * col_size);
  };

  int row_size;
  int col_size;
};
