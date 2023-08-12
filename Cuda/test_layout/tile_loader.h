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
  static const int kElmentBits = cutlass::sizeof_bits<Element>::value;

  int warp_count = kThreads / 32;

  // FIXME(ying): warp arrangement should be modified according to the original
  // problem size.
  // warp arrangement: 4 x 8, 8 x 4, 8 x 8
  static const int kWarpArrangeContiguous = 4;
  static const int kWarpArrangeStrided = 8;

  static const int ld_size = (kLd ? kCol : kRow);
  static const int stride = (kLd ? kRow : kCol);

  // Define a global iterator, a shared iterator and their thread map.
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ld_size, stride> /*shape*/, kThreads,
      cutlass::layout::PitchLinearShape<
          kWarpArrangeContiguous, kWarpArrangeStrided> /*warp arrangement*/,
      kAccessInBits / kElmentBits>;

  using GTileShape = cutlass::layout::PitchLinearShape<ld_size, stride>;
  using GLayout = cutlass::layout::PitchLinear;
  using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      GTileShape, Element, GLayout, 1 /*AdvanceRank*/, ThreadMap>;

  static const int SharedMemoryCacheLineWidth = 128;  // 128B
  // one access of a thread access 128b data along the contiguous dimension
  // how many bits are accessed along the contiguous dimension for a single warp
  static const int count1 =
      kWarpArrangeContiguous * (kAccessInBits / kElmentBits);
  // the number of elements along the contiguous dimension that occupy a single
  // shared memory cache line
  // make crosswise be equal to the number of elements that occupy a single
  // shared memory cache line
  static const int count2 =
      SharedMemoryCacheLineWidth / (kAccessInBits / kElmentBits);
  static const int crosswise = count2 > ld_size ? count1 : count2;

  // using SLayout =
  // cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
  //     cutlass::sizeof_bits<Element>::value, crosswise>;
  using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, crosswise>;
  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 1, ThreadMap>;

  // Ctor
  TileLoader(int64_t row_size, int64_t col_size)
      : row_size(row_size), col_size(col_size) {}

  __device__ void load(Element* src, Element* trg, int ld, int stride,
                       int tid) const {
    typename GmemIterator::Params params(GLayout::packed({ld, stride}));
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
