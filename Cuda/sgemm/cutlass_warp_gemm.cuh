#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/arch/arch.h>
#include <cutlass/array.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/numeric_types.h>
#include <cutlass/platform/platform.h>
#include <cutlass/subbyte_reference.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>

/// Test kernel
template <typename Mma, typename ThreadblockShape, typename WholeShape,
          const uint kThreads>
__global__ void GemmKernel(typename Mma::ElementC* output_C,
                           typename Mma::ElementA* const input_A,
                           typename Mma::ElementB* const input_B) {
  // Use AlignedBuffer to store trivially copyable objects in unions and
  // __shared__ buffers.
  __shared__ cutlass::AlignedBuffer<typename Mma::ElementA,
                                    ThreadblockShape::kM * ThreadblockShape::kK>
      smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<typename Mma::ElementB,
                                    ThreadblockShape::kN * ThreadblockShape::kK>
      smem_buffer_B;

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint warpIdx = threadIdx.x / 32;
  const uint warpRow = warpIdx / (ThreadblockShape::kN / Mma::Shape::kN);
  const uint warpCol = warpIdx % (ThreadblockShape::kN / Mma::Shape::kN);
  //
  // Iterator
  //
  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kK,
                                        ThreadblockShape::kM>,
      kThreads, cutlass::layout::PitchLinearShape<8, 4>, 8>;
  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kK,
                                        ThreadblockShape::kN>,
      kThreads, cutlass::layout::PitchLinearShape<8, 4>, 8>;

  using GmemTileIteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kM>,
          typename Mma::ElementA, cutlass::layout::RowMajor, 1, ThreadMapA>;

  using GmemTileIteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          typename Mma::ElementB, cutlass::layout::ColumnMajor, 0, ThreadMapB>;

  using SmemTileIteratorA =
      cutlass::transform::threadblock::RegularTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          typename Mma::ElementA, typename Mma::LayoutA, 1, ThreadMapA>;

  using SmemTileIteratorB =
      cutlass::transform::threadblock::RegularTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          typename Mma::ElementB, typename Mma::LayoutB, 0, ThreadMapB>;

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;

  using BlockFragmentA = typename GmemTileIteratorA::Fragment;
  using BlockFragmentB = typename GmemTileIteratorB::Fragment;

  BlockFragmentA fragmentA;
  BlockFragmentB fragmentB;
  FragmentC accum;
  accum.clear();

  typename GmemTileIteratorA::Params ParamsA({WholeShape::kK});
  typename GmemTileIteratorB::Params ParamsB({WholeShape::kK});

  cutlass::Coord<2> ExtentA =
      cutlass::make_Coord(WholeShape::kM, WholeShape::kK);
  cutlass::Coord<2> ExtentB =
      cutlass::make_Coord(WholeShape::kK, WholeShape::kN);

  //
  // data pointer
  //

  // block tile address in gmem
  typename Mma::ElementA* tmp_block_A;
  typename Mma::ElementB* tmp_block_B;
  typename Mma::ElementC* tmp_block_C;

  // C block tile address
  tmp_block_C = output_C + cRow * ThreadblockShape::kM * WholeShape::kN +
                cCol * ThreadblockShape::kN;
  // C warp tile address
  typename Mma::ElementC* tmp_warp_c =
      tmp_block_C + warpRow * Mma::Shape::kM * WholeShape::kN +
      warpCol * Mma::Shape::kN;

  typename Mma::LayoutC layout_C =
      Mma::LayoutC::packed({WholeShape::kM, WholeShape::kN});

  for (uint bkIdx = 0; bkIdx < WholeShape::kK; bkIdx += ThreadblockShape::kK) {
    tmp_block_A =
        input_A + cRow * ThreadblockShape::kM * WholeShape::kK + bkIdx;
    tmp_block_B =
        input_B + cCol * ThreadblockShape::kN * WholeShape::kK + bkIdx;

    GmemTileIteratorA GmemIteratorA(ParamsA, tmp_block_A, ExtentA, threadIdx.x);
    GmemTileIteratorB GmemIteratorB(ParamsB, tmp_block_B, ExtentB, threadIdx.x);

    SmemTileIteratorA SmemIteratorA(
        typename SmemTileIteratorA::TensorRef(
            {smem_buffer_A.data(),
             SmemTileIteratorA::Layout::packed(
                 {ThreadblockShape::kM, ThreadblockShape::kK})}),
        threadIdx.x);
    SmemTileIteratorB SmemIteratorB(
        typename SmemTileIteratorB::TensorRef(
            {smem_buffer_B.data(),
             SmemTileIteratorB::Layout::packed(
                 {ThreadblockShape::kK, ThreadblockShape::kN})}),
        threadIdx.x);

    GmemIteratorA.load(fragmentA);
    SmemIteratorA.store(fragmentA);

    GmemIteratorB.load(fragmentB);
    SmemIteratorB.store(fragmentB);
    __syncthreads();

    typename Mma::ElementA* tmp_warp_A =
        smem_buffer_A.data() + warpRow * Mma::Shape::kM * ThreadblockShape::kK;
    typename Mma::ElementB* tmp_warp_B =
        smem_buffer_B.data() + warpCol * Mma::Shape::kN * ThreadblockShape::kK;

    typename Mma::LayoutA layout_A =
        Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
    typename Mma::LayoutB layout_B =
        Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
    typename Mma::IteratorA iter_A({tmp_warp_A, layout_A},
                                   cutlass::arch::LaneId());
    typename Mma::IteratorB iter_B({tmp_warp_B, layout_B},
                                   cutlass::arch::LaneId());

    FragmentA frag_A;
    FragmentB frag_B;

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK; k += Mma::Policy::MmaShape::kK) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);
      ++iter_A;
      ++iter_B;

      mma(accum, frag_A, frag_B, accum);
    }
    __syncthreads();
  }
  typename Mma::IteratorC iter_C({tmp_warp_c, layout_C},
                                 cutlass::arch::LaneId());
  iter_C.store(accum);
}

template <typename Mma, typename ThreadblockShape, typename WholeShape>
float CutlassGemm(typename Mma::ElementC* dC, typename Mma::ElementA* const dA,
                  typename Mma::ElementB* const dB) {
  using Shape = typename Mma::Shape;
  const uint block_m = CEIL_DIV(WholeShape::kM, ThreadblockShape::kM);
  const uint block_n = CEIL_DIV(WholeShape::kN, ThreadblockShape::kN);

  const uint warp_m = CEIL_DIV(ThreadblockShape::kM, Shape::kM);
  const uint warp_n = CEIL_DIV(ThreadblockShape::kN, Shape::kN);
  const uint kWarpSize = 32;

  dim3 gridDim(block_n, block_m);
  const int threads = kWarpSize * warp_m * warp_n;
  dim3 blockDim(threads, 1, 1);

  GemmKernel<Mma, ThreadblockShape, WholeShape, threads>
      <<<gridDim, blockDim>>>(dC, dA, dB);

  return 0.;
}
