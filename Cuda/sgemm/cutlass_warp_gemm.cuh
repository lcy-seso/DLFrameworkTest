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

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

template <typename ElementType, typename WholeShape, typename ThreadblockShape,
          typename SLayout, const int kThreads>
__global__ void IterateAKernel(ElementType* source, ElementType* target) {
  __shared__ cutlass::AlignedBuffer<cutlass::half_t,
                                    ThreadblockShape::kM * ThreadblockShape::kK>
      smem_buffer;

  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("A shape = [%d, %d]\n", ThreadblockShape::kM, ThreadblockShape::kK);
  }

  // Define the ThreadMap between source and target
  // Access by each thread

  const int kAccessInBits = 128;
  const int element_per_access =
      kAccessInBits / cutlass::sizeof_bits<ElementType>::value;

  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kM,  /*contiguous*/
                                        ThreadblockShape::kK>, /*tile shape*/
      kThreads /*num_threads*/,
      cutlass::layout::PitchLinearShape<4, 8> /*warp arrangement*/,
      element_per_access /*element per access*/>;

  using GTileShape =
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>;
  using GLayout = cutlass::layout::RowMajor;

  // Define source iterator
  using GmemTileIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          GTileShape, ElementType /*type*/, GLayout, 1 /*AdvanceRank*/,
          ThreadMap>;

  // Define target shared memory tile iterator
  using SmemTileIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM,
                           ThreadblockShape::kK> /*shape*/,
      ElementType /*Element*/, SLayout, 1, ThreadMap>;

  using BlockFragment = typename GmemTileIterator::Fragment;

  cutlass::Coord<2> Extent =
      cutlass::make_Coord(WholeShape::kM, WholeShape::kK);

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  typename GmemTileIterator::Params Params({WholeShape::kK});

  BlockFragment fragment;
  fragment.clear();

  ElementType* tmp_block;
  for (uint bkIdx = 0; bkIdx < WholeShape::kK; bkIdx += ThreadblockShape::kK) {
    tmp_block = source + cRow * ThreadblockShape::kM * WholeShape::kK + bkIdx;

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      printf("%d\n", bkIdx);
    }

    GmemTileIterator gIterator(
        Params /*pre-computed parameters*/, tmp_block /*start of tensor*/,
        Extent /*extends*/, threadIdx.x /*threads participated*/);

    SmemTileIterator sIterator(
        typename SmemTileIterator::TensorRef(
            {smem_buffer.data(),
             SmemTileIterator::Layout::packed(
                 {ThreadblockShape::kM, ThreadblockShape::kK})}),
        threadIdx.x);

    gIterator.load(fragment);
    sIterator.store(fragment);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    cutlass::debug::dump_shmem(smem_buffer.raw_data(),
                               ThreadblockShape::kM * ThreadblockShape::kK);
}

template <typename WholeShape /*tile on the low-speed memory*/,
          typename ThreadblockShape, typename WarpShape /**/,
          typename SLayout /*layout on shared memory*/>  // 2D tile
void IterateATest(cutlass::half_t* source, cutlass::half_t* target) {
  const uint block_m = CEIL_DIV(WholeShape::kM, ThreadblockShape::kM);
  const uint block_n = CEIL_DIV(WholeShape::kN, ThreadblockShape::kN);

  const uint warp_m = CEIL_DIV(ThreadblockShape::kM, WarpShape::kM);
  const uint warp_n = CEIL_DIV(ThreadblockShape::kN, WarpShape::kN);
  const uint kWarpSize = 32;

  dim3 gridDim(block_m, block_n);
  const int threads = kWarpSize * warp_m * warp_n;
  std::cout << "grid dim = [" << block_m << ", " << block_n << "], "
            << "threads: " << threads << std::endl;

  IterateAKernel<cutlass::half_t, WholeShape, ThreadblockShape, SLayout,
                 threads><<<gridDim, threads>>>(source, target);
}

/// Test kernel
template <typename Mma, typename ThreadblockShape, typename WholeShape,
          const uint kThreads>
__global__ void GemmKernel(typename Mma::ElementC* output_C,
                           typename Mma::ElementA* const input_A,
                           typename Mma::ElementB* const input_B) {
  int blockSize = blockDim.x;                         // 1D block
  int blockId = blockIdx.y * gridDim.x + blockDim.x;  // 2D grid
  int tid = blockId * blockSize + threadIdx.x;

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;
  const uint warpIdx = threadIdx.x / 32;
  const uint warpRow = warpIdx / (ThreadblockShape::kN / Mma::Shape::kN);
  const uint warpCol = warpIdx % (ThreadblockShape::kN / Mma::Shape::kN);

  // declaration of the shared memory buffers for matrix A, B and C.
  __shared__ cutlass::AlignedBuffer<typename Mma::ElementA,
                                    ThreadblockShape::kM * ThreadblockShape::kK>
      smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<typename Mma::ElementB,
                                    ThreadblockShape::kN * ThreadblockShape::kK>
      smem_buffer_B;

  //   __shared__ cutlass::AlignedBuffer<typename Mma::ElementC,
  //                                     ThreadblockShape::kM *
  //                                     ThreadblockShape::kN>
  //       smem_buffer_C;

  // Declaration of the global memory tile iterators.
  // threads in a thread block collectively load from global memory.
  // Partition the workload into working threads
  const int element_per_access = 8;
  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kK,
                                        ThreadblockShape::kM>, /*tile shape*/
      kThreads /*num_threads*/,
      cutlass::layout::PitchLinearShape<8, 4> /*warp arrangement*/,
      element_per_access /*element per access*/>;
  using ATileShape =
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kM>;
  using GmemTileIteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          ATileShape, typename Mma::ElementA /*type*/,
          cutlass::layout::RowMajor, 1 /*AdvanceRank*/, ThreadMapA>;

  // Declaration
  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kK,
                                        ThreadblockShape::kN>,
      kThreads, cutlass::layout::PitchLinearShape<8, 4>, element_per_access>;
  using GmemTileIteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          typename Mma::ElementB, cutlass::layout::ColumnMajor, 0, ThreadMapB>;

  // shared memory tile iterator
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

  // iterate over the K dimension, the stride is k dimension's block size
  for (uint bkIdx = 0; bkIdx < WholeShape::kK; bkIdx += ThreadblockShape::kK) {
    tmp_block_A =
        input_A + cRow * ThreadblockShape::kM * WholeShape::kK + bkIdx;
    tmp_block_B =
        input_B + cCol * ThreadblockShape::kN * WholeShape::kK + bkIdx;

    GmemTileIteratorA GmemIteratorA(
        ParamsA /*pre-computed parameters*/, tmp_block_A /*start of tensor*/,
        ExtentA /*extends*/, threadIdx.x /*threads participated*/);
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
    if (threadIdx.x == 0 && blockIdx.x == 0)
      printf("\nAll threads dump all the elements:\n");
    cutlass::debug::dump_fragment(fragmentA);

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

template <typename Mma, typename WholeShape, typename ThreadblockShape>
float CutlassGemm(typename Mma::ElementC* dC, typename Mma::ElementA* const dA,
                  typename Mma::ElementB* const dB) {
  using Shape = typename Mma::Shape;
  const uint block_m = CEIL_DIV(WholeShape::kM, ThreadblockShape::kM);
  const uint block_n = CEIL_DIV(WholeShape::kN, ThreadblockShape::kN);

  const uint warp_m = CEIL_DIV(ThreadblockShape::kM, Shape::kM);
  const uint warp_n = CEIL_DIV(ThreadblockShape::kN, Shape::kN);
  const uint kWarpSize = 32;

  dim3 gridDim(block_m, block_n);
  std::cout << "grid dim = [" << block_m << ", " << block_n << "]" << std::endl;
  const int threads = kWarpSize * warp_m * warp_n;
  dim3 blockDim(threads, 1, 1);

  GemmKernel<Mma, ThreadblockShape, WholeShape, threads>
      <<<gridDim, blockDim>>>(dC, dA, dB);

  return 0.;
}
