#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

/*
 * Given a list A, this kernel function computes exclusive scan of A.
 */
template <typename T, unsigned int blockSize>
__global__ void __launch_bounds__(blockSize)
    blockExclusiveScan(const T* input, T* output, int numel, T* sum) {
  int tid = threadIdx.x;
  int offset = 1;
  int blockOffset = blockSize * blockIdx.x * 2;

  __shared__ T temp[blockSize << 1];

  // Load to shared memory
  if (blockOffset + (tid * 2) < numel) {
    temp[tid * 2] = input[blockOffset + (tid * 2)];
  }
  if (blockOffset + (tid * 2) + 1 < numel) {
    temp[(tid * 2) + 1] = input[blockOffset + (tid * 2) + 1];
  }

  // Build sum in place up the balanced binary tree
  for (int d = blockSize; d > 0; d >>= 1) {
    __syncthreads();

    if (tid < d) {
      int ai = offset * ((tid * 2) + 1) - 1;
      int bi = offset * ((tid * 2) + 2) - 1;
      temp[bi] += temp[ai];  // the combiner
    }
    offset <<= 1;
  }

  if (tid == 0) {
    if (sum) {
      // If doing a FULL scan, save the last value in the SUMS array for
      // later processing
      sum[blockIdx.x] = temp[(blockSize << 1) - 1];
    }
    temp[(blockSize << 1) - 1] = 0.;  // insert the inital value at the root.
  }

  // Traverse down the balanced binary tree
  for (int d = 1; d < blockSize << 1; d <<= 1) {
    offset >>= 1;
    __syncthreads();

    if (tid < d) {
      int ai = offset * ((tid * 2) + 1) - 1;
      int bi = offset * ((tid * 2) + 2) - 1;

      T t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  //	 store, Copy the new array back to global array
  __syncthreads();
  if (blockOffset + (tid * 2) < numel) {
    output[blockOffset + (tid * 2)] = temp[(tid * 2)];
  }
  if (blockOffset + (tid * 2) + 1 < numel) {
    output[blockOffset + ((tid * 2) + 1)] = temp[(tid * 2) + 1];
  }
}

template <typename T, unsigned int blockSize>
__global__ void uniformAdd(const T* incr, T* output, int numel) {
  int index = threadIdx.x + (2 * blockSize) * blockIdx.x;

  int valueToAdd = incr[blockIdx.x];

  // Each thread sums two elements
  if (index < numel) {
    output[index] += valueToAdd;
  }
  if (index + blockSize < numel) {
    output[index + blockSize] += valueToAdd;
  }
}

#define CONFLICT_FREE_OFFSET(a) (((a) / 32))

template <typename T, unsigned int blockSize>
__global__ void __launch_bounds__(blockSize)
    blockExclusiveScanBCA(const T* input, T* output, int numel, T* sum) {
  int tid = threadIdx.x;
  int blockOffset = blockIdx.x * 2 * blockSize;

  __shared__ T temp[(blockSize << 1) + blockSize];

  if (blockOffset + tid * 2 < numel)
    temp[2 * tid + CONFLICT_FREE_OFFSET(2 * tid)] =
        input[blockOffset + (tid * 2)];
  if (blockOffset + 2 * tid + 1 < numel)
    temp[2 * tid + 1 + CONFLICT_FREE_OFFSET(2 * tid + 1)] =
        input[blockOffset + 2 * tid + 1];
  __syncthreads();

  // int times = 1;
  // for (int thread_cnt = blockSize; thread_cnt >= 1;
  //      thread_cnt >>= 1, times <<= 1) {
  //   if (tid < thread_cnt) {
  //     int b = times * 2 * (tid + 1) - 1;
  //     int a = b - times;
  //     temp[b + CONFLICT_FREE_OFFSET(b)] += temp[a + CONFLICT_FREE_OFFSET(a)];
  //   }
  //   __syncthreads();
  // }

  // if (tid == 0)
  //   temp[2 * blockSize - 1 + CONFLICT_FREE_OFFSET(2 * blockSize - 1)] = 0;
  // __syncthreads();

  // for (int thread_cnt = 1; thread_cnt <= blockSize; thread_cnt <<= 1) {
  //   times >>= 1;
  //   if (tid < thread_cnt) {
  //     int b = times * 2 * (tid + 1) - 1;
  //     int a = b - times;
  //     // swap(temp[a], temp[b]);
  //     T tmp = temp[a + CONFLICT_FREE_OFFSET(a)];
  //     temp[a + CONFLICT_FREE_OFFSET(a)] = temp[b + CONFLICT_FREE_OFFSET(b)];
  //     temp[b + CONFLICT_FREE_OFFSET(b)] += tmp;
  //   }
  //   __syncthreads();
  // }

  if (blockOffset + 2 * tid < numel) {
    output[blockOffset + 2 * tid] =
        temp[tid * 2 + CONFLICT_FREE_OFFSET(tid * 2)];
    if (2 * tid == 2 * blockSize - 1)
      sum[blockIdx.x] = temp[tid * 2 + CONFLICT_FREE_OFFSET(tid * 2)] +
                        input[blockOffset + 2 * tid];
  }
  if (blockOffset + 2 * tid + 1 < numel) {
    output[blockOffset + 2 * tid + 1] =
        temp[tid * 2 + 1 + CONFLICT_FREE_OFFSET(tid * 2 + 1)];
    if (2 * tid + 1 == 2 * blockSize - 1)
      sum[blockIdx.x] = temp[tid * 2 + 1 + CONFLICT_FREE_OFFSET(tid * 2 + 1)] +
                        input[blockOffset + 2 * tid + 1];
  }
}

template <typename T, unsigned int blockSize>
__host__ void fullExlusiveScan(const T* input, T* output, int numel) {
  int blocksPerGridL1 = 1 + (numel - 1) / (blockSize * 2);
  int blocksPerGridL2 = 1 + blocksPerGridL1 / (blockSize * 2);
  int blocksPerGridL3 = 1 + blocksPerGridL2 / (blockSize * 2);

  std::cout << "numel = " << numel << "; blockSize = " << blockSize << std::endl
            << "L1 blocks: " << blocksPerGridL1
            << "; L2 blocks: " << blocksPerGridL2
            << "; L1 blocks: " << blocksPerGridL3 << ";" << std::endl;

  T* d_sums_level1 = nullptr;
  T* d_incr_level1 = nullptr;

  T* d_sums_level2 = nullptr;
  T* d_incr_level2 = nullptr;

  if (blocksPerGridL1 == 1) {
    blockExclusiveScan<float, blockSize>
        <<<blocksPerGridL1, blockSize>>>(input, output, numel, nullptr);

  } else if (blocksPerGridL2 == 1) {
    CudaCheck(cudaMalloc((void**)&d_sums_level1, blocksPerGridL1 * sizeof(T)));
    CudaCheck(cudaMalloc((void**)&d_incr_level1, blocksPerGridL1 * sizeof(T)));

    blockExclusiveScan<float, blockSize>
        <<<blocksPerGridL1, blockSize>>>(input, output, numel, d_sums_level1);

    blockExclusiveScan<float, blockSize><<<blocksPerGridL2, blockSize>>>(
        d_sums_level1, d_incr_level1, blocksPerGridL1, nullptr);

    uniformAdd<float, blockSize>
        <<<blocksPerGridL1, blockSize>>>(d_incr_level1, output, numel);

    CudaCheck(cudaFree(d_sums_level1));
    CudaCheck(cudaFree(d_incr_level1));
  } else if (blocksPerGridL3 == 1) {
    std::cout << "level 3 scan" << std::endl;

    CudaCheck(cudaMalloc((void**)&d_sums_level1, blocksPerGridL1 * sizeof(T)));
    CudaCheck(cudaMalloc((void**)&d_sums_level2, blocksPerGridL2 * sizeof(T)));

    CudaCheck(cudaMalloc((void**)&d_incr_level1, blocksPerGridL1 * sizeof(T)));
    CudaCheck(cudaMalloc((void**)&d_incr_level2, blocksPerGridL2 * sizeof(T)));

    // blocked scan
    blockExclusiveScan<float, blockSize>
        <<<blocksPerGridL1, blockSize>>>(input, output, numel, d_sums_level1);
    // printValue<float>(output, numel);

    // level 1, scan
    blockExclusiveScan<float, blockSize><<<blocksPerGridL2, blockSize>>>(
        d_sums_level1, d_incr_level1, blocksPerGridL1, d_sums_level2);

    // printValue<float>(d_sums_level1, blocksPerGridL1);
    printValue<float>(d_sums_level2, blocksPerGridL2);
    printValue<float>(d_incr_level2, blocksPerGridL2);

    std::cout << "blocks L3: " << blocksPerGridL3 << std::endl;
    // level 2, scan
    blockExclusiveScan<float, blockSize><<<blocksPerGridL3, blockSize>>>(
        d_sums_level2 /*input*/, d_incr_level2 /*output*/, blocksPerGridL3,
        nullptr);

    uniformAdd<float, blockSize><<<blocksPerGridL2, blockSize>>>(
        d_incr_level2 /*input*/, d_incr_level1 /*output*/, blocksPerGridL1);
    uniformAdd<float, blockSize><<<blocksPerGridL1, blockSize>>>(
        d_incr_level1 /*input*/, output, numel);

    CudaCheck(cudaFree(d_sums_level1));
    CudaCheck(cudaFree(d_sums_level2));

    CudaCheck(cudaFree(d_incr_level1));
    CudaCheck(cudaFree(d_incr_level2));
  } else {
    printf("The array of length = %d is to large for a level 3 FULL prescan\n",
           numel);
    exit(EXIT_FAILURE);
  }
}

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD,
          BlockScanAlgorithm ALGORITHM>
__global__ void BlockPrefixSumKernel(
    int* d_in,           // Tile of input
    int* d_out,          // Tile of output
    clock_t* d_elapsed)  // Elapsed cycle count of block scan
{
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for
  // coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD,
                    BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;

  // Specialize BlockStore type for our thread block (uses warp-striped loads
  // for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD,
                     BLOCK_STORE_WARP_TRANSPOSE>
      BlockStoreT;

  // Specialize BlockScan type for our thread block
  typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, data);

  // Barrier for smem reuse
  __syncthreads();

  // Start cycle timer
  clock_t start = clock();

  // Compute exclusive prefix sum
  int aggregate;
  BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

  // Stop cycle timer
  clock_t stop = clock();

  // Barrier for smem reuse
  __syncthreads();

  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0) {
    *d_elapsed = (start > stop) ? start - stop : stop - start;
    d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
  }
}
