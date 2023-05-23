#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#define CHECK(call)                                                        \
  {                                                                        \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess) {                                            \
      printf("Error: %s: %d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason : %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                             \
    }                                                                      \
  }

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

namespace {
__device__ unsigned int mutex = 0U;
__device__ unsigned int count = 0U;
__device__ void Lock() {
  // NOTE: DO NOT call this funcion in multiple threads. Performance can degrade
  // or cause deadlock when many threads attempt to perform atomic operations.
  while (atomicCAS(&mutex, 0U, 1U) != 0U)
    ;
}
__device__ void Unlock() { atomicExch(&mutex, 0U); }
}  // namespace

template <typename T>
struct Sum {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a > b ? a : b;
  }
};

template <typename T>
struct Min {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a > b ? b : a;
  }
};

template <typename T>
struct Prod {
  __forceinline__ __host__ __device__ T operator()(const T& a,
                                                   const T& b) const {
    return a * b;
  }
};

template <typename T>
__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, T val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}

template <typename T, typename Reducer>
__forceinline__ __device__ T WrapReduce(T val, unsigned mask, Reducer reducer) {
  val = reducer(val, CudaShuffleDownSync(mask, val, 16));
  val = reducer(val, CudaShuffleDownSync(mask, val, 8));
  val = reducer(val, CudaShuffleDownSync(mask, val, 4));
  val = reducer(val, CudaShuffleDownSync(mask, val, 2));
  return reducer(val, CudaShuffleDownSync(mask, val, 1));
}

// Works only for power-of-2 arrays.
template <typename T, typename Reducer>
__device__ T Power2Reduce(T val, int tid, T* shm, Reducer reducer,
                          int blockSize, T init_val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);
  val = WrapReduce(val, mask, reducer);

  if (tid < warpSize) shm[tid] = init_val;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);
  if (tid < warpSize) {
    val = shm[tid];
    val = WrapReduce(val, mask, reducer);
  }
  return val;
}

// Works for inputs with an arbitrary size.
// Reduces inputs to a scalar in shared memory.
template <typename T, typename Reducer>
__device__ void BlockScalarReduceKernel(unsigned int num_elements, bool is_pow2,
                                        const T* I, T* O, Reducer reducer,
                                        int block_size, T init_val) {
  const int kWarpSize = 32;
  __shared__ T shm[kWarpSize];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (block_size * 2) + threadIdx.x;
  unsigned int grid_size = block_size * 2 * gridDim.x;

  T val = init_val;
  while (i < num_elements) {
    val = reducer(val, I[i]);
    if (is_pow2 || i + block_size < num_elements)
      val = reducer(val, I[i + block_size]);
    i += grid_size;
  }
  __syncthreads();

  val = Power2Reduce(val, tid, shm, reducer, block_size, init_val);

  if (tid == 0) O[0] = val;
}

template <typename T, typename Reducer>
__global__ void MultiBlockScalarReduceKernel(unsigned int num_elements,
                                             bool is_pow2, const T* I, T* O,
                                             Reducer reducer, int block_size,
                                             T init_val, T scale) {
  __shared__ T partial_sum;

  BlockScalarReduceKernel(num_elements, is_pow2, I, &partial_sum, reducer,
                          block_size, init_val);

  if (gridDim.x > 1) {
    if (threadIdx.x == 0) {
      // TODO(Ying): Use atomic lock may not be optimal. Other choices are
      // synchronizing by using global memory, or launch 2 kernels.
      Lock();
      O[0] = reducer(O[0], partial_sum);
      Unlock();

      if (atomicInc(&count, gridDim.x) != (gridDim.x - 1)) return;
      // The last thread multiplys the scale factor.
      O[0] *= scale;
      count = 0;
    }
  } else
    O[0] = partial_sum * scale;
}

// Works only if there are <= 16 columns.
// Each warp sums over multiple rows at once.
template <typename T, typename Reducer>
__device__ void ColumnReduceMax16ColumnsKernel(const T* I, volatile T* O,
                                               int num_rows, int num_cols,
                                               Reducer reducer, T init_val,
                                               T scale) {
  // NOTE: The output memory `O` MUST be global memory whose size is at least
  // gridDim.y * 16. Here 16 is the max column number.
  const int kWarpSize = 32;
  const int kMaxColNum = 16;

  const int rows_per_warp = kWarpSize / num_cols;
  int stride = rows_per_warp * blockDim.y * gridDim.y;

  const int lane = threadIdx.x % kWarpSize;
  const int lane_row = lane / num_cols;

  const int start_row_warp =
      rows_per_warp * (blockIdx.y * blockDim.y + threadIdx.y);
  const int start_row_lane = start_row_warp + lane_row;

  int row = start_row_lane;   // The actual row index in the input matrix.
  int col = lane % num_cols;  // The actual column index in the input matrix.

  T val = init_val;
  int global_pos = row * num_cols + col;
  if (global_pos < num_rows * num_cols) val = I[global_pos];

  row += stride;
  for (; row < num_rows; row += stride) {
    global_pos = row * num_cols + col;
    if (global_pos < num_rows * num_cols) val = reducer(val, I[global_pos]);
  }

  const int rows_in_this_warp = min(rows_per_warp, num_rows - start_row_warp);
  for (int i = 1; i < rows_in_this_warp; ++i) {
    T tmp = CudaShuffleSync(0xffffffff, val,
                            static_cast<int>(threadIdx.x + i * num_cols));
    if (lane < num_cols) val = reducer(val, tmp);
  }  // Up to now, threads (lane 0 ~ column) in a warp do reduction over
     // `rows_in_this_warp` rows.

  __shared__ T partial_sums[kWarpSize * (kWarpSize + 1)];
  if (lane < num_cols) partial_sums[lane * (kWarpSize + 1) + threadIdx.y] = val;
  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < num_cols) {
    T s = partial_sums[threadIdx.x * (kWarpSize + 1)];

    if (blockDim.y > 1) {
      for (int row = 1; row < blockDim.y; ++row) {
        T t = partial_sums[threadIdx.x * (kWarpSize + 1) + row];
        s = reducer(s, t);
      }
    }

    O[col + blockIdx.y * kMaxColNum] = s * scale;
  }
}

template <typename T, typename Reducer>
__global__ void MultiBlockColumnReduceMax16ColumnsKernel(const T* I, T* O,
                                                         int num_rows,
                                                         int num_cols,
                                                         Reducer reducer,
                                                         T init_val, T scale) {
  const int kMaxColNum = 16;

  ColumnReduceMax16ColumnsKernel(I, O, num_rows, num_cols, reducer, init_val,
                                 gridDim.y > 1 ? 1 : scale);

  if (gridDim.y > 1) {
    __threadfence();
    __shared__ bool isLast;

    if (threadIdx.x == 0 && threadIdx.y == 0)
      isLast = (atomicInc(&count, gridDim.x) == gridDim.y - 1);
    __syncthreads();

    if (isLast) {
      if (threadIdx.x == 0 && threadIdx.y < num_cols) {
        O[threadIdx.y] =
            scale * reducer(O[threadIdx.y], O[threadIdx.y + kMaxColNum]);

        for (int i = 2; i < gridDim.y; ++i)
          O[threadIdx.y] =
              reducer(O[threadIdx.y], O[threadIdx.y + i * kMaxColNum]);
      }

      count = 0;
    }
  }
}

template <typename T, typename Reducer>
__global__ void ColumnReduceSimpleKernel(const T* I, T* O, int num_rows,
                                         int num_cols, Reducer reducer,
                                         T scale) {
  // TODO(Ying) A simple implementation that sssigns one column to a thread.
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  const int plane = gid / num_cols;
  const int col = gid % num_cols;

  if (plane >= 1) return;

  if (num_rows == 1) {
    // Only one row, no need to reduce. Equal to Identity mapping.
    O[plane * elems_per_plane + col] = I[plane * elems_per_plane + col];
    return;
  }

  T val = reducer(I[plane * elems_per_plane + col],
                  I[plane * elems_per_plane + num_cols + col]);
  for (int row = 2; row < num_rows; ++row)
    val = reducer(val, I[plane * elems_per_plane + row * num_cols + col]);

  O[plane * num_cols + col] = val * scale;
}

// Assigns one block to reduce over one row.
template <typename T, typename Reducer>
__global__ void RowReduceKernel(const T* I, T* O, int num_rows, int num_cols,
                                int block_size, Reducer reducer, T init_val,
                                T scale) {
  // TODO(Ying) A simple implementation. Not optimized for colums numbers are
  // less than 32 and 16.

  int tid = threadIdx.x;
  int next_idx = blockIdx.x * num_cols + tid;
  int cur_idx = tid;

  T val = init_val;
  for (; cur_idx < num_cols; next_idx += block_size, cur_idx += block_size)
    val = reducer(val, I[next_idx]);
  __syncthreads();

  const int kWarpSize = 32;
  __shared__ T shm[kWarpSize];
  val = Power2Reduce(val, tid, shm, reducer, block_size, init_val);

  if (tid == 0) O[blockIdx.x] = val * scale;
}
#endif
