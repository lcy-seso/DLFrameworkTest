#pragma once

template <typename T, int GRID_SIZE, int BLOCK_SIZE>
__global__ void CopyTest1(const T* input, T* output, int height, int width) {
  // cache the entire row into shared memory
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* sbuf = reinterpret_cast<T*>(shared_buf);

  int tid = threadIdx.x;
  // pack more workload into a CTA
  for (int64_t row = blockIdx.x; row < height; row += GRID_SIZE) {
    int next_idx = row * width + tid;  // element index in input array
    int cur_idx = tid;                 // element index in current row
    for (; cur_idx < width; next_idx += BLOCK_SIZE, cur_idx += BLOCK_SIZE) {
      sbuf[cur_idx] = input[next_idx];
    }
    __syncthreads();

    // Store result into global memory.
    tid = threadIdx.x;

    cur_idx = tid;
    next_idx = row * width + tid;

    for (; cur_idx < width; next_idx += BLOCK_SIZE, cur_idx += BLOCK_SIZE) {
      output[next_idx] = sbuf[cur_idx];
    }
  }
}
