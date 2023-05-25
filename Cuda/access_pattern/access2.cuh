#pragma once

template <typename T, int N>
struct GetPackType {
  using type =
      typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template <typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
  const SRC* src;
  int64_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template <int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] = static_cast<DST>(src[i]);
    }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

template <typename T, typename LOAD, typename STORE, int GRID_SIZE,
          int BLOCK_SIZE, int pack_size>
__global__ void CopyTest2(LOAD load, STORE store, int height, int width) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<T*>(shared_buf);
  const int tid = threadIdx.x;

  const int num_packs = width / pack_size;
  for (int64_t row = blockIdx.x; row < height; row += GRID_SIZE) {
    // Load into shared memory
    for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE) {
      T pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
      }
    }

    // Store to output
    for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE) {
      T pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = buf[i * num_packs + pack_id];
      }

      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}
