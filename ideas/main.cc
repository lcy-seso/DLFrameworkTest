#include <iostream>

/* ==================[Data]: Register Level ===================*/
// how much data are stored in the thread's local register file
// how many 128-bit registers are used for each thread
// stored on thread-local register file
// each thread cannot acceess other thread's register file.

typedef struct tile_reg {
  // this equals to a 1 x 16 halfs
  uint32_t tile[2];  // this requires user to specify
} tile_reg;

// ======================  The decomposition of shared memory tile ============
typedef struct Reg16_ {
  uint32_t tile[1][2];  // 2 128-bit registers, can store 16 halfs
} Reg16_;

// @brief: thread tiles, stores the mapping between thread id and the data
typedef struct EightThreadTile {
  Reg16_ tile[8];  // corresponds to tile shape hold by 8 threads
} EightThreadTile;

typedef struct WarpTile {
  EightThreadTile tile[2][2];  // corresponds tile shape hold by a warp-tile
} WarpTile;

typedef struct AtomicSharedTile {
  // 2 x 2 warp arrangement,
  // shared memory tile canbe automatically inferred
  WarpTile tile[2][2];  // corresponds tile shape hold by a thread-block
} AtomicSharedTile;

typedef struct SharedDecomposition {
  // two temporal dimension are used to expand the atomic shared tile,
  // this results in the final shape of the shared memory tile.
  AtomicSharedTile tile[2][4];
} SharedDecomposition;

/* ==============================================================*/

int main() {
  SharedDecomposition tile;  // fixed, can be inferred
  return 0;
}
