#include <iostream>

/* ==================[Data]: Register Level ===================*/
typedef struct tile_reg {
  // how much data are stored in the thread's local register file
  // how many 128-bit registers are used for each thread
  uint32_t tile[2];  // this requires user to specify
} tile_reg;
/* ======================= [Data] ============================*/

/* =========[Thread]: thread behaviors dictated by hardware ======
 * used by tensor core's warp collaborative instruction
 * corresponds to the shape of Warp Tile. */

typedef struct tile_8_threads {
  tile_reg tile[8];  // fixed, can be inferred
} tile_8_threads;

typedef struct tile_warp {
  tile_8_threads tile[2][2];  // fixed, can be inferred
} tile_warp;
/* ============================ [Thread] =======================*/

/* ======[Thread]: thread behaviors needs to be configurated =====
 * all possible values for warp arrangement can be enumerate:
 * 1 x 1 -> 32 threads
 * 1 x 2 -> 64 threads
 * 2 x 1 -> 64 threads
 * 2 x 2 -> 128 threads
 * 2 x 4 -> 256 threads
 * 4 x 2 -> 256 threads
 * 4 x 4 -> 512 threads
 * 4 x 8 -> 1024 threads
 * 8 x 4 -> 1024 threads
 * warp arrangement corresponds to the shape of the Shared Memory Tile */

typedef struct spatial_tile_thread_block {
  // 2 x 2 warp arrangement, shared memory tile canbe automatically inferred
  tile_warp tile[2][2];  // use 2 x 2 warp arrangement
} spatial_tile_thread_block;

typedef struct temporal_tile_thread_block {
  // 2 x 2 warp arrangement, shared memory tile canbe automatically inferred
  // can be 1-D or 2-D, depends on how many temporal dimension are used
  spatial_tile_thread_block tile[4];
} temporal_thread_block;

/* ==============================================================*/

int main() {
  temporal_tile_thread_block tile;  // fixed, can be inferred
  return 0;
}
