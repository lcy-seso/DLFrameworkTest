// Load a tile from global memory to shared memory.
// Note: the shared memory layout is swizzled.
template <typename Shape_, typename Element_>
struct TileLoader {
  using Element = Element_;
};
