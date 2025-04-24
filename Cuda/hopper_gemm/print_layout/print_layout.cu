#include <cute/tensor.hpp>

using namespace cute;

int main() {
  // for Hopper
  using TiledMma =
      TiledMMA<MMA_Atom<SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,
                                                   GMMA::Major::K>>,  // TN gemm
               Layout<Shape<_1, _1, _1>>, Tile<_64, _64, _16>>;
  // print(TiledMma{});
  print_layout(TiledMma::LayoutA_TV{});

  //   using TiledMma = TiledMMA<
  //       MMA_Atom<SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,
  //       GMMA::Major::MN>>, Layout<Shape<_1, _1, _1>>, Tile<_64, _64, _16>>;

  // for Ampere
  // using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
  //                           Layout<Shape<_1, _1, _1>>, Tile<_16, _8, _16>>;

  // using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
  //                           Layout<Shape<_1, _1, _1>>, Tile<_16, _16, _16>>;

  // print_latex(TiledMma{});

  return 0;
}
