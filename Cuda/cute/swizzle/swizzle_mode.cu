#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include <iostream>
using namespace cute;

int main() {
  /// K-major
  std::cout << "K-major:" << std::endl;
  {
    std::cout << "128B swizzle for half(K-major):" << std::endl << "\t";
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_K_SW128_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "128B swizzle for float(K-major):" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_K_SW128_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "64B swizzle for half(K-major):" << std::endl;
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_K_SW64_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "64B swizzle for float:" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_K_SW64_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "32B swizzle for half:" << std::endl;
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_K_SW32_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "32B swizzle for float:" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_K_SW32_Atom<DType>;
    print(SwizzleAtom{});
  }

  /// MN-major
  std::cout << std::endl << std::endl << "MN-major:" << std::endl;
  {
    std::cout << "128B swizzle for half:" << std::endl;
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_MN_SW128_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "128B swizzle for float:" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_MN_SW128_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "64B swizzle for half:" << std::endl;
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_MN_SW64_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "64B swizzle for float:" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_MN_SW64_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "32B swizzle for half:" << std::endl;
    using DType = cutlass::half_t;
    using SwizzleAtom = GMMA::Layout_MN_SW32_Atom<DType>;
    print(SwizzleAtom{});
  }

  {
    std::cout << std::endl << "32B swizzle for float:" << std::endl;
    using DType = float;
    using SwizzleAtom = GMMA::Layout_MN_SW32_Atom<DType>;
    print(SwizzleAtom{});
  }

  return 0;
}
