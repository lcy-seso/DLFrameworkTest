#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

#include <bitset>
#include <iomanip>

using namespace cute;

void test(int kRows, int kCols) {
  auto row_major = Layout<Shape<_4, _8>, Stride<_8, _1>>{};
  using SwizzledRowLayout =
      decltype(composition(Swizzle<2, 0, 3>{}, row_major));
  SwizzledRowLayout swizzled;

  // using SmemLayout =
  //     decltype(tile_to_shape(SwizzledColumnLayout{}, Shape<Int<8>,
  //     Int<16>>{}));
  // SmemLayout swizzled;

  std::cout << std::endl << "row major: " << std::endl;
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      std::cout << row_major(i, j) << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl << "swizzled layout: " << std::endl;
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      std::cout << swizzled(i, j) << ", ";
    }
    std::cout << std::endl;
  }
}

void swizzled_function(int lane) {
  // 8 threads form a phrase
  int c = lane % 8;
  int s = lane / 8;

  // std::cout << "(s, c) = (" << s << ", " << c << ")" << std::endl;
  // std::cout << "binary c = " << std::bitset<32>(c) << std::endl;

  int smem_row = (c & 1) | ((c >> 1) & 2);
  int bank = ((c << 1) & 4) | s ^ smem_row;

  std::cout << lane << " => " << bank << std::endl;
  if (lane && (lane + 1) % 8 == 0) std::cout << std::endl;

  // int smem_offset = smem_row * ldm_smem + bank;
}

int main() {
  // using Element = cutlass::half_t;
  // const int kRows = 4;
  // const int kCols = 8;
  // int numel = kRows * kCols;

  // test(kRows, kCols);

  for (int i = 0; i < 32; ++i) swizzled_function(i);

  int a = 2;
  std::bitset<32> x(a);
  // std::cout << "x = " << x << std::endl;
}
