#include <cute/tensor.hpp>
#include <cutlass/util/print_error.hpp>

using namespace cute;

void test1() {
  auto layout1 = make_layout(make_shape(2, 2), make_stride(2, 1));
  print_layout(layout1);
  std::cout << std::endl;

  auto layout2 = make_layout(make_shape(2, 4), make_stride(4, 1));
  print_layout(layout2);
  std::cout << std::endl;
}

void test2() {
  Layout layout1 = Layout<Shape<_2, _2>, Stride<_1, _2>>{};
  Layout layout2 = Layout<Shape<_3, _4>, Stride<_4, _1>>{};

  std::cout << "blocked product: " << std::endl;
  auto rv_layout1 = blocked_product(layout1, layout2);
  print_layout(rv_layout1);
  std::cout << std::endl;

  std::cout << "logical product: " << std::endl;
  auto rv_layout2 = logical_product(layout1, layout2);
  print_layout(rv_layout2);
  std::cout << std::endl;

  std::cout << "raked product: " << std::endl;
  auto rv_layout3 = raked_product(layout1, layout2);
  print_layout(rv_layout3);
  std::cout << std::endl;
}

void test3() {
  Layout vec = Layout<_16, _3>{};  // row vector

  Layout col1 = Layout<_4, _1>{};  // col vector
  Layout mat1 = logical_divide(vec, col1);
  print_layout(mat1);
  std::cout << std::endl;

  Layout col2 = Layout<_4, _4>{};           // row vector
  Layout mat2 = logical_divide(vec, col2);  // row vector
  print_layout(mat2);
  std::cout << std::endl;

  Layout col3 = Layout<_4, _2>{};           // row vector
  Layout mat3 = logical_divide(vec, col3);  // row vector
  print_layout(mat3);
  std::cout << std::endl;

  Layout col4 = Layout<Shape<_2, _2>, Stride<_4, _1>>{};  // row vector
  Layout mat4 = logical_divide(vec, col4);                // row vector
  print_layout(mat4);
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  // test1();
  // test2();
  test3();

  return 0;
}