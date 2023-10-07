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
  // product

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
  // divide
  // Layout vec = Layout<_16, _3>{};  // row vector

  // Layout col1 = Layout<_4, _1>{};  // col vector
  // Layout mat1 = logical_divide(vec, col1);
  // print_layout(mat1);
  // std::cout << std::endl;

  // Layout col2 = Layout<_4, _4>{};           // row vector
  // Layout mat2 = logical_divide(vec, col2);  // row vector
  // print_layout(mat2);
  // std::cout << std::endl;

  // Layout col3 = Layout<_4, _2>{};           // row vector
  // Layout mat3 = logical_divide(vec, col3);  // row vector
  // print_layout(mat3);
  // std::cout << std::endl;

  // Layout col4 = Layout<Shape<_2, _2>, Stride<_4, _1>>{};  // row vector
  // Layout mat4 = logical_divide(vec, col4);                // row vector
  // print_layout(mat4);
  // std::cout << std::endl;

  Layout a = Layout<_24, _2>{};
  Layout b = Layout<_4, _2>{};
  Layout c = logical_divide(a, b);
  print_layout(c);
}

void test4() {
  // complement
  // Layout a = make_layout(4, 1);
  // std::cout << "size(a) = " << size(a) << "; cosize(a) = " << cosize(a)
  //           << std::endl;

  Layout x1 = Layout<_4, _1>{};
  auto y1 = complement(x1, 24);
  std::cout << "y1 = " << y1 << std::endl;

  Layout x2 = Layout<_6, _4>{};
  auto y2 = complement(x2, 24);
  std::cout << "y2 = " << y2 << std::endl;

  Layout b = make_layout(4, 2);
  std::cout << "size(b) = " << size(b) << "; "
            << "cosize(b) = " << cosize(b) << std::endl
            << std::endl;
  auto layout = complement(b, 24);
  print_layout(layout);
}

int main(int argc, char** argv) {
  // test1();
  // test2();
  // test3();
  test4();

  return 0;
}
