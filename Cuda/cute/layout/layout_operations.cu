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
  Layout vec = Layout<_16, _3>{};  // row vector

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
  Layout a = make_layout(4, 1);
  std::cout << "size(a) = " << size(a) << "; cosize(a) = " << cosize(a)
            << std::endl;

  Layout b = make_layout(4, 2);
  std::cout << "size(b) = " << size(b) << "; "
            << "cosize(b) = " << cosize(b) << std::endl
            << std::endl;
  auto layout = complement(b, 24);
  print_layout(layout);
}

void test5() {
  // product
  Layout a = Layout<Shape<_2, _2>, Stride<_1, _2>>{};
  Layout b = Layout<Shape<_3, _4>, Stride<_4, _1>>{};

  Layout c = raked_product(a, b);
  print_layout(a);
  print_layout(b);
  print_layout(c);

  Layout d1 = right_inverse(c);
  std::cout << std::endl
            << "right inverse(c):" << std::endl
            << std::endl
            << d1 << std::endl;
  Layout d2 = d1.with_shape(make_shape(size(a), size(b)));
  print_layout(d2);
}

void test6() {
  // product
  Layout a = Layout<Shape<_2, _4>, Stride<_4, _1>>{};
  Layout b = Layout<Shape<_1, _4>, Stride<_0, _1>>{};

  Layout c = raked_product(a, b);
  print_layout(a);
  print_layout(b);
  print_layout(c);

  Layout d1 = right_inverse(c);
  std::cout << std::endl
            << "right inverse(c):" << std::endl
            << std::endl
            << d1 << std::endl;
  Layout d2 = d1.with_shape(make_shape(size(a), size(b)));
  print_layout(d2);
}

int main(int argc, char** argv) {
  // test1();
  // test2();
  // test3();
  // test4();
  // test5();

  test6();

  return 0;
}
