#pragma once

#include <iomanip>
#include <iostream>

void print_matrix(const __half* data, const int kM, const int kN) {
  std::cout << "Matrix: [" << kM << ", " << kN << "]" << std::endl;
  std::cout << std::fixed << std::setprecision(2);

  for (int i = 0; i < kM * kN; ++i) {
    std::cout << __half2float(data[i]) << ", ";

    if ((i + 1) % 16 == 0) std::cout << std::endl;
  }
}
