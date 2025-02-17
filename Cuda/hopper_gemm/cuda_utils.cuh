#pragma once

namespace utils {
float rand_float(float a = 1e-3, float b = 1) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

__forceinline__ unsigned int ceil_div(int a, int b) { return (a + b - 1) / b; }
}  // namespace utils
