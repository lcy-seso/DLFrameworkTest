#pragma once
#include <iomanip>

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

bool FloatEqual(float a, float b, float epsilon = 1e-3) {
  return fabs(a - b) < epsilon;
}
