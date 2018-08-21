#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int& blocks,
                            int& threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2(n / 2) : maxThreads;
    blocks = max(1, n / (threads * 2));
  }

  blocks = min(maxBlocks, blocks);
}
