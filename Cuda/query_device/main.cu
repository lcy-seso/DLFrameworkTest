#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
  int device_id = 0;  // ID of the GPU device to query
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  int reservedShared = prop.reservedSharedMemPerBlock;
  printf("Reversed Shared Memory per Block: %d bytes\n", reservedShared);

  return 0;
}
