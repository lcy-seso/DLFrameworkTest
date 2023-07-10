#include "errors.h"
#include "my_add_kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void my_add_kernel(const scalar_t *input1, const scalar_t *input2,
                              scalar_t *output, int64_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    output[tid] = input1[tid] + input2[tid];
  }
}

at::Tensor my_add_op(const at::Tensor &input1, const at::Tensor &input2,
                     at::Tensor output) {
  TORCH_CHECK(input1.type().is_cuda());
  TORCH_CHECK(input2.type().is_cuda());
  TORCH_CHECK(output.type().is_cuda());

  CHECK_IS_FLOAT(input1);
  CHECK_IS_FLOAT(input2);
  CHECK_IS_FLOAT(output);

  const int64_t numel = input1.numel();
  int block = 256;
  int grid = (numel + block - 1) / block;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      input1.scalar_type(), "add_kernel", ([&] {
        scalar_t *input1_data = input1.data_ptr<scalar_t>();
        scalar_t *input2_data = input2.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();

        my_add_kernel<scalar_t><<<block, grid, 0, stream>>>(
            input1_data, input2_data, output_data, numel);
      }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_add_op", &my_add_op, "My add operator for test.");
}
