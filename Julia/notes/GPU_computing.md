# [GPU Computing in the Julia Programming Language](https://devblogs.nvidia.com/gpu-computing-julia-programming-language/)

```julia
using CUDAdrv, CUDAnative

function kernel_vadd(a, b, c)
    i = threadIdx().x
    c[i] = a[i] + b[i]
    return
end

# generate some data
len = 512
a = rand(Int, len)
b = rand(Int, len)

# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda (1,len) kernel_vadd(d_a, d_b, d_c)
c = Array(d_c)
```

the `@cuda` macro:

1. generate specialized code for compiling the kernel function to GPU assembly
1. upload the function to the driver
1. prepare the execution environment

---
* On average, the CUDAnative.jl ports perform identical to statically compiled CUDA C++.
* Julia has recently gained support for **[syntactic loop fusion](https://julialang.org/blog/2017/01/moredots)**, where chained vector operations are fused into a single broadcast.
* Julia features a strong foreign function interface (FFI) for calling into other language environments.

# [Syntactic Loop Fusion](https://julialang.org/blog/2017/01/moredots)

* multiple vectorized operations can be “fused” into a single loop, without allocating any extraneous temporary arrays.
  * *to get good performance both of these snippets should be executed inside some function, not in global scope* (??).
