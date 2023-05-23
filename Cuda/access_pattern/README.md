# 测试

- 输入是一个大小为$[30000, 1024]$的矩阵，使用固定大小的`grid_size`和`block_size`。
    - block size 越大，并行线程数越多，每个线程要做的事情越少 （现在只有load/store）。
    - grid size越大，可以并发执行的blocks数越多，留给硬件进行调度。grid size小的时候，一个CTA以**串行**的方式做更多的工作。<ins>在这个非常简单的kernel里面**串行**远比让硬件自己调度效果差</ins>。
- 输入在Global Memory，thread block中的线程合作将输入load到shared memory，再将shared memory中的数据写回另一块在Global Memory上的输出。shared memory大小等于行的大小。
- 每个CTA处理$\frac{30000}{\text{grid\_size}}$行输入，每个线程处理$\frac{width}{\text{block\_size}}$个数据

[结果](figures/data.tsv)


# 背景

Once a threadblock becomes resident on a SM, it stays on that SM until it retires.
This means, for example, that register allocation in the SM register file for a particular resident thread will remain allocated to that thread until that thread completes execution.

There is no context change associated with blocks. Blocks become resident, then stay resident until they retire. There is obviously a preamble block launch cost and a postamble block retirement cost, but these are unpublished and the only way to discover them would be careful microbenchmarking.

1. [Performance cost of too many blocks](https://forums.developer.nvidia.com/t/performance-cost-of-too-many-blocks/67982/8)
1. [Heuristic block size](https://github.com/jaredhoberock/cuda_launch_config)
1. [see the memory section](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart)
