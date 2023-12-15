对一个二维空间的行和列进行swizzle

$\text{swizzle}(B,M,S)$

- $2^M$ 个元素为一组
- swizzle的二维空间有 $2^B$ 行
- swizzle的二维空间中 $2^S$ 个元素为一列

每个线程用向量化指令访问128b数据，$128 / 32 = 4 \ \text{bank}$，每个线程访问4个bank，8个线程访问一条shared memory cache line。

1. 当数据类型是半精度时，$M=3$，因为$2^3=8 \times 16 = 128 \ \text{b}$，128 b 访存指令读取8个元素，这些元素为一组。
1. $S = 3$，1024 / 128 = 8，8个线程访问一整条shared memory cache line
1.  假如原始输入数据有形状，在内存中连续的维度是64，并且数据类型为半精度，$64 \times 16 / 1024 = 1$，一个连续维度就占1个shared memory cache line。因此

swizzle<3, 3, 3>


# Reference

1. [What does bitwise XOR (exclusive OR) mean?](https://stackoverflow.com/questions/6398427/what-does-bitwise-xor-exclusive-or-mean)
1. [DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)
1. [cute 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)
