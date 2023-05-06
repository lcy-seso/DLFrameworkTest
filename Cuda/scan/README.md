# Inclusive and exclusive scan

inclusive scan computes: $y_i = \bigoplus_{j=0}^{i}x_i$
exclusive scan computes: $y_i = \bigoplus_{j=0}^{i-1}x_i$

initial number is 0.

|input|1|2|3|4|5|6|7|8|9|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|**Inclusive scan**|1|3|6|10|15|21|28|36|45|
|**Exclusive scan**|0|1|3|6|10|15|21|28|36|

# References

1. [tensorflow scan ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/scan_ops_gpu.h#L169)
1. [Prefix sum](https://en.wikipedia.org/wiki/Prefix_sum)
1. [Parallel Prefix Sum (Scan) with CUDA](https://github.com/TVycas/CUDA-Parallel-Prefix-Sum)
1. https://github.com/zzhbrr/parallel-scan-prefix-sum
