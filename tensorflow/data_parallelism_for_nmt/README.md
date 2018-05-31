# Test data parallelism on NMT model

- **Test machine**
  - TensorFlow r1.8 compiled by gcc 4.9 with Python 3.6, Cuda 9.0, cudnn 7.1, NCCL 2.1
    - Cuda driver version: 390.30
  - OS: Ubuntu 16.04.2
  - Tesla P100-PCIE-16GB
    - 8 GPU cards on one machine.

*All the data are collected in the above machine.*

- [The performance of data parallelism using PS mode](docs/test_ps_mode.md)
- [The performance of data parallelism using all-reduce algorithms](docs/test_allreduce_algorithm.md)
