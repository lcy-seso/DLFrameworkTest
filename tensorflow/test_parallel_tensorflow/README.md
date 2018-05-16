# Test data parallelism on NMT model

## Test Settings

### Environment

- **Test machine 1**
  - TensorFlow r1.8 compiled by gcc 4.9 with Cuda 9.0, cudnn 7.1, NCCL 2.1
    - Cuda driver version: 390.30
  - OS: Ubuntu 16.04.2
  - Tesla P100-PCIE-16GB
    - 8 GPU cards on one machine.

- **Test machine 2**
  - TensorFlow r1.8 compiled by gcc 4.9, with Cuda 9.0, cudnn 7.1, NCCL 2.1
    - Cuda driver version: 384.130
  - OS: Ubuntu 16.04.2
  - GTX Titan
    - 3 GPU cards on one machine

### Test details

- Disable training data shuffle.
- Run 100 mini-batches and count words per second.
- Test codes are base on this commit: [a0a40f0](https://github.com/lcy-seso/dl_framework/tree/a0a40f065ecf9ceb36061c1ef2d749327d051da8/tensorflow/test_parallel_tensorflow)

### Results

- Exp1 on test machine 1

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|35.774|17575.180|
    |2|128|47.777|26307.289|1.496|
    |3|128|65.451|28664.074|1.631|
    |4|128|79.782|31361.289|1.784|
    |5|128|95.910|32612.374|1.8556|
    |6|128|107.257|34947.664|1.988|
    |7|128|125.969|34777.809|1.979|
    |8|128|142.747|34930.192|1.987|

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|64|26.311|11857.973||
    |2|64|38.632|16275.316|1.373|
    |3|64|52.187|18098.961|1.526|


  - **Also terrible GPU utilization is found when the number of GPU cards increase.**

- Exp2 on test machine 2

  >The GPU memory for GTX Titan is limited (6G). It cannot run the model by using a large batch size like 128, so batch size in this test is decreased to 64.

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|64|50.850|6135.506|
    |2|64|59.803|10513.544|1.714|
    |3|64|65.097|14509.607|2.365|

## How to profile

### use TensorFlow profiler

1. Compile `tfprof` (for more details, please follow [this documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/command_line.md#build-tfprof))

    - Execute the following command under the tensorflow source directory:

        ```bash
        bazel build --config opt tensorflow/core/profiler:profiler
        ```

    - After the building process is finished, the profiler can be found in: `bazel-bin/tensorflow/core/profiler`.

1. generate timeline vitualization file:

    ```bash
    tfprof> graph -step -1 -max_depth 100000 -output timeline:outfile=<filename>
    ```

### virtualize the execution timeline

- on test machine 1, run 5 batches, here virtualize the timeline for batch 3.
