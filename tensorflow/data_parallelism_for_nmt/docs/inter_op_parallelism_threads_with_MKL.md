[TOC]

# `inter_op_parallelism_threads` with MKL

## Problem Analysis

Previous experiments on [totally independent model replica](https://github.com/lcy-seso/dl_framework/blob/master/tensorflow/data_parallelism_for_nmt/docs/test_independent_model_replica.md) has a very mysterious results:

>when compiling TensorFlow with Intel MKL support, the acceleratoin for the independent model replica is way below the theoretically linear speedup ratio."

Let me first summarize the exact settings that will cause the above problem:

1. compile TensorFlow with the option: `--config=mkl`.
1. invoke `session.run` with configuration : `config.inter_op_parallelism_threads = 0`.
1. the entire model is running on GPU device.

Let's see more details:

1. _**What is the behavior of config.inter_op_parallelism_threads = 0 (the default setting) ?**_

    * When `config.inter_op_parallelism_threads` is set to 0, TensorFlow will automatically determine the size of the thread pool which is equal to the logical CPU core numbers.
    * For example, the P100 machine I use has two physical CPU, 14 CPU cores, and 56 logical CPU cores (hyper threading is enabled), when `config.inter_op_parallelism_threads = 0`, there will be 56 threads in the thread pool.

2. _**How tensorflow gets size of the thread pool if users do not explictly set it ?**_

    * TensorFlow invoke [CPU_COUNT](http://www.linuxcertif.com/man/3/CPU_COUNT/) to get how many schedulable CPUs are there.
    * **NOTE that**: the affinity mask affacts the returned value of `CPU_COUNT`. TensorFlow does not set affinity mask in all its codes.

**BUT the behavior of config.inter_op_parallelism_threads = 0 changes when TensorFlow compiled with MKL enable according to conditional compiles**. You can check [this code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/process_util.cc#L68) for more details.

Now, let's conclude what the exact reason for this problem is:

>_**MKL has thread pools internally using OMP threads. When compiling with MKL support, tensorflow will set inter_op_parallelism_threads conservatively to avoid thread oversubscription. Even though all the operators in our model are executed in GPU that MKL library is not used at all, size of the thread pool is set to 2. All the GPU operators are launched from CPU. All of the blocking operations in TensorFlow are enqueued on a pool whose size is determined by inter_op_parallelism_threads and now inter_op_parallelism_threads is equal to 2. This is the reason why the speedup ratio is affected.**_

More explanation：

* When MKL library is loaded, the returned value of `CPU_COUNT` will become 1 instead of 56.
  * It seems that the MKL library set the affinity mask. (??)

* TensorFlow uses the following codes to determine the size of thread pool when compiling with MKL support  (In the codes below `mkl_intra_op` is a constant value 1):

  ```cpp
  const int32 mkl_inter_op = std::max(
    (port::NumSchedulableCPUs() + mkl_intra_op - 1) / mkl_intra_op, 2);
  Consequently:
  ```

If:
1. TensorFlow is compiled with `MKL` support；
1. users do not set `config.inter_op_parallelism_threads`；
1. the model does not use `MKL` library at all.

Then, the default thread pool is very small with a value 2.  But the default settings will work well for CPU models.

- **Solution**

  The solution is quite easy: _**when complied with MKL and the model is totally executed on GPU devices, adjust `intra_op_parallelism_threads`.**_

---

In the test below:

1. **`intra_op_parallelism_threads` is set to 56 (the number of logical CPU cores(.**
2. use WMT 14 En-De dataset.

## Parameter Server mode

### cudnn_lstm

Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=360 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
    --variable_update="parameter_server" \
    2>&1 | tee train.log
```

|GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
|:--|:--|:--|:--|:--|
|1|360|28.418|50925.421||
|2|360|33.160|87286.751|1.71|
|3|360|34.383|126272.785|2.48|
|4|360|35.049|165161.244|3.24|
|5|360|37.526|192827.671|3.79|
|6|360|41.251|210497.859|4.13|
|7|360|44.519|227553.456|4.47|
|8|360|43.847|264045.800|5.18|

### dynamic_rnn

Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=360 \
    --variable_update="parameter_server" \
    2>&1 | tee train.log
```

|GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
|:--|:--|:--|:--|:--|
|1|360|38.542|37548.935||
|2|360|43.843|66017.246|1.76|
|3|360|48.183|90105.618|2.40|
|4|360|52.343|110593.685|2.95|
|5|360|57.860|125059.488|3.33|
|6|360|64.224|135201.853|3.60|
|7|360|69.294|146193.808|3.89|
|8|360|73.799|156880.228|4.18|

## All reduce with dynamic rnn

Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
  --batch_size=360 \
  --variable_update="replicated" \
  --gradient_repacking=4 \
  --all_reduce_spec="nccl" \
  --agg_small_grads_max_bytes=0 \
  --agg_small_grads_max_group=10 \
  2>&1 | tee train.log
```

|GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
|:--|:--|:--|:--|:--|
|1|360|38.828|37272.324||
|2|360|44.752|64676.016|1.74|
|3|360|49.735|87294.230|2.34|
|4|360|54.620|105983.110|2.84|
|5|360|61.491|117674.826|3.16|
|6|360|71.865|120827.291|3.24|
|7|360|81.590|124162.210|3.33|
|8|360|87.747|131943.589|3.51|
