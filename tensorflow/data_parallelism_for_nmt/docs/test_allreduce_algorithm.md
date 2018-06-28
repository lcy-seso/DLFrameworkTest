[TOC]

_**NOTE: This test runs with `inter_op_parallelism_threads=0` which means TensorFlow automatically determines size of the thread pool. This setting causes problems. See more detail reason and data in [this test](https://github.com/lcy-seso/dl_framework/blob/master/tensorflow/data_parallelism_for_nmt/docs/inter_op_parallelism_threads_with_MKL.md).**_

# Test Allreduce for NMT model

- Data collected based on this commit [95e17f7](https://github.com/lcy-seso/dl_framework/tree/95e17f79d06939b4fc9e588fc084b6f554270640/tensorflow/data_parallelism_for_nmt)

## Test on Tesla P100

- Commandline to run the test:

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

  - _Batch size per GPU card is fixed to be 360 which is the largest batch size can be used on one P100 for the test model._

  |GPU number|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|
  |1|40.211|35990.222|
  |2|47.865|60470.170|1.68|
  |3|60.360|71928.146|2.00|
  |4|70.868|81684.778|2.27|
  |5|84.110|86029.952|2.39|
  |6|101.521|85531.174|2.38|
  |7|113.831|88995.177|2.47|
  |8|123.035|94099.686|2.61|

## Test on GeForce GTX TITAN

### all reduce

- _batch size = 64_

  |GPU number|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|
  |1|26.681|5948.299||
  |2|29.111|10740.958|1.81|
  |3|31.026|14729.752|2.48|

- _batch size = 80_

  |GPU number|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|
  |1|31.678|6175.217||
  |2|33.619|11461.920|1.86|
  |3|36.391|15488.196|2.51|

### PS mode

- _batch size = 64_

  |GPU number|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|
  |1|26.202|6056.890||
  |2|27.833|11234.160|1.85|
  |3|29.745|15364.250|2.54|

- _batch size = 80_

  |GPU number|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|
  |1|31.584|6193.557||
  |2|32.999|11677.075|1.89|
  |3|35.552|15853.687|2.56|
