<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Fix the warm-up problem](#fix-the-warm-up-problem)
    - [Test the ideal case](#test-the-ideal-case)
        - [cudnn_lstm](#cudnnlstm)
        - [dynamic rnn](#dynamic-rnn)
    - [Test Parameter Sever Mode](#test-parameter-sever-mode)
        - [cudnn_lstm](#cudnnlstm)
        - [dynamic rnn](#dynamic-rnn)
    - [Test All-reduce](#test-all-reduce)
    - [dynamic control flow vs. `cudnn_lstm`](#dynamic-control-flow-vs-cudnnlstm)
    - [Summarization](#summarization)

<!-- /TOC -->

# Fix the warm-up problem

Previous test results calculate the processing speed in the following way:

1. Calculate the time to run 30 mini-batches.
1. Divide the total time by the total words processed.

However, I made a stupid mistake. _**The total time is counted from the first mini-batch, which includes lots of memory pool allocation and object creation overheads**_.

In the test below, the first ten mini-batches are for warm up. All the experimental data are collected base on commit:  [156eea6](https://github.com/lcy-seso/dl_framework/commit/156eea6405966916902cbfe8783f33a6e64fa03a#diff-2c39bc19b761ac36dc046245d1d47fe6).

## Test the ideal case

* Each GPU card has balanced computation workload.
* No communication among cards.

### cudnn_lstm

* The command to run the experiment:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
      --batch_size=384 \
      --variable_update="replicated" \
      --independent_replica="true" \
      --use_synthetic_data="true" \
      --src_max_len=100 \
      --num_encoder_layers=4 \
      --num_decoder_layers=4 \
      --encoder_type="cudnn_lstm" \
      --direction="bi" \
      2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|384|27.720|55411.433||
  |2|384|28.461|107938.535|1.95|
  |3|384|28.535|161488.413|2.91|
  |4|384|29.213|210317.094|3.80|
  |5|384|29.139|263568.201|4.76|
  |6|384|28.626|321945.466|5.81|
  |7|384|29.261|367456.175|6.63|
  |8|384|29.228|420416.706|7.59|

### dynamic rnn

* The command to run the experiment:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
      --batch_size=384 \
      --variable_update="replicated" \
      --independent_replica="true" \
      --use_synthetic_data="true" \
      --src_max_len=100 \
      --num_encoder_layers=4 \
      --num_decoder_layers=4 \
      --encoder_type="cudnn_lstm" \
      --direction="bi" \
      2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|384|35.394|43397.602||
  |2|384|36.964|83107.755|1.92|
  |3|384|37.750|122065.760|2.81|
  |4|384|38.179|160924.565|3.71|
  |5|384|38.818|197846.996|4.56|
  |6|384|40.010|230339.877|5.31|
  |7|384|42.132|255199.864|5.88|

## Test Parameter Sever Mode

### cudnn_lstm

* The command to run the experiment:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
      --batch_size=384 \
      --num_encoder_layers=4 \
      --num_decoder_layers=4 \
      --encoder_type="cudnn_lstm" \
      --direction="bi" \
      --prefetch_data_to_device="true" \
      --variable_update="parameter_server" \
      2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|384|27.560|56011.193||
  |2|384|29.075|106185.639|1.90|
  |3|384|29.375|157653.380|2.81|
  |4|384|30.139|204871.508|3.66|
  |5|384|30.888|249886.395|4.46|
  |6|384|31.461|294399.139|5.26|
  |7|384|30.762|351269.919|6.27|
  |8|384|34.244|360629.809|6.44|

### dynamic rnn

* The command to run the experiment:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
      --batch_size=384 \
      --num_encoder_layers=4 \
      --num_decoder_layers=4 \
      --prefetch_data_to_device="true" \
      --variable_update="parameter_server" \
      2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|384|35.880|43022.861||
  |2|384|37.633|82039.298|1.91|
  |3|384|38.959|118870.799|2.76|
  |4|384|39.998|154376.996|3.59|
  |5|384|40.937|188543.296|4.38|
  |6|384|41.674|222249.778|5.17|
  |7|384|42.485|254344.076|5.91|
  |8|384|44.706|276234.665|6.42|

## Test All-reduce

* TF's current implementation only support dynamic rnn + all-reduce

* The command to run the experiment:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=384 \
    --variable_update="replicated" \
    --prefetch_data_to_device="true" \
    --gradient_repacking=4 \
    --all_reduce_spec="nccl" \
    --agg_small_grads_max_bytes=0 \
    --agg_small_grads_max_group=10 \
    2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|384|35.938|42953.483||
  |2|384|38.141|80944.954|1.88|
  |3|384|40.404|114619.504|2.67|
  |4|384|42.471|145385.072|3.38|
  |5|384|45.899|168159.556|3.91|
  |6|384|50.470|183514.791|4.27|
  |7|384|53.284|202794.006|4.72|
  |8|384|58.793|210048.363|4.89|


## dynamic control flow vs. `cudnn_lstm`

* number of operators in C++ end graph

  |_**GPU card number**_|_**cudnn lstm**_|_**dynamic rnn**_|_**ratio**_ (_dynamic_lstm / cudnn_lstm_)|
  |:--:|:--:|:--:|:--:|
  |_**1**_|212|1522|7.1|

* number of kernel invocation during runtime

  |_**GPU card number**_|_**cudnn_lstm**_ <br>_[CPU kernel / GPU kernel]_|_**dynamic_rnn**_ <br>_[CPU kernel / GPU kernel]_|_**ratio**_<br> _dynamic_rnn / cudnn_lstm_|
  |:--:|:--:|:--:|:--:|
  |1|1,281<br>[187 / 1094]|37,644 <br> [28,151 / 9,493]|29.39|
  |2|2,527<br>[351 / 2176]|75,395<br>[56,615 / 18,780]|29.84|
  |3|3,792<br>[529 / 3263]|112,856<br>[85,015 / 27,841]|29.76|
  |4|4,995<br>[692 / 4303]|150,511<br>[113,649 / 36,862]|30.13|
  |5|6,235<br>[858 / 5377]|187,871<br>[45,712 / 142,159]|30.13|
  |6|7,492<br>[1032 / 6460]|224,994<br>[54,277 / 170,717]|30.03|
  |7|8,658<br>[1185 / 7473]||

Models that have no other choices but have to use dynamic control flow oeprators:
  * Attention models
  * Recursive NN (and other tree-based models)
  * Beam search
    * including _**beam decoding**_ and _**beam training**_.
* _**cudnn lstm cannot be used for sequence generation even for models without attention**_.

## Summarization

  1. Implementation optimization.
      * partition parameters among parameter servers.
      * reduce communication when merging gradients of sparse tensors.
  1. Balance the computation workload of each GPU device by padding all training sequence pairs to the same lengths. _**This harms the learning performance**_.
  1. Optimize implementation of I/O pipeline, and prefetch data to devices.
  1. Use enough threads for OP scheduling.
      * _**When compiled with MKL support and all the model is executed on GPU devices, do not use the default setting of `inter_op_parallelism_threads`. Explicitly set `inter_op_parallelism_threads` to a number larger than the number of all devices used**_.
  1. The current implementation of NCCL all-reduce is not well optimized.
      * Too much memory access overhead.
      * Due to TF's current implementation, all-reduce cannot apply to `cudnn_lstm`.
