[TOC]

# Test totally independent model replica on synthetic data

Data collected based on this commit [fb9dc73](https://github.com/lcy-seso/dl_framework/tree/fb9dc733a774610a67034eb91038ba832e4bf898/tensorflow/data_parallelism_for_nmt)

Commandline to run the test:

1. `dynamic rnn` encoder and decoder:

    ```bash
    CUDA_VISIBLE_DEVICES="x,...,x" python train.py \
      --batch_size=360 \
      --variable_update="replicated" \
      --independent_replica="True" \
      --use_synthetic_data="True" \
      --src_max_len=100 \
      2>&1 | tee train.log
    ```

1. `cudnn_lstm` encoder and decoder:

    ```bash
    CUDA_VISIBLE_DEVICES="x,...,x" python train.py \
      --encoder_type="cudnn_lstm" \
      --direction="bi" \
      --batch_size=360 \
      --variable_update="replicated" \
      --independent_replica="True" \
      --use_synthetic_data="True" \
      --src_max_len=100 \
      2>&1 | tee train.log
    ```

  - **NOTE: the current implementation cannot use NCCL allreduce to merge gradients**.

## [cudnn_lstm](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM) encoder and decoder

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|360|28.244|50984.016||
  |2|360|30.283|95103.727|1.87|
  |3|360|34.275|126041.164|2.47|
  |4|360|40.678|141601.424|2.78|
  |5|360|45.267|159056.709|3.12|
  |6|360|49.542|174397.548|3.42|
  |7|360|54.835|183824.444|3.61|
  |8|360|58.998|195259.401|3.83|

- I paste the results for experiment that uses PS mode to merge gradients and use `cudnn_lstm` encoder and decoder here for comparison.

  |GPU number|batch size per GPU|total time to run 20 batches(s)|processing speed (words/second)|speed-up ratio|
  |:--|:--|:--|:--|:--|
  |1|360|28.176|51106.551||
  |2|360|31.722|90787.648|1.78|
  |3|360|37.075|116521.261|2.28|
  |4|360|42.303|136161.554|2.66|
  |5|360|47.482|151635.090|2.97|
  |6|360|53.096|162722.755|3.18|
  |7|360|60.311|167133.973|3.27|
  |8|360|69.595|165528.995|3.24|

## [dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) encoder and decoder

  |GPU number|batch size per GPU|total time to run 20 batches(s)|processing speed (words/second)|speed-up ratio|
  |:--|:--|:--|:--|:--|
  |1|360|36.385|39576.279||
  |2|360|41.428|69518.163|1.76|
  |3|360|46.470|92963.990|2.35|
  |4|360|53.906|106852.905|2.70|
  |5|360|60.421|119164.782|3.01|
  |6|360|69.678|123999.777|3.13|
  |7|360|78.928|127711.625|3.23|
  |8|360|82.623|139428.114|3.52|

- I paste the results for experiment that uses PS mode to merge gradients and use`dynamic rnn` encoder and decoder here for comparison.

  |GPU number|batch size per GPU|total time to run 20 batches(s)|processing speed (words/second)|speed-up ratio|
  |:--|:--|:--|:--|:--|
  |1|360|36.497|39455.242||
  |2|360|41.340|69666.464|1.77|
  |3|360|47.367|91203.687|2.31|
  |4|360|54.651|105395.328|2.67|
  |5|360|61.107|117825.190|2.99|
  |6|360|68.696|125771.308|3.19|
  |7|360|75.263|133931.073|3.39|
  |8|360|83.640|137732.931|3.49|
