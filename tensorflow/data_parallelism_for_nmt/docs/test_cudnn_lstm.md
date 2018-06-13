# Test Allreduce for NMT model

- Data collected based on this commit [dd8cd72](https://github.com/lcy-seso/dl_framework/tree/dd8cd72468774604d39daefc7570e6e786eab7a5/tensorflow/data_parallelism_for_nmt)

## Test on Tesla P100

- Commandline to run the test:

  ```bash
  CUDA_VISIBLE_DEVICES="7" python train.py \
  --encoder_type="cudnn_lstm" \
  --direction="bi" \
  --batch_size=360 \
  --variable_update="parameter_server" \
    2>&1 | tee train.log
  ```

  - **NOTE: the current implementation cannot use NCCL allreduce to merge gradients**.

  - Use [cudnn_lstm](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM) for both encoder and decoder.

    |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
    |:--|:--|:--|:--|:--|
    |1|360|29.217|49532.893||
    |2|360|33.618|86096.435|1.74|
    |3|360|48.257|89968.832|1.82|
    |4|360|53.618|107964.402|2.18|
    |5|360|64.983|111351.868|2.25|
    |6|360|71.581|121305.375|2.45|
    |7|360|86.774|116744.428|2.36|
    |8|360|97.896|118263.894|2.39|

  - I directly copy the test result when encoder and decoder are both [dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)

    |GPU number|batch size per GPU|total time to run 20 batches(s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|360|40.574|35668.346||
    |2|360|47.222|61293.843|1.72|
    |3|360|58.628|74052.914|2.08|
    |4|360|57.281|75795.220|2.12|
    |5|360|66.166|87488.569|2.45|
    |6|360|83.896|103499.177|2.90|
    |7|360|100.488|100812.133|2.83|
    |8|360|108.465|106740.810|2.99|
