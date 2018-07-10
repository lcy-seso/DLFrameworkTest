[TOC]

# Optimized I/O pipline

Data collected based on this commit [636a8d8](https://github.com/lcy-seso/dl_framework/tree/636a8d81a052f4b07b54b54818174e7975e696a3/tensorflow/data_parallelism_for_nmt)

## Parameter Server Mode

This test uses WMT 14 En-De dataset.

## cudnn_lstm

Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=360 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
    --prefetch_data_to_device="true" \
    --variable_update="parameter_server" \
    2>&1 | tee train.log
```

|GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
|:--|:--|:--|:--|:--|
|1|360|28.940|50007.279||
|2|360|31.380|92236.723|1.84|
|3|360|34.085|127377.172|2.55|
|4|360|36.987|156508.555|3.13|
|5|360|37.961|190614.737|3.81|
|6|360|39.321|220829.452|4.42|
|7|360|43.179|234614.995|4.69|
|8|360|42.113|274915.239|5.50|

## dynamic_rnn

Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=360 \
    --prefetch_data_to_device="true" \
    --variable_update="parameter_server" \
    2>&1 | tee train.log
```

|GPU number|batch size per GPU|total time to run 20 batches(s)|processing speed (words/second)|speed-up ratio|
|:--|:--|:--|:--|:--|
|1|360|38.036|38048.046||
|2|360|41.916|69053.092|1.81|
|3|360|45.441|95543.591|2.51|
|4|360|48.005|120588.115|3.17|
|5|360|51.378|140838.851|3.70|
|6|360|54.729|158656.805|4.17|
|7|360|57.532|176083.321|4.63|
|8|360|61.370|188651.349|4.96|

## NCCL all reduce (dynamic rnn)

This test uses WMT 14 En-De dataset.
Commandline to run the test:

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
  --batch_size=360 \
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
|1|360|38.633|37460.412||
|2|360|43.087|67175.406|1.79|
|3|360|47.355|91681.983|2.45|
|4|360|51.592|112202.447|3.00|
|5|360|55.764|129761.878|3.46|
|6|360|63.535|136668.138|3.65|
|7|360|70.506|143680.666|3.84|
|8|360|76.319|151700.671|4.05|

---

## Independent Model Replica

This test uses synthetic data.

- cudnn_lstm

  Commandline to run the test:

  ```bash
  CUDA_VISIBLE_DEVICES="0" python3 train.py \
      --batch_size=360 \
      --variable_update="replicated" \
      --independent_replica="true" \
      --src_max_len=100 \
      --use_synthetic_data="true" \
      --direction="bi" \
      --encoder_type="cudnn_lstm" \
      2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|360|28.210|51046.074||
  |2|360|30.707|93790.686|1.84|
  |3|360|32.355|133520.790|2.62|
  |4|360|32.848|175351.156|3.44|
  |5|360|33.270|216412.574|4.24|
  |6|360|36.205|238642.350|4.68|
  |7|360|36.115|279107.499|5.47|
  |8|360|37.170|309930.493|6.07|

- dynamic_rnn

  Commandline to run the test:

  ```bash
  CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=360 \
    --variable_update="replicated" \
    --independent_replica="true" \
    --use_synthetic_data="true" \
    --src_max_len=100 \
    2>&1 | tee train.log
  ```

  |GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
  |:--|:--|:--|:--|:--|
  |1|360|36.577|39368.756||
  |2|360|40.420|71252.362|1.81|
  |3|360|42.965|100546.336|2.55|
  |4|360|45.313|127116.260|3.23|
  |5|360|48.213|149338.599|3.79|
  |6|360|51.191|168780.420|4.29|
  |7|360|54.529|184857.154|4.70|
  |8|360|55.707|206795.206|5.25|
