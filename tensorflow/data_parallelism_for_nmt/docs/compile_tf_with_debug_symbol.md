## Compile TensorFlow with debug symbol

```bash
bazel build \
  --config=debug \
  --config=cuda \
  --verbose_failures \
  --config=mkl \
  -c dbg --strip=never \
 //tensorflow/tools/pip_package:build_pip_package
```

run the model with the command line:

```bash
export CUDA_VISIBLE_DEVICES="x,...,x"
gdb -ex r --args python train.py \
    --batch_size=384 \
    --variable_update="replicated" \
    --independent_replica="true" \
    --use_synthetic_data="true" \
    --src_max_len=100 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
```

* In GDB use `dir` to set the directory of the source codes.

## Experiment Results

cudnn_lstm

```bash
CUDA_VISIBLE_DEVICES="x,...,x" python3 train.py \
    --batch_size=384 \
    --variable_update="replicated" \
    --independent_replica="true" \
    --use_synthetic_data="true" \
    --src_max_len=100 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
    2>&1 | tee train.log
```


|GPU number|Batch size|Total time to process 20 batches (s)|Processing speed (words/s)|Speedup ratio|
|:--|:--|:--|:--|:--|
|1|384|29.680|51751.293||
|8|384|39.299|312680.888|6.04|
