[TOC]

# Summarization

Usually the following four factors are ensential to time performance:

* Computation
* Memory Access
* Communication
* _**OP scheduling**_ to hide latency
* I/O pipline

[Arithmetic_intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity)

---

## Experimental Settings

* The model topology
  * RNN encoder-decoder
  * All training samples has the same lengths (source sentence and the target sentence all have 100 words).
  * Hyper-parameters
    * vocabulary size : 5w
    * word embedding dimension : 512
    * LSTM hidden dimension : 512
    * bi-direction LSTM encoder : 2 stacked layers, 4 LSTM units in total
    * uni-direction LSTM decoder : 4 stacked layers, 4 LSTM units in total
    * batch size = 360

* Communication caused by transferring gradients of learnable parameters to another device

  ||parameter size (MB)|
  |--|--|
  |embedding| (512 * 100 * 360) * 4 / 1024 / 1024 $\approx$ 70.31 MB|
  |LSTM| (512 * 512 * 4 + 512 * 4) * 4 / 1024 / 1024 $\approx$ 4MB|
  |pre-softmax projection| (512 * 50000) * 4 / 1024 / 1024 $\approx$ 97.66 MB|

  * 2 embedding + 8 LSTM + 1 pre-softmax projection $\approx$ 270.28 MB

* If `cudnn_lstm` is used, the stacked multi-layer LSTMs only has one parameter.

## PS mode vs. All-reduce and dynamic_rnn vs. cudnn_lstm

### Experiment results

* **defalut setting of `inter_op_parallelism_threads`**
  * Exp1:
    * dynamic rnn
    * PS mode
    * all learnable parameters are on GPU
  * Exp2:
    * dynamic rnn
    * all reduce
    * all learnable parameters are on GPU
  * Exp3:
    * cudnn lstm
    * PS mode
    * all learnable parameters are on GPU

    |GPU cards|[Exp1](https://github.com/lcy-seso/dl_framework/blob/master/tensorflow/data_parallelism_for_nmt/docs/test_ps_mode.md#exp5)<br> word/s (speedup-ratio)|[Exp2](https://github.com/lcy-seso/dl_framework/blob/master/tensorflow/data_parallelism_for_nmt/docs/test_allreduce_algorithm.md#test-on-tesla-p100)<br> word/s (speedup-ratio)|[Exp3](https://github.com/lcy-seso/dl_framework/blob/master/tensorflow/data_parallelism_for_nmt/docs/test_cudnn_lstm.md#use-wmt-14-en-de)<br> word/s (speedup-ratio)|
    |--|--|--|--|
    |1|35668.346|35990.222|49532.893|
    |2|61293.843 (1.72)|60470.170(1.68)|86096.435 (1.74)|
    |3|74052.914 (2.08)|71928.146(2.00)|89968.832 (1.82)|
    |4|75795.220 (2.12)|81684.778(2.27)|107964.402(2.18)|
    |5|87488.569 (2.45)|86029.952(2.39)|111351.868(2.25)|
    |6|103499.177 (2.90)|85531.174(2.38)|121305.375(2.45)|
    |7|100812.133 (2.83)|88995.177(2.47)|116744.428(2.36)|
    |8|106740.810 (2.99)|94099.686(2.61)|118263.894(2.39)|

### Foundings

## the effect of `inter_op_parallelism_threads`

### ideal situation

|GPU cards|dynamic_rnn|cudnn_lstm|
|--|--|--|
|1|39368.756|51046.074|
|2|71252.362(1.81)|93790.686(1.84)|
|3|100546.336(2.55)|133520.790(2.62)|
|4|127116.260(3.23)|175351.156(3.44)|
|5|149338.599(3.79)|216412.574(4.24)|
|6|168780.420(4.29)|238642.350(4.68)|
|7|184857.154(4.70)|279107.499(5.47)|
|8||309930.493(6.07)|

### cudnn_lstm + PS mode

* all learnable parameters are on GPU

    |GPU cards|2 cpu threads|56 cpu threads|56 cpu threads + prefetch to device|
    |--|--|--|--|
    |1|49532.893|50925.421|50007.279|
    |2|86096.435 (1.74)|87286.751(1.71)|92236.723(1.84)|
    |3|89968.832 (1.82)|126272.785(2.48)|127377.172(2.55)|
    |4|107964.402(2.18)|165161.244(3.24)|156508.555(3.13)|
    |5|111351.868(2.25)|192827.671(3.79)|190614.737(3.81)|
    |6|121305.375(2.45)|210497.859(4.13)|220829.452(4.42)|
    |7|116744.428(2.36)|227553.456(4.47)|234614.995(4.69)|
    |8|||274915.239(5.50)|

### dynamic_rnn + PS mode

* all learnable parameters are on GPU

    |GPU cards|2 cpu threads|56 CPU threads|56 threads + prefetch to device|
    |--|--|--|--|
    |1|35668.346|37548.935|38048.046|
    |2|61293.843 (1.72)|66017.246(1.76)|69053.092(1.81)|
    |3|74052.914 (2.08)|90105.618(2.40)|95543.591(2.51)|
    |4|75795.220 (2.12)|110593.685(2.95)|120588.115(3.17)|
    |5|87488.569 (2.45)|125059.488(3.33)|140838.851(3.70)|
    |6|103499.177 (2.90)|135201.853(3.60)|158656.805(4.17)|
    |7|100812.133 (2.83)|146193.808(3.89)|176083.321(4.63)|
    |8|106740.810 (2.99)||188651.349(4.96)|

### dynamic_rnn + all reduce

* all learnable parameters are on GPU

    |GPU cards|2 cpu threads|56 CPU threads|56 threads + prefetch to device|
    |--|--|--|--|
    |1|35990.222|37272.324|37460.412|
    |2|60470.170(1.68)|64676.016(1.74)|67175.406(1.79)|
    |3|71928.146(2.00)|87294.230(2.34)|91681.983(2.45)|
    |4|81684.778(2.27)|105983.110(2.84)|112202.447(3.00)|
    |5|86029.952(2.39)|117674.826(3.16)|129761.878(3.46)|
    |6|85531.174(2.38)|120827.291(3.24)|136668.138(3.65)|
    |7|88995.177(2.47)|124162.210(3.33)|143680.666(3.84)|
    |8|94099.686(2.61)||151700.671(4.05)|
