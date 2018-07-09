[TOC]

# Summarization

Usually the following factors are ensential to time performance:

1. Computation
2. Memory Access
3. *Communication*
4. *I/O pipline*
5. *synchronization SGD is used, balance of computation workload* (may harm learning performance)
6. _**OP scheduling**_ to hide latency

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

## PS mode vs. All-reduce and `dynamic_rnn` vs. `cudnn_lstm`

### Experiment results

* **defalut setting of `inter_op_parallelism_threads`**
* all learnable parameters are on GPU

    |GPU cards|dynamic rnn + ps mode|dynamic rnn + all reduce|cudnn lstm + ps mode|
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

1. In the test above, all the settings cannot scale well. They are all mainly bound by the host computation threads.
2. All-reduce according to current implementation may have the following limitations:
    * All-reduce has an explicit synchronization point: gradients of all the learnable parameters are concatenated all together.
    * All-reduce has extra memory access overload, it involves a pack and unpack process.
      * before all-reduce gradients of all the learnable parameters are concatenated together and then split into several parts.
      * After all-reduce, a reverse post-process is required to produce merged gradients.
3. If `cudnn_lstm` is used:
    * multiple stacked LSTMs only has one parameter (approximately 16 MB).

## The effect of host thread pool

### The ideal situation: no I/O and communication cost

|GPU cards|dynamic_rnn|cudnn_lstm|
|--|--|--|
|1|39368.756|51046.074|
|2|71252.362(1.81)|93790.686(1.84)|
|3|100546.336(2.55)|133520.790(2.62)|
|4|127116.260(3.23)|175351.156(3.44)|
|5|149338.599(3.79)|216412.574(4.24)|
|6|168780.420(4.29)|238642.350(4.68)|
|7|184857.154(4.70)|279107.499(5.47)|
|8|206795.206(5.25)|309930.493(6.07)|

* In the ideal situation, there is no I/O and communication cost. However, `dynamic rnn` scales worse than `cudnn lstm`.

### `cudnn_lstm` + PS mode

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
    |8|118263.894(2.39)|264045.800(5.18)|274915.239(5.50)|

* If the I/O pipline works correctly, compared to the ideal situation experiment, this test only adds the communication cost.
* The currently speedup ratio is near the speedup ratio obtained in the *ideal situation experiment*.

### `dynamic_rnn` + PS mode

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
    |8|106740.810 (2.99)|156880.228(4.18)|188651.349(4.96)|

### `dynamic_rnn` + all reduce

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
    |8|94099.686(2.61)|131943.589(3.51)|151700.671(4.05)|
## Problems

1. Even in the idea situation, the speedup ratio is below linear. dynamic rnn scale worse than cudnn_lstm.
    * current suspection is the speedup ratio is bound by host op scheduling.
1. All-reduce scale worse than PS mode under current implementation.
