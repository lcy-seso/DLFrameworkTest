[TOC]

# Test data parallelism on NMT model

## Environment

- **Test machine**
  - TensorFlow r1.8 compiled by gcc 4.9 with Python 3.6, Cuda 9.0, cudnn 7.1, NCCL 2.1
    - Cuda driver version: 390.30
  - OS: Ubuntu 16.04.2
  - Tesla P100-PCIE-16GB
    - 8 GPU cards on one machine.

*All the data are collected in the above machine.*

## Current Problems

1. In the best case, the speedup ratio cannot increase when more than 3 GPU cards are used.
1. The time of forward computation increase significantly which is the most important factor that makes the training process is hard to scale to more than 3 cards. This indicates the copy overhead seems to be too high as more and more GPU cards are used.
1. When using 1 to 3 cards, the speedup ratio is worse in P100 than that in Titian. [**see [this](https://github.com/lcy-seso/dl_framework/tree/master/tensorflow/test_parallel_tensorflow)**]

## Current Best Practice

1. The implementation of data splitting for each GPU card greatly influence the processing speed. . Try best to balance the computation workload for each GPU card, this will reduce the idle time of GPU cards when using synchronization SGD, leading to a higher processing speed.
    - Make sequences in mini-batches for each GPU card having similar lengths.
    - When using TensorFlow dataset API, worker on each GPU card calls `get_next_iterator` to retrieve a mini-batch from the `Dataset`. The `Dataset` are alreadly organized into several buckets. Call `get_next_iterator` several times may get mini-batch from different buckets, so each GPU card may have very different computation workloads.

    > **A potential problem is, if we break the randomness of the training samples, the learning performance will always decrease for SGD updates. This still needs to be further evaluated.**

1. Use Parameter Sever mode. Save the master copy of learnable weights all on GPUs. Each GPU only has a part of learnable weights of the model.
1. A large batch size is required, otherwise the speedup ratio may not be satisfying.

## Test Settings

- Disable training data shuffle.
- Run 50 mini-batches and calculate words processed per second.
- `batch size` = 128
- Test codes are base on this commit: [159b5e5](https://github.com/lcy-seso/dl_framework/tree/159b5e53875ccbe5823ca61b40d3a0a8202a9ddb/tensorflow/test_parallel_tensorflow)

- Information about the model:
    - There are 19 learnable parameters in the model.
      - Three largest parameters are: source embedding, target embedding, and pre-softmax projection. Their size are approximately: 50000 * 512 * 4 / 1024 / 1024 $\approx$ 97.6 MB.
      - There are 8 LSTM unit in the model, Size of the main parameters of the LSTM unit is approximately: 512 * 512 * 4 * 4 / 1024 /1024 $\approx$ 4MB
      - The other parameters are bias. Their size are very small.

## Test Results

### Exp1

- **Settings**

    |<br>|device|
    |:--|:--|
    |trainable variables|`CPU`|
    |source and target embedding|`CPU`|
    |computation worker|`GPU`|
    |optimizer and weight updates|`GPU`|

- **Results**

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|28.373|10816.393|
    |2|128|55.923|11182.421|1.03|
    |3|128|52.519|17891.269|1.65|
    |4|128|65.448|19103.799|1.76|
    |5|128|79.668|19725.808|1.82|
    |6|128|93.542|20209.326|1.86|
    |7|128|105.591|20885.196|1.93|
    |8|128|117.738|21352.595|1.97|

### Exp2

- **Settings**

    |<br>|device|
    |:--|:--|
    |trainable variables|`CPU`|
    |source and target embedding|`CPU`|
    |computation worker|`GPU`|
    |optimizer and weight updates|`CPU`|

- **Results**

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|27.466|11173.585|
    |2|128|55.894|11188.093|1.00|
    |3|128|51.901|18104.287|1.62|
    |4|128|66.127|18907.713|1.69|
    |5|128|79.933|19660.441|1.75|
    |6|128|90.772|20826.179|1.86|
    |7|128|107.045|20601.448|1.84|
    |8|128|114.709|21916.338|1.96|

### Exp3

- **Settings**

    |<br>|device|
    |:--|:--|
    |trainable variables|`GPU`|
    |source and target embedding|`CPU`|
    |computation worker|`GPU`|
    |optimizer and weight updates|`GPU`|

- **Results**

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|24.796|12377.012|
    |2|128|41.467|15080.536|1.22|
    |3|128|48.220|19486.340|1.57|
    |4|128|60.752|20580.360|1.66|
    |5|128|73.943|21253.191|1.71|
    |6|128|87.715|21552.069|1.74|
    |7|128|100.891|21858.216|1.77|
    |8|128|110.517|22747.611|1.84|

### Exp4

- **Settings**

    |<br>|device|
    |:--|:--|
    |trainable variables|`GPU`|
    |source and target embedding|`GPU`|
    |computation worker|`GPU`|
    |optimizer and weight updates|`GPU`|

- **Results**

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|20.918|14671.235|
    |2|128|31.647|19760.121|1.34|
    |3|128|43.218|21741.600|1.48|
    |4|128|56.280|22215.981|1.51|
    |5|128|66.731|23550.140|1.61|
    |6|128|78.113|24201.220|1.64|
    |7|128|86.868|25386.618|1.73|
    |8|128|96.958|25928.852|1.76|

### Exp5

- **Settings**

  - all the settings are the same as Exp4, except that all the training data are fixed to have the same length (**both the source sequence and target sequence have the same lengths: 100 words**).
  - This is to force all the `data iterator` to read data from the same bucket.

- **Results**

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|128|44.519|28895.548|
    |2|128|58.158|44238.325|1.53|
    |3|128|77.895|49543.760|1.71|
    |4|128|92.629|55550.522|1.92|

    |GPU number|batch size per GPU|total time (s)|processing speed (words/second)|speed-up ratio|
    |:--|:--|:--|:--|:--|
    |1|360|92.718|39021.362|
    |2|360|103.826|69693.418|1.78|
    |3|360|122.760|88416.662|2.26|
    |4|360|144.734|99990.391|2.56|

### Conclusions

1. Putting parameter server on CPU does not help. It harms the time performance.
2. Even though the trainable parameters: source embedding, target embedding and the pre-softmax projection are large (512 * 50000 * 4 / 1024 / 1024 $\approx$ 97.6 MB), putting them on GPU devices and copy them among GPUs are stilling faster than putting them on CPU devices.

> From the above two facts, it seems that the cost of data transfer among devices are still cheaper than the cost of putting (some) data on host device which causes data transfer between host and devices.
> But in all the situation, the speed-up ratio for RNN model (or dynamic rnn) are terrible.

---

## Some More Detail Profiling Results

### Evaluate forward and backward computation

Let's (approximately) evaluate the time forward and backward computation used from 1 GPU card to 8 GPU cards.

- How to decide the time forward computation, backward computation and gradients merge process consumes?
  - the `profiler` records the start timestamp when one operator is called.
  - find the start timestamp and duration of the following 6 operators and regard them as special events:

  |event|operator|
  |--|--|
  |start of forward computation|`encoder/embedding_lookup`
  |end of forward computation (need to plus the operator's duration)|`SparseSoftmaxCrossEntropyWithLogits`
  |start of backward computation|`SparseSoftmaxCrossEntropyWithLogits_grad/mul`|
  |end of backward computation (need to plus the operator's duration)|`gradients/AddN_24`|
  |start of gradient merge process|`merge_gradients_18/AddN`|
  |end of gradient merge process (need to plus the operator's duration)|`merge_gradients/Mul`|

- How to decide how much time the learnable weight update process consumes?
  - Once gradients are merged in one GPU card, the learnable weight update process can be executed in parallel (there is no computation dependencies). So "Adam" related operators are not executed in a fixed order.
  - I sorted all the "Adam" related operators by their start timestamps. Then, take the operator with the smallest start timestamp as the start of weight update process. Take the operator with the largest start timestamp as the end of weight update process.

- How to decide the overal forward and backward computation time?
  - Take all the operators (in all GPU cards) in forward and backwark computation into consideration,  take the forward operator that has the smallest start timestamp as start of forward computation, and take the backward operator that has the largest start timestamp as end of backward computation. The difference between these two operators are the overal computation time.

- **Note: the three stages: forward and backward computation, gradients merge, and weight updates have some extent parallelism, the total time of one mini-batch computation is not equal to the sum of these three stages, but less than the sum. In the below analysis, however, I still sum the time of these three stages just to illustrate which stage will be the bottleneck for the speedup ratio.**

---

- 1 GPU card

    ||`gpu 0`|
    |:--|:--|
    |forward (ms)|747.960|
    |backward (ms)|1498.157|

    |computation (ms)|update (ms)|total(ms)|
    |:--|:--|:--|
    |2247.415(0.5989)|1504.964(0.4011)|3752.379(1.0)|

- 2 GPU cards

    ||`gpu 0`|`gpu1`|Average|
    |:--|:--|:--|:--|
    |forward (ms)|1126.084|1134.687|1130.385|
    |backward (ms)|1594.287|1597.329|1595.808|

    |computation|merge gradients|update|total|
    |:--|:--|:--|:--|
    |2741.233(0.5832)|977.091(0.2079)|981.861(0.2089)|4700.185(1.0)|

- 3 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|Average|
    |:--|:--|:--|:--|:--|
    |forward (ms)|1058.086|795.585|1069.084|974.252|
    |backward (ms)|1774.869|1926.459|1569.226|1756.851|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |3628.875(0.5846)|1287.437(0.2074)|1291.460(0.2080)|6207.772(1.0)|

- 4 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|`gpu3`|Average|
    |:--|:--|:--|:--|:--|:--|
    |forward (ms)|847.653|1370.498|2108.995|1048.929|1344.019|
    |backward (ms)|2881.264|2547.138|1597.858|1564.338|2147.649|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |5113.559(0.6133)|1609.991(0.1931)|1614.299(0.1936)|8337.849(1.0)|

- 5 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|`gpu3`|`gpu4`|Average|
    |:--|:--|:--|:--|:--|:--|:--|
    |forward (ms)|1686.037|1050.090|1952.967|1105.849|1340.153|1427.019|
    |backward (ms)|1654.657|2395.931|1434.436|2097.151|2387.262|1993.887|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |5156.716(0.6400)|1447.891(0.1797)|1452.248(0.1802)|8056.855(1.0)|

- 6 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|`gpu3`|`gpu4`|`gpu5`|Average|
    |:--|:--|:--|:--|:--|:--|:--|:--|
    |forward (ms)|1565.595|1849.580|961.208|961.924|1145.740|1081.643|1260.948|
    |backward (ms)|1453.504|1932.095|2291.623|2783.860|2046.814|2482.983|2165.147|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |6568.995(0.7976)|829.013(0.1007)|838.283(0.1018)|8236.291(1.0)|

- 7 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|`gpu3`|`gpu4`|`gpu5`|`gpu6`|Average|
    |:--|:--|:--|:--|:--|:--|:--|:--|:--|
    |forward (ms)|1522.207|1279.141|2178.819|1142.911|1233.777|2246.082|918.746|1503.098|
    |backward (ms)|1518.541|2773.027|2077.349|3301.667|2092.899|2251.638|3006.145|2431.609|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |8094.371(0.8182)|895.308(0.0905)|903.447(0.0913)|9893.126(1.0)|

- 8 GPU cards

    ||`gpu 0`|`gpu 1`|`gpu 2`|`gpu3`|`gpu4`|`gpu5`|`gpu6`|`gpu7`|Average|
    |--|--|--|--|--|--|--|--|--|--|
    |forward (ms)|1035.903|1951.314|1414.271|1361.873|1691.980|1738.065|3083.426|1445.978|1715.351|
    |backward (ms)|1385.246|1854.531|2973.311|2905.979|3199.676|2581.493|1695.937|1401.610|2249.723|

    |computation (ms)|merge gradients (ms) |update (ms)|total(ms)|
    |:--|:--|:--|:--|
    |8834.801(0.8521)|761.964(0.0735)|771.562(0.0744)|10368.327(1.0)|

### Findings

1. The time forward computation consumes noticeably increase as more and more GPU cards are used.
2. Theoretically, the time backward computation will not increase as more and more GPU cards are used. However, from the profiling results, the backward time also increases slightly when more GPU cards are used.
3. Another problem is, in the above test, all the GPU cards are fixed to have exactly the same amount of computation, but there is always one GPU card whose forward/backward computation is much slower than other GPU cards.
