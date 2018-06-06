# Test Allreduce for NMT model

- based on this commit [7803a08](https://github.com/lcy-seso/dl_framework/tree/7803a08db6389c9c530da0b3814b2973531a034a/tensorflow/data_parallelism_for_nmt)
- batch size per GPU card is fixed to be 360.

  |gpu number|total time to process 20 batches(s)|processing speed (words/s)|speedup-ratio|
  |:--|:--|:--|:--|
  |1|40.211|35990.222|
  |2|47.865|60470.170|1.68|
  |3|64.863|66935.080|1.86|
  |4|71.245|81252.331|2.26|
  |5|84.110|86029.952|2.39|
