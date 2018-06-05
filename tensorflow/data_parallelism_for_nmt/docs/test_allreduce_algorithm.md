- based on this commit [7803a08](https://github.com/lcy-seso/dl_framework/tree/7803a08db6389c9c530da0b3814b2973531a034a/tensorflow/data_parallelism_for_nmt)
- fix batch size per GPU card to 360

  |gpu number|total time to process 20 batches(s)|processing speed (words/s)|speedup-ratio|
  |:--|:--|:--|:--|
  |1|40.834|35440.978|
  |2|47.870|60464.357|1.70|
  |3|60.548|71705.558|2.02|
  |4|70.313|82328.905|2.32|
