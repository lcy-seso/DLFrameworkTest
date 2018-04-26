## Test TensorFlow data parallelism on RNNLM

- GTX Titan, 3 cards on one machine.
- Settings (how the test runs):
  1. Run 5 epoches (0, 1, 2, 3, 4), and count the total time that epoch 1, 2 and 3 use.
  1. In the data feeding part:
      - read all the training data into CPU memory, so that the time I/O consumes will not count.
      - Fix the data bucket number to 1.
   1. **Dataset shuffle is used this test.**

### Test 1: fix the total batch size

>**This does not frequently happend when using multiple cards.**

- Fix the total batch size, so that the rounds of forward-backward computation and parameter updates in on training epoch are the same.
- But this means, when using multiple GPU cards, the matirx sizes for each card become smaller.

  |GPU number| Total time to run 3 epoches (s) |batch size per card|Total batch size|Speed-up Ratio|
  |--|--|--|--|--|
  |1|43.096173|768| 786 * 1 = 768 ||
  |2|30.948307|384| 384 * 2 = 768 |1.39|
  |3|27.553479|256 | 256 * 3 = 768|1.56|

### Test 2: Fix batch size per card

>**This is often the situation when using the multiple GPU cards that relatively increase the total batch size.**

- This will lead to a very large batch size which potentially (sligthly) harm the learning performance (not always the truth, need carefully tune the hyper parameters.)
- A large batch size means in one epoch forward-backward computation will become less (less parameter updates). The overall computations are slightly reduced.
- A large batch size means when merge gradients, the data communication among GPU cards becomes heavier.


  |GPU number| Total time to run 3 epoches (s) |batch size per card|Total batch size|Speed-up Ratio|
  |--|--|--|--|--|
  |1|45.019466|512|512 * 1 = 768||
  |2|29.178264|512|512 * 2 = 1024|1.54|
  |3|22.716143|512|512 * 3 =  1536|1.98|

### Some points:

1. A proper batch size helps, neither too small nor too large to make best use of GPU's parallel computation power.
2. By using data parallism, it is relatively easy to achieve a more balanced worklaod for each GPU card.
3. Buckets helps to accelerate training speed.
    - But will potentially cause problem for this simple implementation, because `tf.split` are used to split data which requires the batch size can be evenly divided by the number of GPU cards. This cannot be garanteed if construct the buckets by using TensorFlow's dataset API dynamically.
