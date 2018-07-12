[TOC]

# How operators are scheduled in TensorFlow ?

## [Rendezvous](https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/core/framework/rendezvous.h#L30) in TensorFlow

stackoverflow answers:

[TensorFlow Execution on a single (multi-core) CPU Device](https://stackoverflow.com/questions/47416445/tensorflow-execution-on-a-single-multi-core-cpu-device)

[Tensorflow Cross Device Communication](https://stackoverflow.com/questions/40710224/tensorflow-cross-device-communication/40711279#40711279)

* TensorFlow represents communication in the dataflow graph using `Send` and `Recv` ops that are added to the graph automatically _**when the graph is partitioned across devices**_.
* For each edge that has a source and destination on different devices, _**the graph partitioner**_ inserts a pair of `Send` and `Recv` ops that share the same "`rendezvous key`" (an automatically generated string name that is used as a key in the rendezvous' index of pending tensors to be communicated).
  * The implementation of the `Send` op is simple:
    * it calls `Rendezvous::Send()`, passing in its `rendezvous` key and single input tensor
    * then returns immediately without blocking.
  * The implementation of the `Recv` op is slightly more complicated:
    * it registers a callback to be called when the tensor with the given key becomes available.
    * That callback is responsible for "producing" the output of the `Recv` op, and unblocking subsequent computation.
* [IntraProcessRendezvous](https://github.com/tensorflow/tensorflow/blob/41285cf7a11fa3a2c2ead6b6e9adcec4232b18ad/tensorflow/core/common_runtime/rendezvous_mgr.h#L32) handles the transfer of data between devices in the same process. In the (unlikely) event that the transfer is between two CPU devices in the same process, the transfer can be achieved by a simple Tensor assignment. Otherwise, TensorFlow kicks off a device-specific DMA routine to transfer data between a CPU and GPU device.

## Codes Flow

### create a `session` C API :  [TF_NewSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.cc#L2444)

The execution flow starts from Python end, session --> constructor --> C API :  [TF_NewSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.cc#L2444)

C API `TF_NewSession` uses the [factory method pattern](https://en.wikipedia.org/wiki/Factory_method_pattern).

  * The local training invokes [NewSession]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L149) implemented in  [DirectSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.h#L55).
  * `NewSession` invokes `DirectSession`'s [`Cor`]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L244)
      1. create a global thread pool
          * _It seems that there canbe multiple thread pools. Each device can also has its own thread pool, this canbe configurated. I do not look into more details of this._
      2. add device.

### [DirectSession::Run](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L645)

* Call ExecutorState::RynAsynch, which initializes the TensorFlow ready queue with the roots nodes.
* ExecutorState::Process, which executes the operation.
* [DirectSession::GetOrCreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1255)
  * [DirectSession::CreateGraphs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1362)
  * Create _**several graphs**_ given the existing `GraphDef` and the input feeds and fetches
  * [Partition](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1452)
  * [ConvertGraphDefToGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1481)
  * [graph optimization pass](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1486) : post partition
* [DirectSession::CreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1094)
* [DirectSession::RunInternal](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L433)
  * The main step in `RunInternal` is to call [RunAsync](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L572) for each [executor](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L559).

  <p align="center">
  <img src="images/01_code_flow.png" width=700><br/>
  </p>

### [ExecutorState::Process](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1618)

>_**NOTE THAT: Process is run in the thread from the thread pool.**_

1. [create a `inline_ready` queue and push the `ready` node into the queue](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1655).
    * when calling `Process`, _**only one**_ one ready node is passed to it.
1. while the `inline_ready` queue is not empty, repeat the following steps:
    * For _**asynchronous**_ kernels
      * call [ComputeAsync](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1784) with a callback funciton [done](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1738) which will be called after OpKernel is finished.
      * In the `done` callback
         * [ExecutorState::PropagateOutputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2063) is called, which propagates outputs along out edges, and puts newly ready nodes into the ready queue.
         * Call [NodeDone](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2168) and `NodeDone` _**calls `ScheduleReady` again with `inline_ready` being fixed to `null`**_.
           ```cpp
           NodeDone(s, state->item->node, ready, stats, nullptr);
           ```

    * For _**synchronous**_ kernels
       * call [Compute](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1789)
       * call [ExecutorState::PropagateOutputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2063)
       * Call [NodeDone](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2168)

           ```cpp
           NodeDone(s, item.node, ready, stats, &inline_ready);
           ```

## [ScheduleReady](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2218)

There is a very important function `ScheduleReady`, let's first summarize its behavior

* `ScheduleReady` scheduled operators based on _**two FIFO queue**_
  * `ready` queue : operators will be **dispathed to other theads** in the thread pool.
  * `inline_ready` queue: operators will be **run in the current thread**.
* If the `inline_ready` queue is empty:
  * all the operator kernels in the ready queue will be run in the thread pool.
* If the `inline_ready` queue is not empty:
  * Go through the `ready` queue:
    * _**dispath expensive nodes**_ to thread in the thread pool
    * _**push all inexpensive nodes to `inline_ready`**_
    * If all the nodes in `ready` queue, push the last expensive node to `inline_ready` queue instead of dispathing it.

---

## Summarization

- Partition graphs across devices.
- One executor is created for each partitioned graph.
- All the executors share a global pool thread.
  - When Tensorflow directly uses `DirectSession`, it uses Eigen's ThreadPool.
  - Threads in the thread pool are scheduled by [`Eigen` implementation](https://github.com/ROCmSoftwarePlatform/eigen-upstream/blob/master/unsupported/Eigen/CXX11/src/ThreadPool/ThreadPoolInterface.h#L20). TensorFlow does not schedule the threads itself.
  - The Eigen ThreadPpool that TF used is using this queue implementation: [RunQueue](https://eigen.tuxfamily.org/dox/unsupported/RunQueue_8h_source.html). There is one queue per thread.
- Each executor runs operators in its sub-graphs "one by one":
    * initialize the `ready` queue with root nodes on each device.
    * If there are multiple root nodes in one device, _**each root node will be directly run**_ by a thread (if there are available threads in the thread pool).
      * Run the kernel function.
      * Once a node finishes computing, it will annotate its successor nodes as ready, this means push its successor into the `ready` queue.
      * Then all the working items (the OP kernels to be executed) will be examined one by one:
        * synchronous kernels (most GPU kernels) will be directly dispatched to threads in the thread pool.
          * *If all the kernels in the thread pool are  synchronous, the current thread will execute the last one instead of dispatching it*.
        * Synchronous kernels (all CPU kernels) will be devided into two parts: expensive kernels and inexpensive kernels.
          * the current thread will execute one expensive kernel or all the inexpensive kernels.
