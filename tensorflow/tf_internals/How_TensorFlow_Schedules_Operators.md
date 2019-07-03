# _How TensorFlow schedules operators ?_

---

# Overall Scheduling Process (I)

* A <span style="background-color:#B3D9D9">global</span> thread pool is created at the time the `Session` object is created.
* User's computation graph will be _**partitioned to multiple sub-graphs across the device**_.
* One executor will be created for one sub-graph.
   * *For example, in a training task using 2 GPU cards, the computation graph will be partitioned into three sub-graphs: one for CPU, two for GPU1 and GPU2 respectively.*
* The thread pool is shared among all executors.

---

# Overall Scheduling Process (II)

Each executor run operators in its sub-graph one by one, and the execution flow is bounded by dataflow dependencies among nodes.

For each executor:
1. a `ready` queue is created and is intialized with the root node
1. if there is an idle thread in the thread pool, let's call it `thread A`, _**the root node will be directly scheduled to `thread A` to run**_

---

# Overall Scheduling Process (III)

<span style="background-color:#B3D9D9;">_**NOTE: Now the execution is in `thread A`**_</span>:
  * Once the root node(operator) finishes execution, it will annotate its successor nodes as ready by pushing them into the `ready` queue.
  * Operators in the `ready` queue will be scheduled to run one by one.

---

# Overall Scheduling Process (IV)

<span style="background-color:#B3D9D9;">_**NOTE Again: Now the execution is in `thread A`**_</span>:

For **asynchronous kernels**:

* directly dispatched to other threads in the thread pool.
* *If all the kernels in the `ready` queue are asynchronous, `thread A` will execute the last one instead of dispatching it*.

For **synchronous kernels**:
* synchronous kernels are registered as expensive kernels and inexpensive kernels.
* `thread A` will execute <span style="background-color:#E0FFFF;">*one*</span> expensive kernel <span style="background-color:#DB7093;">_**or**_</span> <span style="background-color:#E0FFFF;">*all*</span> the inexpensive kernels.

---

# Some More Implementation Detials

* In local training (`DirectSession` is called), TensorFlow uses Eigen's ThreadPool.
* Threads in the thread pool are scheduled by [`Eigen` implementation](https://github.com/ROCmSoftwarePlatform/eigen-upstream/blob/master/unsupported/Eigen/CXX11/src/ThreadPool/ThreadPoolInterface.h#L20). TensorFlow does not schedule the threads itself.
* The worker function implemented in TensorFlow <span style="background-color:#B3D9D9;">only make a binary decision</span>: _**run the OP kernel in current thread or dispatch it to other threads**_.
  * Then all the other things are left to Eigen scheduling.
* The Eigen ThreadPpool that TensorFlow used is using this queue implementation: [RunQueue](https://eigen.tuxfamily.org/dox/unsupported/RunQueue_8h_source.html). There is one queue per thread.
  * _Currently I cannot explain more details about Eigen's scheduling algorithm._

---

# _Now, Let's look into implementation details._

---

# Execution Flow (I)

* The execution flow starts from Python end
  * define a `session`, call its `Cor`.
  * `session`'s `Cor` will call TensorFlow's C API :  [TF_NewSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.cc#L2444)

* C API `TF_NewSession` uses the [factory method pattern](https://en.wikipedia.org/wiki/Factory_method_pattern).

  * The local training invokes [NewSession]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L149) implemented in  [DirectSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.h#L55).
  * `NewSession` invokes `DirectSession`'s [`Cor`]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L244)
      1. create a global thread pool
      1. add device.

---

# Execution Flow (II)

* When `seesion.run` in Python end is called, [DirectSession::Run](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L645) is called.
* Call [DirectSession::RunInternal](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L433) to execute a mini-batch computation.

---

# Execution Flow (III) : [RunInternal](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L433)

* [DirectSession::GetOrCreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1255)
  * [DirectSession::CreateGraphs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1362)
  * Create _**several graphs**_ given the existing `GraphDef` and the input feeds and fetches
  * [Partition](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1452) the graph across devices
  * [ConvertGraphDefToGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1481)
  * [graph optimization pass](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1486) : *post partition*
* [DirectSession::CreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1094)
* _**Then the last and the most important step in `RunInternal` is to call [ExecutorState::RunAsync](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L572) for each [executor](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L559)**_.

---

# Execution Flow (IV)

<p align="center">
<img src="images/01_code_flow.png" width=700><br/>
</p>

---

# Operators are scheduled in `RunAsync`

<font size=5>

```cpp
void ExecutorState::RunAsync(Executor::DoneCallback done) {
  // The Callback done here is an execution barrier.

  const Graph* graph = impl_->graph_.get();
  TaggedNodeSeq ready;

  ...

  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }
  if (ready.empty()) {
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = std::move(done);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}
```
</font>

---

# [ScheduleReady](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2218)

There is a <span style="background-color:#B3D9D9;">very important function `ScheduleReady`</span>, let's first summarize its behavior

<font size=5.5>

* `ScheduleReady` scheduled operators based on _**two FIFO queue**_
  * <span style="background-color:#DDA0DD;">_ready queue_ </span>: operators will be **dispathed to other theads** in the thread pool.
  * <span style="background-color:#DDA0DD;">_inline_ready queue_ </span>: operators will be **run in the current thread**.
* If the `inline_ready` queue is empty, all the operator kernels in the `ready` queue will be scheduled in the thread pool.
* If the `inline_ready` queue is not empty, then go through the `ready` queue:
    * _**dispath expensive nodes**_ to threads in the thread pool
    * _**push all inexpensive nodes to `inline_ready`**_
    * If all the nodes in `ready` queue are expensive and `inline_ready` is empty, _**push the last expensive node to `inline_ready` queue instead of dispathing it**_.

</font>

---

<font size=4>

```cpp
void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  TaggedNodeReadyQueue* inline_ready) {
  if (ready.empty()) return;

  ...

  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      runner_([=]() { Process(tagged_node, scheduled_usec); });
    }
    return;
  }

  const GraphView& gview = impl_->gview_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = *gview.node(tagged_node.node->id());
    if (tagged_node.is_dead || !item.kernel_is_expensive) {
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                          scheduled_usec));
      }
      curr_expensive_node = &tagged_node;
    }
  }
  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      inline_ready->push_back(*curr_expensive_node);
    } else {
      runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                        scheduled_usec));
    }
  }
}
```

</font>

---

# [ExecutorState::Process](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1618)

<span style="background-color:#DDA0DD;">_**NOTE: Process is running in the thread from the thread pool.**_</span>

* [create a `inline_ready` queue and push the `ready` node into the queue](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1655).
    * when calling `Process`, <span style="background-color:#B0E0E6;">_**only one**_</span> one ready node is passed to it.
* while the `inline_ready` queue is not empty, repeat the following steps:

---

# For _**asynchronous**_ kernels

* call [ComputeAsync](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1784) with a callback funciton [done](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1738) which will be called after OpKernel is finished.
* In the `done` callback
  * [ExecutorState::PropagateOutputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2063) is called, which propagates outputs along out edges, and puts newly ready nodes into the ready queue.
  * Call [NodeDone](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2168) and `NodeDone` _**calls `ScheduleReady` again with `inline_ready` being fixed to `null`**_.
  <font size=5>

  ```cpp
  NodeDone(s, state->item->node, ready, stats, nullptr);
  ```

  _in the above call, the last parameter is the inline_ready queue, which is always null, for asynchronous kernels._
</font>

---

# For _**synchronous**_ kernels
* call [Compute](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L1789)
* call [ExecutorState::PropagateOutputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2063)
* Call [NodeDone](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2168)

  ```cpp
  NodeDone(s, item.node, ready, stats, &inline_ready);
  ```

---

# _Let's put too detailed implementations aside, summarize the overall execution flow._

---

<p align="center">
<img src="images/02_op_scheduling.png" width=1000><br/>
</p>
