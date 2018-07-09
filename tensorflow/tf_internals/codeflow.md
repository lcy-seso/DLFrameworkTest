
[Rendezvous](https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/core/framework/rendezvous.h#L30)

## create a `session`
* The execution flow starts from Python end, session --> constructor --> C API:  [TF_NewSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.cc#L2444)
* C API `TF_NewSession` uses the [factory method pattern](https://en.wikipedia.org/wiki/Factory_method_pattern). The local training will invoke [NewSession]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L149) implemented in  [DirectSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.h#L55).
  * `NewSession` invokes `DirectSession`'s [`Cor`]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L244)
      1. create global thread pool.
      1. add device.

## `session.run`

* [DirectSession::Run](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L645)

  * [DirectSession::GetOrCreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1255)
    * [DirectSession::CreateGraphs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1362)
      * Create _**several graphs**_ given the existing `GraphDef` and the input feeds and fetches
      * [Partition](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1452)
      * [ConvertGraphDefToGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1481)
      * [graph optimization pass](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1486) : post partition
    * [DirectSession::CreateExecutors](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L1094)
  * [DirectSession::RunInternal](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L433)
    * The main step in `RunInternal` is to call [RunAsync](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L572)
