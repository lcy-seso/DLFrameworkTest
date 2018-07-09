* The execution flow starts from Python end, session --> constructor --> C API:  [TF_NewSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.cc#L2444)
* C API `TF_NewSession` uses the [factory method pattern](https://en.wikipedia.org/wiki/Factory_method_pattern). The local training will invoke [NewSession]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L149) implemented in  [DirectSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.h#L55).
  * `NewSession` invokes `DirectSession`'s [`Cor`]( https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc#L244)
* session.run -->
