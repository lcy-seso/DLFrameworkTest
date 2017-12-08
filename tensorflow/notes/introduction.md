### Basic Concepts
#### [Tensor](https://www.tensorflow.org/programmers_guide/tensors)
- API doc : https://www.tensorflow.org/api_docs/python/tf/Tensor
- A **tensor** is a generalization of vectors and matrices to potentially higher dimensions.
- TensorFlow represents tensors as n-dimensional arrays of base datatypes.
- A `tf.Tensor` object represents a partially defined computation that will eventually produce a value.
- TensorFlow programs work by first building a graph of `tf.Tensor objects`.
- A `tf.Tensor` has:
    1. a shape: is always known
    1. a data type: maybe partially known
- With the exception of `tf.Variable`, the value of a tensor is immutable:
    1. which means that in the context of a single execution tensors only have a single value.
    1. Evaluating the same tensor twice can return different values.
- Some types of Tensors are special:
    1. `tf.Variable`
    1. `tf.Constant`
    1. `tf.Placeholder`
    1. `tf.SparseTensor`
#### [Variable](https://www.tensorflow.org/programmers_guide/variables)
- API doc : https://www.tensorflow.org/api_docs/python/tf/Variable
- A `Variable` is to represent shared, persistent state manipulated by your program.
- A `tf.Variable` represents a tensor **whose value can be changed by running ops on it**.
- Unlike `tf.Tensor` objects, a `tf.Variable` exists outside the context of a single `session.run` call.
- Internally, a `tf.Variable` stores a persistent tensor.
- Specific ops allow you to read and modify the values of this tensor.
- These modifications are **visible across multiple `tf.Sessions`**, so multiple workers can see the same values for a `tf.Variable`.
- Create a variable:
    - call the `tf.get_variable` function.
    - can specify `trainable=False` as an argument to `tf.get_variable`.
    - It is required to specify the `Variable`'s name.
    - This name will be used by other replicas to access the same variable.
    - as well as to name this variable's value when checkpointing and exporting models.
    - `tf.get_variable` also allows you to reuse a previously created variable of the same name, which is useful in defining models which reuse layers.
- **Variable collections**
    - Named lists of tensors or other objects, such as `tf.Variable` instances.
    - By default every `tf.Variable` gets placed in:
      1. *tf.GraphKeys.GLOBAL_VARIABLES*: variables that can be shared across multiple devices.
      1. *tf.GraphKeys.TRAINABLE_VARIABLES*: variables for which TensorFlow will calculate gradients.
      1. API
          ```python
          tf.add_to_collection("my_collection_name", my_local)
          tf.get_collection("my_collection_name")
          ```
- Device placement
    - Just like any other TensorFlow operation, you can place variables on particular devices.
- Initializing variables
    - Before you can use a variable, it must be initialized.
    - If you are explicitly creating your own graphs and sessions, you must explicitly initialize the variables.
    - Each `Variable` can be given initializer ops as part of their construction.
    - Explicit initialization:
        1. allows you not to rerun potentially expensive initializers when reloading a model from a checkpoint.
        1. allows determinism when randomly-initialized variables are shared in a distributed setting.
    - To initialize all trainable variables in one go, before training starts, call `tf.global_variables_initializer()`.
- Using Variable
    - To use the value of a `tf.Variable` in a TensorFlow graph, simply treat it like a normal `tf.Tensor`.
- Sharing variable
    - Explicitly passing `tf.Variable` objects around.
    - Implicitly wrapping `tf.Variable` objects within tf.variable_scope objects.
- Name scope
    - Each layer is created beneath a unique `tf.name_scope` that acts as a prefix to the items created within that scope.
    - Each variable is given initializer ops as part of their construction.

#### [Graphs and Session](https://www.tensorflow.org/programmers_guide/graphs)
- TensorFlow uses a dataflow graph to computation in terms of the dependencies between individual operations.
    1. first define the dataflow graph
    1. then create a TensorFlow session to run parts of the graph across a set of local and remote devices.

### How to organize data
1. Dense Matrix
1. Sparse Matrix
1. How to organize sequence
### How to train
### Evaluate

### Save and Load Models
### How to Infer
