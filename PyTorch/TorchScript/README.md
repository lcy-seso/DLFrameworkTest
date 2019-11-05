# [TorchScript](https://pytorch.org/docs/stable/jit.html)

## What is the TorchScript program?

1. TorchScript is an intermediate representation of _**a PyTorch model**_. Specifically, a PyTorch model is users' models defined by subclassinng [nn.Module](https://pytorch.org/docs/stable/nn.html#module).
1. Pure Python prgram could be incrementally translated into a TorchScript program.
1. A TorchScript program can be run independently from Python in a high-performance environment such as C++.

## Why TorchScript?

PyTorch is an imperative style library. It is easy to write, but suffer from high interpretive overhead and are not easily deployable in production or mobile settings. Performance favors declarative programming style, and it exposes more opportunities for compile-time analysis.

Below is cited from official doc:

>So why did we do all this? There are several reasons:
>
>1. TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the same instance simultaneously.
>1. This format allows us to save the whole model to disk and load it into another environment, such as in a server written in a language other than Python.
>1. TorchScript gives us a representation in which we can do compiler optimizations on the code to provide more efficient execution.
>1. TorchScript allows us to interface with many backend/device runtimes that require a broader view of the program than individual operators.


## How to convert PyTorch modules to TorchScript?

1. Trace an existing module.
1. Use scripting to directly compile a model.

# Reference

1. [TorchScript](https://pytorch.org/docs/stable/jit.html)
1. [JIT Technical Overview](https://github.com/pytorch/pytorch/blob/83331bf12373c54eb7f3af7ca4e50b91a22d23ba/torch/csrc/jit/docs/OVERVIEW.md#graph) 
1. [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
