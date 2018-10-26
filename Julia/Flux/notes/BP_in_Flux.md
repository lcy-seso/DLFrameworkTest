Important types to implement BP:

1. Tracker* (`TrackedReal`, `TrakcedVector`, `TrackedMatrix`, ...)
    * [TrackedReal](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/scalar.jl#L1)
      ```julia {.line-numbers}
      struct TrackedReal{T<:Real} <: Real
        data::T
        tracker::Tracked{T}
      end
      ```
    * [TrackedArray](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/array.jl#L9)

        ```julia {.line-numbers}
        struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
          tracker::Tracked{A}
          data::A
          grad::A
          ... # Cor
        end
        ```
1. [Tracked](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/Tracker.jl#L31): a [mutable struct](https://docs.julialang.org/en/v1/manual/types/#Mutable-Composite-Types-1w)

    ```julia
    mutable struct Tracked{T}
      ref::UInt32
      f::Call
      isleaf::Bool
      grad::T
      ... # Cor
    end
    ```

---

Tracker in Flux is a [Julia module](https://docs.julialang.org/en/v1/manual/modules/#modules-1) which is a separate variable workspace.

Backpropagation begins from the training loop in [train.jl](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/train.jl#L55):

```julia {.line-numbers}
function train!(loss, data, opt; cb = () -> ())
  ...
  @progress for d in data
    l = loss(d...)
    @interrupts back!(l)
    opt()
    cb()
end
```
Line 5, function `back!` calculates the gradients.

Let's look into implementation details of `back!`

1. `scalar.jl` --> [back!(x::TrackedReal)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/scalar.jl#L13)
    ```julia {.line-numbers}
    function back!(x::TrackedReal)
      isinf(x) && error("Loss is Inf")
      isnan(x) && error("Loss is NaN")
      return back!(x, 1)
    end
    ```
    * Line 2 ~ 3 is a [short-circuit evaluation](https://docs.julialang.org/en/v1/manual/control-flow/#Short-Circuit-Evaluation-1).

1. `back.jl` --> [back!(x, Δ))](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L59)
    ```julia {.line-numbers}
    function back!(x, Δ)
      istracked(x) || return
      scan(x)  # initialize gradients to zeros
      back(tracker(x), Δ)
      return
    end
    ```
    * Let's look into the implmentation of [scan](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L7):
        ```julia {.line-numbers}
        function scan(x::Tracked)
          x.isleaf && return  # return if x is a leaf.
          ref = x.ref += 1
          if ref == 1
            scan(x.f)
            # initialize gradients to zeors if x is a leaf and reference count is equal to 1
            isdefined(x, :grad) && (x.grad = zero_grad!(x.grad))
          end
          return
        end
        ```
1. back.jl --> [back(x::Tracked, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L34)
    ```julia {.line-numbers}
    function back(x::Tracked, Δ)
      x.isleaf && (x.grad = accum!(x.grad, Δ); return)
      ref = x.ref -= 1
      if ref > 0 || isdefined(x, :grad)
        if isdefined(x, :grad)
          x.grad = accum!(x.grad, Δ)
        else
          x.grad = Δ
        end
        ref == 0 && back_(x.f, x.grad)
      else
        ref == 0 && back_(x.f, Δ)
      end
      return
    end
    ```
    * [back_(c::Call, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L22)
        ```julia {.line-numbers}
        function back_(c::Call, Δ)
          Δs = c.func(Δ)
          (Δs isa Tuple && length(Δs) >= length(c.args)) ||
            error("Gradient is not a tuple of length $(length(c.args))")
          foreach(back, c.args, data.(Δs))
        end
        ```
---

* [back(g::Grads, x::Tracked, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L111)

    ```julia {.line-numbers}
    accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] .+ Δ : Δ
    ```

    ```julia {.line-numbers}
    function back(g::Grads, x::Tracked, Δ)
      x.isleaf && (accum!(g, x, Δ); return)
      ref = x.ref -= 1
      if ref > 0 || haskey(g, x)
        accum!(g, x, Δ)
        ref == 0 && back_(g, x.f, g[x])
      else
        ref == 0 && back_(g, x.f, Δ)
      end
      return
    end
    ```

* [back_(g::Grads, c::Call, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L102)

    ```julia {.line-numbers}
    function back_(g::Grads, c::Call, Δ)
      Δs = c.func(Δ)
      (Δs isa Tuple && length(Δs) >= length(c.args)) ||
        error("Gradient is not a tuple of length $(length(c.args))")
      foreach((x, Δ) -> back(g, x, Δ), c.args, Δs)
    end
    ```
