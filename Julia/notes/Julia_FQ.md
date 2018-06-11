[TOC]

# Julia

## My goals

- [ ] Get familar with Julia's syntax.
- [ ] Figure out are there any package can be directly use for ML or DL.
    - [ ] Can I use them to write a "Hello work" program.
    - [ ] or I have to write almost everything from scratch.
- [ ] How to write GPU code in Julia.
- [ ] Figure out the role JIT / LLVM played in Julia.

## Let's get familiar with Julia's syntax

## [Integers and Floating-Point Numbers](https://github.com/JuliaLang/julia/blob/master/doc/src/manual/integers-and-floating-point-numbers.md)

1. Integers and Floating-Point Numbers

    - Built-in representations of integers and floating-point values are called numeric primitives.
    - All numeric types interoperate naturally without explicit casting.

1. Support [arbitrary-precision integers and floating point numbers](https://docs.julialang.org/en/stable/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic-1): `BigInt`, `BigFloat`
    - wrap the [GNU Multiple Precision Arithmetic Library (GMP)](https://www.mpfr.org/) and the [GNU MPFR Library](https://www.mpfr.org/)
    - at the cost of relatively slower performance

1. Julia allows variables to be immediately preceded by a numeric literal, implying multiplication.

    ```Julia
    x = 3
    y = 2x^2 - 3x + 1
    z = 1.5x^2 - .5x + 1
    ```
    >_The precedence of numeric literal coefficients used for implicit multiplication is higher than other binary operators._

### [String](https://docs.julialang.org/en/stable/manual/strings/)

- The built-in concrete type used for strings (and string literals) in Julia is `String`.
- Support lexicographically compare strings using the standard comparison operators.
- The shortest complete expression after the `$` is taken as the expression whose value is to be interpolated into the string.

   ```bash
   julia> "1 + 2 = $(1 + 2)"
   "1 + 2 = 3"
   ```

### [Functions](https://docs.julialang.org/en/stable/manual/functions/)

>Functions in Julia are first-class objects, then can be:
>1. assigned to variables
>2. called using the standard function call syntax from the variable they have been assigned to
>3. used as arguments
>4. returned as values
>5. created anonymously, without being given a name,

The basic syntax for defining functions in Julia is:

#### Declaration Syntax

1. **traditional function declaration syntax**

    ```julia
    function f(x, y)
        x + y
    end
    ```

    - About the `return` keyword
      - The value returned by a function **by default, is the last expression in the body of the function definition**.
      - the `return` keyword causes a function to return immediately, providing an expression whose value is returned:

        ```julia
        function f(x, y)
            return x + y
            x * y
        end
        ```
      - In a purely linear function body like the usage of `return` is pointless. In conjunction with other control flow, however, `return` is of real use.

1. **the assignment form**

    ```julia
    f(x, y) = x + y
    ```
    - the body of the function **must be** a single expression.
    - Without parentheses, the expression `f` refers to the function object, and can be passed around like any value:
        ```julia
        g = f
        ```

#### Operators Are Functions

In Julia, most operators are just functions with support for special syntax.

- you can apply oprators using parenthesized argument lists:

    ```bash
    julia> 1 + 2 + 3
    6

    julia> +(1,2,3)
    6
    ```

- you can assign and pass around operators such as `+()` and `*()` just like you would with other function values

    ```bash
    julia> f = +;

    julia> f(1,2,3)
    6
    ```

#### Anonymous Functions

```julia
x -> x^2 + 2x - 1
```

or:

```julia
function (x)
    x^2 + 2x -1
end
```

_How to call anonymous functions ?_

#### Multiple Return Values

- return a tuple of values to simulate returning multiple values.
- tuples can be created and destructured without needing parentheses.

    ```julia
    function f(x, y)
        a + b, a * b
    end
    ```

#### [Optional Arguments](https://docs.julialang.org/en/stable/manual/methods/#Note-on-Optional-and-keyword-Arguments-1)

```julia
function parse(T, num, base=10)
    ####
end
```

#### Keyword Arguments

Functions with keyword arguments are defined using **a semicolon** in the signature:

```julia
function plot(x, y; style="solid", width=1)
    ####
end
```

_Keyword argument default values are evaluated onlywhen a corresponding keyword argument is not passed, and in left-to-right order._

- When the function is called, the semicolon is optional: one can either call `plot(x, y, width=2`) or `plot(x, y; width=2)`.
- The types of keyword arguments can be made explicit as follows:
    ```julia
    function f(;x::Int64=1)
        ###
    end
    ```
- Extra keyword arguments can be collected using ..., as in varargs functions:

    ```julia
    function f(x; y=0, kwargs...)
        ###
    end
    ```
    - Inside `f`, kwargs will be a collection of `(key,value)` tuples.

#### [Do-Block Syntax for Function Arguments](https://docs.julialang.org/en/stable/manual/functions/#Do-Block-Syntax-for-Function-Arguments-1)

Passing functions as arguments to other functions is especially awkward to write when the function argument requires multiple lines. For example:

```julia
map(x->begin
           if x < 0 && iseven(x)
               return 0
           elseif x == 0
               return 1
           else
               return x
           end
        end,
    [A, B, C])
```

Julia provides a reserved word `do` for rewriting this code more clearly:

```julia
map([A, B, C]) do x
    if x < 0 && iseven(x)
        return 0
    elseif x == 0
        return 1
    else
        return x
    end
end
```

- The `do x` syntax creates an anonymous function with argument `x` and passes it as the first argument to `map()`.
- Similarly, `do a, b` would create a two-argument anonymous function
- a plain `d`o would declare that what follows is an anonymous function of the form `() -> ....`

### Dot Syntax for Vectorizing Functions

>_**Julia has a special dot syntax that converts any scalar function into a "vectorized" function call, and any operator into a "vectorized" operator, with the special property that nested "dot calls" are fusing: they are combined at the syntax level into a single loop, without allocating temporary arrays.**_

>In Julia, vectorized functions are not required for performance, and indeed it is often beneficial to write your own loops, see [Performance Tips](https://docs.julialang.org/en/stable/manual/performance-tips/#man-performance-tips-1).

- Any Julia function f can be applied elementwise to any array (or other collection) with the syntax `f.(A)`.
- `f.(args...)` is actually equivalent to `broadcast(f, args...)`, see [Broadcast](https://docs.julialang.org/en/stable/manual/arrays/#Broadcasting-1).
- **Nested `f.(args...)` calls are fused into a single broadcast loop**.

  - `sin.(cos.(X))` is equivalent to `broadcast(x -> sin(cos(x)), X)`, similar to `[sin(cos(x)) for x in X]`
  - there is only a single loop over `X`, and a single array is allocated for the result.
      >This loop fusion is not a compiler optimization that may or may not occur, it is a syntactic guarantee whenever nested `f.(args...)` calls are encountered.

**the maximum efficiency is typically achieved when the output array of a vectorized operation is [pre-allocated](https://docs.julialang.org/en/stable/manual/performance-tips/#Pre-allocating-outputs-1).**

---

## My questions

1. How to implement memory managment, or just us garbage collection？
1. Is the support for Tensor (ndarrary) well enough?
1. How to support generic programming like template in C++？
1. Is the support for serilization well enough？
1. What is the recommanded program paradigm? a functional language? OOP?
1. How JIT in Julia benefit deep learning framework?

---

## Let's look at some code snippets

```julia
evalcb = () -> @show loss(Xs[5], Ys[5])
```
