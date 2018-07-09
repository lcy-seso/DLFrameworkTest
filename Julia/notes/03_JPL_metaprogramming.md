# [Metaprogramming](https://docs.julialang.org/en/release-0.4/manual/metaprogramming/)

* Julia represents its own code as a data structure of the language itself.
  * allow sophisticated code generation without extra build steps
  * allow true Lisp-style macros <span style="background-color:#A3D1D1;">_**operating at the level of abstract syntax trees**_</span>.
  * powerful [reflection](https://en.wikipedia.org/wiki/Reflection_%28computer_programming%29) capabilities

## Program representation

* every Julia program starts life as a string
* parse (I understand this function as lexical analysis) each string into an object called an expression, represented by the Julia's type `Expr`.
  * `Expr` objects contain three parts:
    1. `Symbol`
        * In the context of an expression, symbols are used to indicate access to variables.
  * when an expression is evaluated, a symbol is replaced with the value bound to that symbol in the appropriate scope.
    2. the expression arguments
    3. the expression result type
