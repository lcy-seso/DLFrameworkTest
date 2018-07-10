#!/usr/bin/env julia

abstract type NN
end

mutable struct Param
  n::Integer    # input size
  d::Integer    # output size
  w::AbstractArray{AbstractFloat}    # learnable weight matrix
  dw::AbstractArray{AbstractFloat}   # gradients of learnable weight matrix

  Param(n::Integer) = new(n, n, randn(n, n), randn(n, n))
  Param(n::Integer, d::Integer) = new(n, d, randn(n, d), randn(n, d))
  Param(n::Integer, d::Integer, w::Array, dw::Array) = new(n, d, w, dw)
end

randParam(n::Integer, d::Integer, std::Real=0.1) = Param(
        n, d, randn(n, d) * std, zeros(n, d))
onesParam(n::Integer, d::Integer) = Param(n, d, ones(n, d), zeros(n, d))

function softmax(m::Param)
  out = Param(m.n, m.d)
  maxval = maximum(m.w, 2)
  out.w .= exp.(m.w .- maxval)
  out.w ./= sum(out.w, 2)
  return out
end

 σ(x) = 1.0 / (1.0 + exp(-x))
