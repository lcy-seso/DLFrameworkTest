#!/usr/bin/env julia
module Recurrent

using CuArrays

include("lstm.jl")

export LSTMCell
export σ, lstm_forward


end # module
