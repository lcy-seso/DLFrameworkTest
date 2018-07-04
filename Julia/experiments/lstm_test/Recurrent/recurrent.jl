#!/usr/bin/env julia
module Recurrent

export Param, randParam, onesParam, softmax, sigmoid
export LSTMCell, LSTM_forward

include("common.jl")
include("lstm.jl")

end  # module
