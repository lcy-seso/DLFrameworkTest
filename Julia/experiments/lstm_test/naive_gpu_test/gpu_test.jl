#!/usr/bin/env julia
include("Recurrent/recurrent.jl")

using .Recurrent
using CuArrays: CuArray

srand(1)

const batch_size = 512
const seq_len = 10
const input_dim = 512
const hidden_dim = 512

rand_inputs_d = CuArray(randn(batch_size * seq_len, input_dim))
lstm_cell = LSTMCell(input_dim, hidden_dim)


for i = 1 : 10
  cell_states, hidden_states = lstm_forward(rand_inputs_d, lstm_cell, seq_len)

  # println("cell states : ")
  # display(cell_states)

  # println("hidden_states : ")
  # display(hidden_states)
end
