#!/usr/bin/env julia

include("Recurrent/recurrent.jl")

using Recurrent

srand(1)

const batch_size = 2
const seq_len = 3
const input_dim = 4
const hidden_dim = 4

rand_inputs = randn(batch_size * seq_len, input_dim)

lstm_cell = LSTMCell(input_dim, hidden_dim)

hidden_init = zeros(batch_size, hidden_dim)
cell_init = zeros(batch_size, hidden_dim)

lstm = LSTM_forward(rand_inputs, lstm_cell, input_dim, hidden_dim, seq_len)
