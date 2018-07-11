#!/usr/bin/env julia

using CUDAnative

struct LSTMCell
  n::Integer
  hidden_dim::Integer

  input_to_hidden::CuArray
  hidden_to_hidden::CuArray
  bias::CuArray

  LSTMCell(n::Integer, hidden_dim::Integer, std::Real=1e-3) = new(
          n,
          hidden_dim,
          CuArray(randn(n, hidden_dim * 4) .* std),
          CuArray(randn(n, hidden_dim * 4) .* std),
          CuArray(zeros(1, hidden_dim * 4)))
end

σ(x) = 1 / (1 + exp(-x))
tanh(x) = (exp.(x) - exp.(-x)) ./ (exp.(x) + exp.(-x))

function lstm_forward(inputs::CuArray, lstm_param::LSTMCell,
                      forward_step::Integer;
                      cell_act=tanh, output_act=tanh, kwargs...)
  batch_size = size(inputs, 1)  # total token number
  hidden_dim = lstm_param.hidden_dim
  # sequence number in a mini-batch
  sample_num = Int32(batch_size / forward_step)

  if isempty(kwargs)
    hidden_init = CuArray(zeros(sample_num, hidden_dim))
    cell_init = CuArray(zeros(sample_num, hidden_dim))
  else
    @assert (size(kwargs) == 2
             && typeof(hidden_init) <: CuArray
             && typeof(cell_init) <: CuArray)
    hidden_init, cell_init = kwargs
  end

  cell_states = CuArray[]
  hidden_states = CuArray[]

  start = 1
  for i = 1 : forward_step
    input_t = inputs[start : start + sample_num - 1, :]

    hidden_prev = i > 1 ? hidden_states[end] : hidden_init
    cell_prev = i > 1 ? cell_states[end] : cell_init

    input_proj = input_t * lstm_param.input_to_hidden
    hidden_proj = input_t * lstm_param.hidden_to_hidden

    ig = σ.(input_proj[:, 1 : hidden_dim] +
           hidden_proj[:, 1 : hidden_dim] .+
           lstm_param.bias[:, 1 : hidden_dim])
    fg = σ.(input_proj[:, hidden_dim + 1 : 2 * hidden_dim] +
            hidden_proj[:, hidden_dim + 1 : 2 * hidden_dim] .+
            lstm_param.bias[:, hidden_dim + 1 : 2 * hidden_dim])
    cell_candidate = cell_act(
            input_proj[:, 2 * hidden_dim + 1 : 3 * hidden_dim] +
            hidden_proj[:, 2 * hidden_dim + 1 : 3 * hidden_dim] .+
            lstm_param.bias[:, 2 * hidden_dim + 1 : 3 * hidden_dim])

    push!(cell_states, ig .* cell_candidate + fg .* cell_prev)

    og = σ.(input_proj[:, 3 * hidden_dim + 1 : hidden_dim * 4] +
           hidden_proj[:, 3 * hidden_dim + 1 : hidden_dim * 4] .+
          lstm_param.bias[:, 3 * hidden_dim + 1 : hidden_dim * 4])

    push!(hidden_states, og .* output_act(cell_states[end]))
    start += sample_num
  end

  return (cell_states, hidden_states)
end
