#!/usr/bin/env julia

mutable struct LSTMCell  # for test, not optimized
  # input gate parameters
  wix::Param
  wih::Param
  bi::Param

  # forget gate parameters
  wfx::Param
  wfh::Param
  bf::Param

  # parameters for candidate input computation
  wcx::Param
  wch::Param
  bc::Param

  # output gate parameters
  wox::Param
  woh::Param
  bo::Param

  function LSTMCell(input_dim::Integer, hidden_dim::Integer, std::Real=1e-3)
    wix = randParam(input_dim, hidden_dim, std)
    wih = randParam(hidden_dim, hidden_dim, std)
    bi = randParam(1, hidden_dim, std)

    wfx = randParam(input_dim, hidden_dim, std)
    wfh = randParam(hidden_dim, hidden_dim, std)
    bf = randParam(1, hidden_dim, std)

    wcx = randParam(input_dim, hidden_dim, std)
    wch = randParam(hidden_dim, hidden_dim, std)
    bc = randParam(1, hidden_dim, std)

    wox = randParam(input_dim, hidden_dim, std)
    woh = randParam(hidden_dim, hidden_dim, std)
    bo = randParam(1, hidden_dim, std)

    new(wix, wih, bi, wfx, wfh, bf, wcx, wch, bc, wox, woh, bo)
  end
end

@inline function LSTM_forward(inputs::AbstractArray, lstm_cell::LSTMCell,
                              input_dim::Integer, hidden_dim::Integer,
                              seq_len::Integer;
                              cell_act=tanh, output_act=tanh, kwargs...)
  batch_size = size(inputs, 1)

  hidden_states::AbstractArray{AbstractFloat} = zeros(batch_size, hidden_dim)
  cell_states::AbstractArray{AbstractFloat} = zeros(batch_size, hidden_dim)

  sample_num = Integer(batch_size / seq_len)
  if isempty(kwargs)
    hidden_init = zeros(sample_num, hidden_dim)
    cell_init = zeros(sample_num, hidden_dim)
  else
    @assert size(kwargs) == 2
    hidden_init, cell_init = kwargs
  end

  start = 1
  for i = 1: seq_len
    input_t = inputs[start : start + sample_num - 1, :]
    hidden_prev = (i > 1 ? hidden_states[start - sample_num : start - 1, :] :
                           hidden_init)
    cell_prev = (i > 1 ? cell_states[start - sample_num : start - 1, :] :
                         cell_init)

    # input gate
    ig = σ.(input_t * lstm_cell.wix.w .+
            hidden_prev * lstm_cell.wih.w .+ lstm_cell.bi.w)
    # forget gate
    fg = σ.(input_t * lstm_cell.wfx.w .+
            hidden_prev * lstm_cell.wfh.w .+ lstm_cell.bf.w)

    # candidate cell
    candidate_cell = cell_act.(input_t * lstm_cell.wcx.w .+
                               hidden_prev * lstm_cell.wch.w .+ lstm_cell.bc.w)
    cell_states[start : start + sample_num - 1, :] = (ig .* candidate_cell .+
                                                      fg .* cell_prev)

    # output gate
    og = σ.(input_t * lstm_cell.wox.w .+
            hidden_prev * lstm_cell.woh.w .+ lstm_cell.bo.w)

    hidden_states[start : start + sample_num - 1, :] =
            og .* output_act.(cell_states[start : start + sample_num - 1, :])

    start += sample_num
  end
  return (cell_states, hidden_states)
end

