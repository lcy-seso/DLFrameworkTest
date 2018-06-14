#!/usr/bin/env python
#coding=utf-8
import pdb
import os

from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.python.framework import dtypes
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import variable_mgr
import variable_mgr_util
from iterator_helper import get_iterator, get_synthetic_data
from utils import get_available_gpus

CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER
CUDNN_RNN_UNIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
CUDNN_RNN_BIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

__all__ = [
    "Seq2SeqModel",
]


class CudnnRNNModel(object):
    def __init__(self,
                 inputs,
                 rnn_mode,
                 num_layers,
                 num_units,
                 input_size,
                 initial_state=None,
                 direction=CUDNN_RNN_UNIDIRECTION,
                 dropout=0.,
                 dtype=dtypes.float32,
                 training=False,
                 seed=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        if rnn_mode == "cudnn_lstm":
            model_fn = cudnn_rnn.CudnnLSTM
        else:
            #(TODO) support other cudnn RNN ops.
            raise NotImplementedError(
                "Invalid rnn_mode: %s. Not implemented yet." % rnn_mode)

        if initial_state is not None:
            assert isinstance(initial_state, tuple)

        self._initial_state = initial_state

        self._rnn = model_fn(
            num_layers,
            num_units,
            direction=direction,
            dropout=dropout,
            dtype=dtype,
            seed=seed,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

        # parameter passed to biuld is the shape of input Tensor.
        self._rnn.build([None, None, input_size])

        # self._outputs is a tensor of shape:
        # [seq_len, batch_size, num_directions * num_units]
        # self._output_state is a tensor of shape:
        # [num_layers * num_dirs, batch_size, num_units]

        self._outputs, self._output_state = self._rnn(
            inputs, initial_state=self._initial_state, training=training)

    @property
    def inputs(self):
        return self._inputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_state(self):
        return self._output_state

    @property
    def rnn(self):
        return self._rnn

    @property
    def total_sum(self):
        return self._AddUp(self.outputs, self.output_state)


class Seq2SeqModel(object):
    def __init__(self,
                 hparams,
                 mode=tf.contrib.learn.ModeKeys.TRAIN,
                 worker_prefix=""):
        self.params = hparams
        self.num_gpus = len(get_available_gpus())

        # devices for computation workers
        self.raw_devices = [
            "%s/%s:%i" % (worker_prefix, hparams.local_parameter_device, i)
            for i in xrange(self.num_gpus)
        ]
        if hparams.use_synthetic_data:
            self.iterator = get_synthetic_data(
                hparams.src_max_len, hparams.batch_size, hparams.time_major,
                hparams.src_vocab_size, hparams.tgt_vocab_size,
                self.raw_devices)
        else:
            # NOTE: batch size passed to get_iterator here is batch size for a
            # single GPU card. the total batch size is:
            # num_splits *  hparams.batch_size

            self.iterator = get_iterator(
                src_file_name=hparams.src_file_name,
                tgt_file_name=hparams.tgt_file_name,
                src_vocab_file=hparams.src_vocab_file,
                tgt_vocab_file=hparams.tgt_vocab_file,
                batch_size=hparams.batch_size,
                num_splits=self.num_gpus,
                disable_shuffle=True,
                output_buffer_size=self.num_gpus * 1000 *
                self.params.batch_size)

        self.word_count = tf.reduce_sum(
            self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)
        self.mode = mode

        self.source = self.iterator.source
        self.target_input = self.iterator.target_input
        self.target_output = self.iterator.target_output
        self.source_sequence_length = self.iterator.source_sequence_length
        self.target_sequence_length = self.iterator.target_sequence_length

        self.batch_size = tf.reduce_sum(self.target_sequence_length)

        # specified which device to place the master copy of all the
        # trainable parameters
        self.param_server_device = hparams.param_server_device
        self.local_parameter_device = hparams.local_parameter_device
        # Device to use for ops that need to always run on the local worker"s CPU.
        self.cpu_device = "%s/cpu:0" % worker_prefix

        if hparams.variable_update == "replicated":
            self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
                self, self.params.all_reduce_spec,
                self.params.agg_small_grads_max_bytes,
                self.params.agg_small_grads_max_group)
        elif hparams.variable_update == "parameter_server":
            self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
            raise ValueError(
                ("Unsupported setting for variable_update. "
                 "Possible setting is parameter_server and replicated."))

        # Device to use for running on the local worker"s compute device, but
        # with variables assigned to parameter server devices.
        self.devices = self.variable_mgr.get_devices()

        self.global_step_device = self.cpu_device

        self.fetches = self.make_data_parallel(
            self.build_model_replica,
            hparams=hparams,
            source=self.source,
            target_input=self.target_input,
            target_output=self.target_output,
            source_sequence_length=self.source_sequence_length,
            target_sequence_length=self.target_sequence_length)

        fetches_list = nest.flatten(list(self.fetches.values()))
        self.main_fetch_group = tf.group(*fetches_list)

        local_var_init_op = tf.local_variables_initializer()
        table_init_ops = (tf.tables_initializer()
                          if self.params.use_synthetic_data else None)
        variable_mgr_init_ops = [local_var_init_op]
        if table_init_ops:
            variable_mgr_init_ops.extend([table_init_ops])
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
        self.local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    def make_data_parallel(self, fn, **kwargs):
        """ Wrapper for data parallelism.
        """

        in_splits = {}
        for k, v in kwargs.items():
            if isinstance(v, list):  # input tensors
                in_splits[k] = v
            else:  # hyper parameters
                in_splits[k] = [v] * len(self.devices)

        losses = []
        device_grads = []
        for device_num in range(len(self.devices)):
            # when using PS mode, learnable parameters are placed on different
            # GPU devices. when using all-reduced algorithm, each GPU card has
            # a entire copy of model parameters.
            with self.variable_mgr.create_outer_variable_scope(
                    device_num), tf.name_scope("tower_%i" %
                                               device_num) as name_scope:
                results = self.add_forward_pass_and_gradients(
                    device_num, device_num,
                    **{k: v[device_num]
                       for k, v in in_splits.items()})
                losses.append(results["loss"])
                device_grads.append(results["gradvars"])

        with tf.device(self.global_step_device):
            self.global_step = tf.train.get_or_create_global_step()

        return self.build_gradient_merge_and_update(self.global_step, losses,
                                                    device_grads)

    def add_forward_pass_and_gradients(self, rel_device_num, abs_device_num,
                                       **inputs):
        """
        Args:
          rel_device_num: local worker device index.
          abs_device_num: global graph device index.
        """

        with tf.device(self.devices[rel_device_num]):
            logits, loss, final_context_state = self.build_model_replica(
                source=inputs["source"],
                target_input=inputs["target_input"],
                target_output=inputs["target_output"],
                source_sequence_length=inputs["source_sequence_length"],
                target_sequence_length=inputs["target_sequence_length"],
                hparams=inputs["hparams"])

            params = self.variable_mgr.trainable_variables_on_device(
                rel_device_num, abs_device_num)
            grads = tf.gradients(
                loss, params, aggregation_method=tf.AggregationMethod.DEFAULT)
            gradvars = list(zip(grads, params))
            return {
                "logits": logits,
                "loss": loss,
                "final_context_state": final_context_state,
                "gradvars": gradvars,
            }

    def build_gradient_merge_and_update(self, global_step, losses,
                                        device_grads):
        fetches = {}

        apply_gradient_devices = self.devices
        gradient_state = device_grads

        training_ops = []

        # gradient_state is the merged gradient.
        apply_gradient_devices, gradient_state = (
            self.variable_mgr.preprocess_device_grads(
                device_grads, self.params.independent_replica))

        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                average_loss = (losses[d] if self.params.independent_replica
                                else tf.reduce_sum(losses))
                avg_grads = self.variable_mgr.get_gradients_to_apply(
                    d, gradient_state)

            #TODO(caoying): add gradient clipping.
            self.learning_rate = tf.constant(self.params.learning_rate)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            loss_scale_params = variable_mgr_util.AutoLossScaleParams(
                enable_auto_loss_scale=False,
                loss_scale=None,
                loss_scale_normal_steps=None,
                inc_loss_scale_every_n=1000,
                is_chief=True)

            # append optimizer operators into the graph
            self.variable_mgr.append_apply_gradients_ops(
                gradient_state, opt, avg_grads, training_ops,
                loss_scale_params)

        fetches["train_op"] = tf.group(training_ops)
        fetches["average_loss"] = (average_loss
                                   if self.params.independent_replica else
                                   average_loss / tf.to_float(self.batch_size))
        return fetches

    def build_model_replica(self,
                            source,
                            target_input,
                            target_output,
                            source_sequence_length,
                            target_sequence_length,
                            hparams,
                            dtype=tf.float32):

        self.output_layer = Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

        encoder_outputs, encoder_state = self._build_encoder(
            hparams, source, source_sequence_length, dtype)

        logits, final_context_state = self._build_decoder(
            hparams, encoder_outputs, encoder_state, source_sequence_length,
            target_input, target_sequence_length, dtype)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            loss = self._compute_loss(logits, target_output,
                                      target_sequence_length,
                                      hparams.time_major)
        else:
            loss = None

        return logits, loss, final_context_state

    def _init_embeddings(self, input, embed_name, embedding_dim, vocab_size,
                         dtype):
        embed_var = tf.get_variable(
            embed_name, [vocab_size, embedding_dim], dtype=dtype)
        return tf.nn.embedding_lookup(embed_var, input)

    def _single_cell(self, unit_type, num_units, forget_bias, dropout, mode):
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        # Cell Type
        if unit_type == "lstm":
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units, forget_bias=forget_bias)
        elif unit_type == "gru":
            single_cell = tf.contrib.rnn.GRUCell(num_units)
        elif unit_type == "layer_norm_lstm":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units, forget_bias=forget_bias, layer_norm=True)
        elif unit_type == "nas":
            single_cell = tf.contrib.rnn.NASCell(num_units)
        else:
            raise ValueError("Unknown unit type %s!" % unit_type)

        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        return single_cell

    def _build_rnn_cell(self, num_layers, unit_type, num_units, forget_bias,
                        dropout, mode):
        cell_list = []
        for i in range(num_layers):
            cell_list.append(
                self._single_cell(
                    unit_type=unit_type,
                    num_units=num_units,
                    forget_bias=forget_bias,
                    dropout=dropout,
                    mode=mode, ))
        if num_layers == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_bidirectional_rnn(self, inputs, num_layers, unit_type,
                                 num_units, forget_bias, dropout, mode,
                                 sequence_length, dtype, time_major):
        # Construct forward and backward cells
        fw_cell = self._build_rnn_cell(num_layers, unit_type, num_units,
                                       forget_bias, dropout, mode)
        bw_cell = self._build_rnn_cell(num_layers, unit_type, num_units,
                                       forget_bias, dropout, mode)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=time_major,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state

    def _build_encoder(self, hparams, source, source_sequence_length, dtype):
        num_layers = hparams.num_encoder_layers

        if hparams.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            encoder_emb_inp = self._init_embeddings(
                source, "src_embedding", hparams.embedding_dim,
                hparams.src_vocab_size, dtype)

            # Encoder_outputs: [batch_size, max_time, num_units]
            if hparams.encoder_type == "cudnn_lstm":
                if not hparams.time_major:
                    # NOTE: inputs of the cudnn_lstm should be time major:
                    # [sequence_length, batch_size, hidden_dim]
                    encoder_emb_inp = tf.transpose(
                        encoder_emb_inp, perm=[1, 0, 2])

                # TODO: hard code batch size and sequence length for current
                # experiment.
                rnn = CudnnRNNModel(
                    inputs=encoder_emb_inp,
                    rnn_mode=hparams.encoder_type,
                    num_layers=(hparams.num_encoder_layers
                                if hparams.direction == "uni" else
                                hparams.num_encoder_layers / 2),
                    num_units=hparams.num_units,
                    input_size=encoder_emb_inp.get_shape().as_list()[-1],
                    direction=(CUDNN_RNN_UNIDIRECTION
                               if hparams.direction == "uni" else
                               CUDNN_RNN_BIDIRECTION),
                    dropout=hparams.dropout,
                    dtype=dtypes.float32,
                    training=True)  #TODO: support inference and generation.

                encoder_outputs = rnn._outputs
                encoder_state = rnn._output_state

            elif hparams.encoder_type == "uni":
                cell = self._build_rnn_cell(
                    num_layers=hparams.num_encoder_layers,
                    unit_type=hparams.unit_type,
                    num_units=hparams.num_units,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=source_sequence_length,
                    time_major=hparams.time_major,
                    swap_memory=True)

            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)

                encoder_outputs, bi_encoder_state = (
                    self._build_bidirectional_rnn(
                        inputs=encoder_emb_inp,
                        num_layers=num_bi_layers,
                        unit_type=hparams.unit_type,
                        num_units=hparams.num_units,
                        forget_bias=hparams.forget_bias,
                        dropout=hparams.dropout,
                        mode=self.mode,
                        sequence_length=source_sequence_length,
                        dtype=dtype,
                        time_major=hparams.time_major))

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(
                            bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(
                            bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" %
                                 hparams.encoder_type)
        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state):
        cell = self._build_rnn_cell(
            num_layers=hparams.num_decoder_layers,
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode)

        if (self.mode == tf.contrib.learn.ModeKeys.INFER and
                hparams.beam_width > 0):
            #TODO(caoying): not implemented yet.
            raise NotImplementedError("To be implemented")
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state

    def _build_cudnn_rnn_decoder(self, hparams, encoder_state, decoder_emb,
                                 dtype):
        if not hparams.time_major:
            # NOTE: inputs of the cudnn_lstm should be time major:
            # [sequence_length, batch_size, hidden_dim]
            decoder_emb = tf.transpose(decoder_emb, perm=[1, 0, 2])

        rnn = CudnnRNNModel(
            inputs=decoder_emb,
            rnn_mode=hparams.encoder_type,
            num_layers=hparams.num_decoder_layers,
            num_units=hparams.num_units,
            input_size=decoder_emb.get_shape().as_list()[-1],
            direction=CUDNN_RNN_UNIDIRECTION,
            initial_state=encoder_state,
            dropout=hparams.dropout,
            dtype=dtypes.float32,
            training=True)  #TODO: support inference and generation.

        outputs = (rnn.outputs if hparams.time_major else tf.transpose(
            rnn.outputs, perm=[1, 0, 2]))

        logits = self.output_layer(outputs)
        return logits

    def _build_decoder(self, hparams, encoder_outputs, encoder_state,
                       source_sequence_length, target_input,
                       target_sequence_length, dtype):

        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                if hparams.time_major:
                    target_input = tf.transpose(target_input)

                decoder_emb_inp = self._init_embeddings(
                    target_input, "tgt_embedding", hparams.embedding_dim,
                    hparams.tgt_vocab_size, dtype)

            if hparams.encoder_type == "cudnn_lstm":
                # (TODO): Is it possible that in the encoder-decoder
                # architecture, use cudnn lstm to implement encoder and use
                # dynamic_rnn to implement decoder.
                logits = self._build_cudnn_rnn_decoder(hparams, encoder_state,
                                                       decoder_emb_inp, dtype)
                final_context_state = None
            else:
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp,
                    target_sequence_length,
                    time_major=hparams.time_major)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state, )

                # Dynamic decoding
                (outputs, final_context_state,
                 _) = tf.contrib.seq2seq.dynamic_decode(
                     decoder,
                     output_time_major=hparams.time_major,
                     swap_memory=True,
                     scope=decoder_scope)
                logits = self.output_layer(outputs.rnn_output)

        return logits, final_context_state

    def _get_max_time(self, tensor, time_major):
        time_axis = 0 if time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _compute_loss(self, logits, target_output, target_sequence_length,
                      time_major):
        if time_major:
            target_output = tf.transpose(target_output)

        max_time = self._get_max_time(target_output, time_major)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            target_sequence_length, max_time, dtype=logits.dtype)
        if time_major:
            target_weights = tf.transpose(target_weights)
        return tf.reduce_sum(crossent * target_weights)
