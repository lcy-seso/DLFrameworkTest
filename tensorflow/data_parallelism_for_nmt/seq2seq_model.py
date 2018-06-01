#!/usr/bin/env python
#coding=utf-8
import pdb
import os

from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

import variable_mgr
import variable_mgr_util
from iterator_helper import get_iterator

__all__ = [
    "Seq2SeqModel",
    "hparams",
]

hparams = tf.contrib.training.HParams(
    src_file_name="data/train.en",
    tgt_file_name="data/train.de",
    src_vocab_file="data/vocab.50K.en",
    tgt_vocab_file="data/vocab.50K.de",
    src_vocab_size=50000,
    tgt_vocab_size=50000,
    bos="<s>",
    eos="</s>",
    unk_id=0,
    src_max_len=None,
    tgt_max_len=None,
    num_parallel_calls=4,
    num_buckets=5,
    output_buffer_size=None,
    disable_shuffle=False,
    # when using multi-gpu cards, this means bath size per card.
    batch_size=360,

    # hyper parameters for model topology
    time_major=False,
    dropout=0.,
    unit_type="lstm",
    num_units=512,
    forget_bias=1.,
    embedding_dim=512,
    encoder_type="bi",
    num_encoder_layers=4,
    # TODO(caoying) The current implementation requries encoder and decoder has
    # the same number RNN cells.
    num_decoder_layers=4,
    optimizer="adam",
    learning_rate=0.001,
    num_keep_ckpts=5,
    max_gradient_norm=5.,

    # parameter server places
    # variable_update="replicated",
    variable_update="parameter_server",
    param_server_device="cpu",
    local_parameter_device="gpu",

    # used for all reduced algorithm
    num_gpus=2,
    variable_consistency="strong",
    gradient_repacking=4,
    all_reduce_spec="nccl",
    agg_small_grads_max_bytes=0,
    agg_small_grads_max_group=10, )


class Seq2SeqModel(object):
    def __init__(self,
                 num_gpus,
                 hparams,
                 mode=tf.contrib.learn.ModeKeys.TRAIN,
                 worker_prefix=""):
        self.params = hparams
        self.num_gpus = num_gpus
        # NOTE: batch size passed to get_iterator here is batch size for a single
        # GPU card. the total batch size is num_splits *  hparams.batch_size
        self.iterator = get_iterator(
            src_file_name=hparams.src_file_name,
            tgt_file_name=hparams.tgt_file_name,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            batch_size=hparams.batch_size,
            num_splits=num_gpus,
            disable_shuffle=True,
            output_buffer_size=num_gpus * 1000 * hparams.batch_size)

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
        # devices for computation workers
        self.raw_devices = [
            "%s/%s:%i" % (worker_prefix, hparams.local_parameter_device, i)
            for i in xrange(self.num_gpus)
        ]

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
            self.build_mode_replica,
            hparams=hparams,
            source=self.source,
            target_input=self.target_input,
            target_output=self.target_output,
            source_sequence_length=self.source_sequence_length,
            target_sequence_length=self.target_sequence_length)

        fetches_list = nest.flatten(list(self.fetches.values()))
        self.main_fetch_group = tf.group(*fetches_list)

        local_var_init_op = tf.local_variables_initializer()
        table_init_ops = tf.tables_initializer()
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
            logits, loss, final_context_state = self.build_mode_replica(
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

        # merget gradients using PS mode or all-reduce algorithm
        apply_gradient_devices, gradient_state = (
            self.variable_mgr.preprocess_device_grads(device_grads))

        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                average_loss = tf.reduce_mean(losses)
                avg_grads = self.variable_mgr.get_gradients_to_apply(
                    gradient_state)

            #TODO(caoying): add gradient clipping.
            self.learning_rate = tf.constant(hparams.learning_rate)
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
        fetches["average_loss"] = average_loss / tf.to_float(self.batch_size)
        return fetches

    def build_mode_replica(self,
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
            if hparams.encoder_type == "uni":
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
            else:
                #TODO(caoying): not implemented yet.
                raise NotImplementedError("To be implemented")

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
