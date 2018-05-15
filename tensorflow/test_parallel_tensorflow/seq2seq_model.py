#!/usr/bin/env python
#coding=utf-8
import pdb
import os

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense

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
    batch_size=100,

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
    max_gradient_norm=5., )


class Seq2SeqModel(object):
    def __init__(self,
                 gpu_num,
                 iterator,
                 hparams,
                 mode=tf.contrib.learn.ModeKeys.TRAIN):
        self.device_merge_gradient = "/gpu:0"

        self.iterator = iterator
        self.mode = mode

        self.source = iterator.source
        self.target_input = iterator.target_input
        self.target_output = iterator.target_output
        self.source_sequence_length = iterator.source_sequence_length
        self.target_sequence_length = iterator.target_sequence_length

        self.batch_size = tf.reduce_sum(self.target_sequence_length)

        if gpu_num > 1:
            res = self._make_parallel(
                self.build_graph,
                gpu_num,
                hparams=hparams,
                source=self.source,
                target_input=self.target_input,
                target_output=self.target_output,
                source_sequence_length=self.source_sequence_length,
                target_sequence_length=self.target_sequence_length)

            self.logits = res[0]
            self.loss = res[1]
            self.final_context_state = res[2]
        else:
            self.logits, self.loss, self.final_context_state = self.build_graph(
                self.source, self.target_input, self.target_output,
                self.source_sequence_length, self.target_sequence_length,
                hparams)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.device(self.device_merge_gradient):
                self.train_loss = self.loss / tf.to_float(self.batch_size)
                self.word_count = tf.reduce_sum(
                    self.iterator.source_sequence_length) + tf.reduce_sum(
                        self.iterator.target_sequence_length)

                self.learning_rate = tf.constant(hparams.learning_rate)
                self.global_step = tf.Variable(0, trainable=False)

                # Optimizer
                if hparams.optimizer == "sgd":
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                elif hparams.optimizer == "adam":
                    opt = tf.train.AdamOptimizer(self.learning_rate)

                colocate_gradients_with_ops = True
                params = tf.trainable_variables()
                gradients = tf.gradients(
                    self.train_loss,
                    params,
                    colocate_gradients_with_ops=colocate_gradients_with_ops, )

                clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                    gradients, hparams.max_gradient_norm)
                self.grad_norm = gradient_norm

                self.update = opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step)

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            #TODO(caoying): not implemented yet.
            raise NotImplementedError("To be implemented")
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            #TODO(caoying): not implemented yet.
            raise NotImplementedError("To be implemented")

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    def _merge_outputs(self, out_splits, device_str):
        assert len(out_splits), "No output is given."

        split_num = len(out_splits)
        if split_num == 1: return out_splits[0]

        out = []
        output_num = len(out_splits[0])
        for out_i in range(output_num):
            out_i_splits = [out_splits[i][out_i] for i in range(split_num)]
            with tf.device(device_str):
                out.append(tf.add_n(out_i_splits))
        return out

    def _make_parallel(self, fn, gpu_num, **kwargs):
        """ Wrapper for data parallelism.
        """

        in_splits = {}
        for k, v in kwargs.items():
            if isinstance(v, tf.Tensor):
                in_splits[k] = tf.split(v, gpu_num)
            else:
                in_splits[k] = [v] * gpu_num

        # FIXME(caoying) Temprorarily hard-code this for quick experiments.
        # Need an elegant implementation.
        target_output_splits = in_splits["target_output"]
        in_splits["target_output"] = []
        for i in range(gpu_num):
            size = [-1, tf.reduce_max(in_splits["target_sequence_length"][i])]
            in_splits["target_output"].append(
                tf.slice(target_output_splits[i], [0, 0], size))

        out_splits = []
        for i in range(gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.name_scope("replica_%02d" % (i)) as scope:
                    with tf.variable_scope(
                            tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        out_i = fn(**{k: v[i] for k, v in in_splits.items()})
                        out_splits.append(out_i)

        return self._merge_outputs(out_splits, self.device_merge_gradient)

    def build_graph(self,
                    source,
                    target_input,
                    target_output,
                    source_sequence_length,
                    target_sequence_length,
                    hparams,
                    dtype=tf.float32,
                    scope=None):

        source_embed = self._init_embeddings("src_embedding",
                                             hparams.embedding_dim,
                                             hparams.src_vocab_size, dtype)
        target_embed = self._init_embeddings("tgt_embedding",
                                             hparams.embedding_dim,
                                             hparams.tgt_vocab_size, dtype)

        with tf.variable_scope("decoder/output_projection"):
            self.output_layer = Dense(
                hparams.tgt_vocab_size,
                use_bias=False,
                name="output_projection")

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            encoder_outputs, encoder_state = self._build_encoder(
                hparams, source, source_embed, source_sequence_length, dtype)

            logits, final_context_state = self._build_decoder(
                hparams, encoder_outputs, encoder_state,
                source_sequence_length, target_embed, target_input,
                target_sequence_length)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                loss = self._compute_loss(logits, target_output,
                                          target_sequence_length,
                                          hparams.time_major)
            else:
                loss = None

            return logits, loss, final_context_state

    def _init_embeddings(self, embed_name, embedding_dim, vocab_size, dtype):
        return tf.get_variable(
            embed_name, [vocab_size, embedding_dim], dtype=dtype)

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

    def _build_encoder(self, hparams, source, source_embed,
                       source_sequence_length, dtype):
        num_layers = hparams.num_encoder_layers

        if hparams.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            encoder_emb_inp = tf.nn.embedding_lookup(source_embed, source)

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
                       source_sequence_length, target_embed, target_input,
                       target_sequence_length):

        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                if hparams.time_major:
                    target_input = tf.transpose(target_input)

                decoder_emb_inp = tf.nn.embedding_lookup(target_embed,
                                                         target_input)

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
