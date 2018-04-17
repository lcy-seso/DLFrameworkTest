#!/usr/bin/env python
#coding=utf-8
import pdb

import tensorflow as tf


class LMConfig(object):
    """Configuration of language model"""
    batch_size = 2048
    time_major = False

    train_data_path = "data/ptb.train.txt"
    vocab_file_path = "data/vocab.txt"
    vocab_size = 10001
    unk_id = 1

    embedding_dim = 32
    hidden_dim = 128
    num_layers = 2

    learning_rate = 1e-3

    num_passes = 50


class RNNLM(object):
    def __init__(self, config, curwd, nxtwd, seq_len, is_training=True):
        self.curwd = curwd
        self.nxtwd = nxtwd
        self.seq_len = seq_len
        self.batch_size = tf.size(nxtwd)

        self.time_major = config.time_major
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.learning_rate = config.learning_rate

        # build the model
        self.logits, self.prediction = self.rnn()
        self.cost = self.cost()
        self.optim = self.optimize()
        self.word_error = self.word_error()

    def input_embedding(self):
        embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.embedding_dim],
            dtype=tf.float32)
        return tf.nn.embedding_lookup(embedding, self.curwd)

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_dim, state_is_tuple=True)

        cells = [lstm_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _inputs = self.input_embedding()
        _outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=_inputs,
            dtype=tf.float32,
            sequence_length=self.seq_len,
            time_major=self.time_major,
            swap_memory=True)

        logits = tf.layers.dense(
            inputs=_outputs, units=self.vocab_size, use_bias=False)

        return logits, tf.nn.softmax(logits)

    def cost(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.nxtwd, logits=self.logits)
        target_weights = tf.sequence_mask(
            self.seq_len,
            self.get_max_time(self.logits),
            dtype=self.logits.dtype)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(cross_entropy *
                             target_weights) / tf.to_float(self.batch_size)
        return loss

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.cost)

    def word_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.nxtwd, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
