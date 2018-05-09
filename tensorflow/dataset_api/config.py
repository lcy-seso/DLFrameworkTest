#!/usr/bin/env python
#coding=utf-8


class LMConfig(object):
    """Configuration of language model"""
    batch_size = 200 * 1
    time_major = False

    train_data_path = "data/ptb.train.txt"
    vocab_file_path = "data/vocab.txt"
    vocab_size = 10001
    unk_id = 1

    embedding_dim = 32
    hidden_dim = 128
    num_layers = 2

    learning_rate = 1e-3

    num_passes = 5
