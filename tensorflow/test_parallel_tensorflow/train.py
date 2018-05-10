#!/usr/bin/env python
#coding=utf-8
import pdb
import time
import tensorflow as tf

from iterator_helper import get_iterator
from seq2seq_model import Seq2SeqModel, hparams

from utils import get_available_gpus


def train():
    gpu_num = len(get_available_gpus())
    iterator = get_iterator(
        src_file_name=hparams.src_file_name,
        tgt_file_name=hparams.tgt_file_name,
        src_vocab_file=hparams.src_vocab_file,
        tgt_vocab_file=hparams.tgt_vocab_file,
        batch_size=(hparams.batch_size * gpu_num
                    if gpu_num > 1 else hparams.batch_size),
        disable_shuffle=True)

    model = Seq2SeqModel(gpu_num, iterator, hparams)

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)

        pass_id = 0
        batch_id = 0

        start_time = time.time()
        while True:
            try:
                _, loss, bs, src_len = sess.run([
                    model.update, model.train_loss, model.batch_size,
                    model.iterator.source_sequence_length
                ])

                if not batch_id % 10:
                    print("Pass %d, Batch %d, Loss : %.5f" % (pass_id,
                                                              batch_id, loss))
                batch_id += 1

                if batch_id == 50:
                    print("time to run 50 batches : %f" %
                          (time.time() - start_time))

            except tf.errors.OutOfRangeError:
                sess.run(initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
