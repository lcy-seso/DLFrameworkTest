#!/usr/bin/env python
#coding=utf-8
import time

import tensorflow as tf

from iterator_helper import get_iterator
from seq2seq_model import Seq2SeqModel, hparams

from utils import get_available_gpus


def train():
    gpu_num = len(get_available_gpus())
    batch_size = (hparams.batch_size * gpu_num
                  if gpu_num > 1 else hparams.batch_size)
    print("batch size = %d" % (batch_size))

    iterator = get_iterator(
        src_file_name=hparams.src_file_name,
        tgt_file_name=hparams.tgt_file_name,
        src_vocab_file=hparams.src_vocab_file,
        tgt_vocab_file=hparams.tgt_vocab_file,
        batch_size=batch_size,
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
        total_word_count = 0

        while True:
            try:
                _, loss, word_count = sess.run(
                    [model.update, model.train_loss, model.word_count])
                total_word_count += word_count

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d, Loss : %.5f" % (pass_id,
                                                              batch_id, loss))
                batch_id += 1

                if batch_id == 100:
                    time_elapsed = time.time() - start_time
                    print("total time : %.3f, speed : %.3f (w/s)" %
                          (time_elapsed, total_word_count / time_elapsed))
                    break

            except tf.errors.OutOfRangeError:
                sess.run(initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
