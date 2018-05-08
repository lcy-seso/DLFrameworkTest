#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf

from iterator_helper import get_iterator
from seq2seq_model import Seq2SeqModel, hparams


def train():
    iterator = get_iterator(hparams.src_file_name, hparams.tgt_file_name,
                            hparams.src_vocab_file, hparams.tgt_vocab_file,
                            hparams.batch_size)
    model = Seq2SeqModel(iterator, hparams)

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

        while True:
            try:
                _, loss = sess.run([model.update, model.loss])

                if not batch_id % 10:
                    print("Pass %d, Batch %d, Loss : %.5f" % (pass_id,
                                                              batch_id, loss))
                batch_id += 1

            except tf.errors.OutOfRangeError:
                sess.run(initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
