#!/usr/bin/env python
#coding=utf-8
import time
import pdb

import tensorflow as tf
from tensorflow.python.client import timeline

from seq2seq_model import Seq2SeqModel, hparams
from utils import get_available_gpus

ENABLE_PROFILE = False


def make_config():
    config = tf.ConfigProto()

    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 0
    return config


def train():
    num_gpus = len(get_available_gpus())
    print("num_gpus = %d, batch size = %d" % (num_gpus,
                                              hparams.batch_size * num_gpus))

    model = Seq2SeqModel(num_gpus, hparams)
    config = make_config()

    options = tf.RunOptions(
        trace_level=tf.RunOptions.FULL_TRACE) if ENABLE_PROFILE else None
    run_metadata = tf.RunMetadata() if ENABLE_PROFILE else None

    sv = tf.train.Supervisor(
        is_chief=True,
        logdir="train_log",
        ready_for_local_init_op=None,
        local_init_op=model.local_var_init_op_group,
        saver=model.saver,
        global_step=model.global_step,
        summary_op=None,
        save_model_secs=600,
        summary_writer=None)
    with sv.managed_session(
            master="", config=config, start_standard_services=False) as sess:

        pass_id = 0
        batch_id = 0

        start_time = time.time()
        total_word_count = 0
        sess.run(model.iterator.initializer)

        while True:
            try:
                _, loss, word_count = sess.run(
                    list(model.fetches.values()) + [model.word_count],
                    options=options,
                    run_metadata=run_metadata)

                total_word_count += word_count

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d, Loss : %.5f" % (pass_id,
                                                              batch_id, loss))
                batch_id += 1

                if ENABLE_PROFILE and batch_id == 4:
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    )
                    with open("profiling_log/nmt_%02d_cards_timeline.json" %
                              (num_gpus), "w") as f:
                        f.write(chrome_trace)
                    break

                if batch_id == 50:
                    time_elapsed = time.time() - start_time
                    print("total time : %.3f, speed : %.3f (w/s)" %
                          (time_elapsed, total_word_count / time_elapsed))
                    break

            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
