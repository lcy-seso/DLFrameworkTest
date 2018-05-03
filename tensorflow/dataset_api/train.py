#!/usr/bin/env python
#coding=utf-8

import time

import tensorflow as tf
from tensorflow.python.client import timeline

from rnnlm import LMConfig, RNNLM
from dataset_api_example import get_dataset
from timeline_utils import TimeLiner

PROFILE = False


def train():
    model_config = LMConfig()
    initializer, curwd, nxtwd, nxtwd_len = get_dataset(
        model_config.train_data_path, model_config.vocab_file_path,
        model_config.batch_size)

    model = RNNLM(model_config, curwd, nxtwd, nxtwd_len)

    options = None
    run_metadata = None
    if PROFILE:
        """
        This is for profiling only. Do not use this in normal training
        because it harms the time performance.
        """

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(initializer)

        pass_id = 0
        batch_id = 0

        while True:
            try:
                start_time = time.time()

                cost, _ = sess.run(
                    [model.cost, model.optim],
                    options=options,
                    run_metadata=run_metadata)

                if PROFILE:
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = \
                            fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.update_timeline(chrome_trace)

                batch_id += 1
                print("Pass %d, Batch %d, Loss %.4f" % (pass_id, batch_id,
                                                        cost))
                if PROFILE and batch_id == 3:
                    many_runs_timeline.save("rnn_lm_timeline.json")
                    break

            except tf.errors.OutOfRangeError:
                if pass_id == 3:
                    elapsed = time.time() - start_time
                    print "Time to train Three epoch: %.6f" % (elapsed)

                pass_id += 1
                if pass_id == model_config.num_passes: break
                sess.run(initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
