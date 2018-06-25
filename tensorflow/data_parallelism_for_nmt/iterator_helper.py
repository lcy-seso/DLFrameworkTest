#!/usr/bin/env python
#coding=utf-8

import collections
import pdb

import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.data.python.ops import prefetching_ops

__all__ = [
    "get_iterator",
    "build_prefetch_processing",
]


class BatchedInput(
        collections.namedtuple("BatchedInput", (
            "initializer", "source", "target_input", "target_output",
            "source_sequence_length", "target_sequence_length", ))):
    pass


def get_synthetic_data(seq_len, batch_size, time_major, src_vocab_size,
                       tgt_vocab_size, devices):
    def __gen_one_part(seq_len, batch_size, time_major, src_vocab_size,
                       tgt_vocab_size, device):
        with tf.device(device):
            input_shape = ([seq_len, batch_size]
                           if time_major else [batch_size, seq_len])

            src_ids = tf.random_uniform(
                input_shape,
                minval=0,
                maxval=src_vocab_size - 1,
                dtype=tf.int32,
                seed=None,
                name="src")

            tgt_input_ids = tf.random_uniform(
                input_shape,
                minval=0,
                maxval=tgt_vocab_size - 1,
                dtype=tf.int32,
                seed=None,
                name="tgt_input")

            tgt_output_ids = tf.random_uniform(
                input_shape,
                minval=0,
                maxval=tgt_vocab_size - 1,
                dtype=tf.int32,
                seed=None,
                name="tgt_output")

            len_shape = [batch_size]
            src_seq_len = tf.random_uniform(
                len_shape,
                minval=seq_len,
                maxval=seq_len + 1,
                dtype=tf.int32,
                seed=None,
                name="src_len")
            tgt_seq_len = tf.random_uniform(
                len_shape,
                minval=seq_len,
                maxval=seq_len + 1,
                dtype=tf.int32,
                seed=None,
                name="src_len")

            return (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
                    tgt_seq_len)

    num_splits = len(devices)
    src_ids = [[] for _ in range(num_splits)]
    tgt_input_ids = [[] for _ in range(num_splits)]
    tgt_output_ids = [[] for _ in range(num_splits)]
    src_seq_len = [[] for _ in range(num_splits)]
    tgt_seq_len = [[] for _ in range(num_splits)]

    for i, device in enumerate(devices):
        (src_ids[i], tgt_input_ids[i], tgt_output_ids[i], src_seq_len[i],
         tgt_seq_len[i]) = __gen_one_part(seq_len, batch_size, time_major,
                                          src_vocab_size, tgt_vocab_size,
                                          device)

    return BatchedInput(
        initializer=None,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


def get_iterator(src_file_name,
                 tgt_file_name,
                 src_vocab_file,
                 tgt_vocab_file,
                 batch_size,
                 bos="<s>",
                 eos="</s>",
                 unk_id=0,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 num_buckets=5,
                 output_buffer_size=None,
                 disable_shuffle=False,
                 num_splits=1):
    def __get_word_dict(vocab_file_path, unk_id):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file_path,
            key_column_index=0,
            default_value=unk_id)

    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    with tf.name_scope("batch_processing"):
        src_vocab_table = __get_word_dict(src_vocab_file, unk_id)
        tgt_vocab_table = __get_word_dict(tgt_vocab_file, unk_id)

        src_eos_id = tf.cast(
            src_vocab_table.lookup(tf.constant(eos)), tf.int32)
        tgt_bos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(bos)), tf.int32)
        tgt_eos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

        src_dataset = tf.data.TextLineDataset(src_file_name)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_name)

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        if not disable_shuffle:
            dataset = dataset.shuffle(
                buffer_size=output_buffer_size, reshuffle_each_iteration=True)

        src_tgt_dataset = dataset.map(
            lambda src, tgt: (tf.string_split([src]).values, \
                    tf.string_split([tgt]).values),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        # Filter zero length input sequences.
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

        if src_max_len:
            src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src[:src_max_len], tgt),
                num_parallel_calls=num_parallel_calls).prefetch(
                    output_buffer_size)
        if tgt_max_len:
            src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src, tgt[:tgt_max_len]),
                num_parallel_calls=num_parallel_calls).prefetch(
                    output_buffer_size)

        # convert word string to word index
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (
                   tf.cast(src_vocab_table.lookup(src), tf.int32),
                    tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src,
                              tf.concat(([tgt_bos_id], tgt), 0),
                              tf.concat((tgt, [tgt_eos_id]), 0)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        # Add in sequence lengths.
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out: (\
                    src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        def __batching_func(x):
            return x.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt_in
                    tf.TensorShape([None]),  # tgt_out
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # tgt_len
                ),
                padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, 0, 0))

        if num_buckets > 1:

            def __key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                if tgt_max_len:
                    bucket_width = (
                        tgt_max_len + num_buckets - 1) // num_buckets
                else:
                    bucket_width = 10

                bucket_id = tgt_len // bucket_width
                return tf.to_int64(tf.minimum(num_buckets, bucket_id))

            def __reduce_func(unused_key, windowed_data):
                return __batching_func(windowed_data)

            batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=__key_func,
                    reduce_func=__reduce_func,
                    window_size=batch_size))

        else:
            batched_dataset = __batching_func(curwd_nxtwd_dataset)

        src_ids = [[] for _ in range(num_splits)]
        tgt_input_ids = [[] for _ in range(num_splits)]
        tgt_output_ids = [[] for _ in range(num_splits)]
        src_seq_len = [[] for _ in range(num_splits)]
        tgt_seq_len = [[] for _ in range(num_splits)]

        batched_iter = batched_dataset.make_initializable_iterator()
        for i in range(num_splits):
            (src_ids[i], tgt_input_ids[i], tgt_output_ids[i], src_seq_len[i],
             tgt_seq_len[i]) = batched_iter.get_next()

        return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            target_input=tgt_input_ids,
            target_output=tgt_output_ids,
            source_sequence_length=src_seq_len,
            target_sequence_length=tgt_seq_len)


def create_iterator(src_file_name,
                    tgt_file_name,
                    src_vocab_file,
                    tgt_vocab_file,
                    batch_size,
                    bos="<s>",
                    eos="</s>",
                    unk_id=0,
                    src_max_len=None,
                    tgt_max_len=None,
                    num_parallel_calls=4,
                    num_buckets=5,
                    output_buffer_size=None,
                    disable_shuffle=False,
                    num_splits=1):
    def __get_word_dict(vocab_file_path, unk_id):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file_path,
            key_column_index=0,
            default_value=unk_id)

    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    with tf.name_scope("batch_processing"):
        src_vocab_table = __get_word_dict(src_vocab_file, unk_id)
        tgt_vocab_table = __get_word_dict(tgt_vocab_file, unk_id)

        src_eos_id = tf.cast(
            src_vocab_table.lookup(tf.constant(eos)), tf.int32)
        tgt_bos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(bos)), tf.int32)
        tgt_eos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

        src_dataset = tf.data.TextLineDataset(src_file_name)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_name)

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        if not disable_shuffle:
            dataset = dataset.shuffle(
                buffer_size=output_buffer_size, reshuffle_each_iteration=True)

        src_tgt_dataset = dataset.map(
            lambda src, tgt: (tf.string_split([src]).values, \
                    tf.string_split([tgt]).values),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        # Filter zero length input sequences.
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

        if src_max_len:
            src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src[:src_max_len], tgt),
                num_parallel_calls=num_parallel_calls).prefetch(
                    output_buffer_size)
        if tgt_max_len:
            src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src, tgt[:tgt_max_len]),
                num_parallel_calls=num_parallel_calls).prefetch(
                    output_buffer_size)

        # convert word string to word index
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (
                   tf.cast(src_vocab_table.lookup(src), tf.int32),
                    tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src,
                              tf.concat(([tgt_bos_id], tgt), 0),
                              tf.concat((tgt, [tgt_eos_id]), 0)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        # Add in sequence lengths.
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out: (\
                    src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        def __batching_func(x):
            return x.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt_in
                    tf.TensorShape([None]),  # tgt_out
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # tgt_len
                ),
                padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, 0, 0))

        if num_buckets > 1:

            def __key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                if tgt_max_len:
                    bucket_width = (
                        tgt_max_len + num_buckets - 1) // num_buckets
                else:
                    bucket_width = 10

                bucket_id = tgt_len // bucket_width
                return tf.to_int64(tf.minimum(num_buckets, bucket_id))

            def __reduce_func(unused_key, windowed_data):
                return __batching_func(windowed_data)

            batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=__key_func,
                    reduce_func=__reduce_func,
                    window_size=batch_size))

        else:
            batched_dataset = __batching_func(curwd_nxtwd_dataset)

        batched_iter = batched_dataset.make_initializable_iterator()
        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
        #                      batched_iter.initializer)
        return batched_iter


def minibatch_fn(src_file_name,
                 tgt_file_name,
                 src_vocab_file,
                 tgt_vocab_file,
                 batch_size,
                 bos="<s>",
                 eos="</s>",
                 unk_id=0,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 num_buckets=5,
                 output_buffer_size=None,
                 disable_shuffle=False,
                 num_splits=1):
    iterator = create_iterator(src_file_name, tgt_file_name, src_vocab_file,
                               tgt_vocab_file, batch_size, bos, eos, unk_id,
                               src_max_len, tgt_max_len, num_parallel_calls,
                               num_buckets, output_buffer_size,
                               disable_shuffle, num_splits)
    iterator_string_handle = iterator.string_handle()

    @function.Defun(tf.string)
    def _fn(h):
        remote_iterator = tf.data.Iterator.from_string_handle(
            h, iterator.output_types, iterator.output_shapes)
        (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
         tgt_seq_len) = remote_iterator.get_next()
        return (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
                tgt_seq_len)

    return _fn, [iterator_string_handle], iterator.initializer


def build_prefetch_processing(cpu_device,
                              gpu_devices,
                              src_file_name,
                              tgt_file_name,
                              src_vocab_file,
                              tgt_vocab_file,
                              batch_size,
                              bos="<s>",
                              eos="</s>",
                              unk_id=0,
                              src_max_len=None,
                              tgt_max_len=None,
                              num_parallel_calls=4,
                              num_buckets=5,
                              output_buffer_size=None,
                              disable_shuffle=False,
                              num_splits=1):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    function_buffering_resources = []

    remote_fn, args, initializer = minibatch_fn(
        src_file_name, tgt_file_name, src_vocab_file, tgt_vocab_file,
        batch_size, bos, eos, unk_id, src_max_len, tgt_max_len,
        num_parallel_calls, num_buckets, output_buffer_size, disable_shuffle,
        num_splits)

    for device_num, device in enumerate(gpu_devices):
        buffer_resource_handle = prefetching_ops.function_buffering_resource(
            f=remote_fn,
            target_device=cpu_device,
            string_arg=args[0],
            buffer_size=output_buffer_size,
            shared_name=None)
        function_buffering_resources.append(buffer_resource_handle)
    return function_buffering_resources, initializer


def get_input_data(function_buffering_resource):
    # there are 5 outputs in the model whose types are all tf.int32
    return prefetching_ops.function_buffering_resource_get_next(
        function_buffer_resource=function_buffering_resource,
        output_types=[tf.int32] * 5)


def test():
    use_synthetic_data = True

    src_file_name = "data/train.en"
    tgt_file_name = "data/train.de"
    src_vocab_file = "data/vocab.50K.en"
    tgt_vocab_file = "data/vocab.50K.de"

    seq_len = 10
    src_vocab_size = 100
    tgt_vocab_size = 100
    time_major = True

    batch_size = 3 * 5
    num_splits = 3

    if use_synthetic_data:
        iterator = get_synthetic_data(seq_len, batch_size, time_major,
                                      src_vocab_size, tgt_vocab_size,
                                      num_splits)
    else:
        iterator = get_iterator(
            src_file_name,
            tgt_file_name,
            src_vocab_file,
            tgt_vocab_file,
            batch_size,
            num_splits=num_splits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if not use_synthetic_data:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

        for i in range(5):
            res = sess.run(iterator.source + iterator.target_input + iterator.
                           target_output + iterator.source_sequence_length +
                           iterator.target_sequence_length)


def test_build_prefetch():
    src_file_name = "data/train.en"
    tgt_file_name = "data/train.de"
    src_vocab_file = "data/vocab.50K.en"
    tgt_vocab_file = "data/vocab.50K.de"

    seq_len = 10
    src_vocab_size = 100
    tgt_vocab_size = 100
    time_major = True

    num_splits = 2
    batch_size = num_splits * 5

    gpu_devices = ["/gpu:%d" % (i) for i in range(num_splits)]
    cpu_device = "/cpu:0"

    function_buffering_resources, initializer = build_prefetch_processing(
        cpu_device,
        gpu_devices,
        src_file_name,
        tgt_file_name,
        src_vocab_file,
        tgt_vocab_file,
        batch_size,
        num_splits=num_splits)

    ops = []
    for device_num, device in enumerate(gpu_devices):
        with tf.device(device):
            ops.append(
                get_input_data(function_buffering_resources[device_num]))

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(initializer)

        for i in range(2):
            res = sess.run(ops)


if __name__ == "__main__":
    test_build_prefetch()
