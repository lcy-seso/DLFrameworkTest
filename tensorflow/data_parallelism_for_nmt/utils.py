#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.python.client import device_lib

__all__ = [
    "get_available_gpus",
    "create_hparams",
    "add_arguments",
]


def get_available_gpus():
    """Returns a list of available GPU devices names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--src_file_name", type=str, default="data/train.en", help="")
    parser.add_argument(
        "--tgt_file_name", type=str, default="data/train.de", help="")
    parser.add_argument(
        "--src_vocab_file", type=str, default="data/vocab.50K.en", help="")
    parser.add_argument(
        "--tgt_vocab_file", type=str, default="data/vocab.50K.de", help="")
    parser.add_argument("--src_vocab_size", type=int, default=50000, help="")
    parser.add_argument("--tgt_vocab_size", type=int, default=50000, help="")
    parser.add_argument("--bos", type=str, default="<s>", help="")
    parser.add_argument("--eos", type=str, default="</s>", help="")
    parser.add_argument("--unk_id", type=int, default=0, help="")
    parser.add_argument("--src_max_len", type=int, default=None, help="")
    parser.add_argument("--tgt_max_len", type=int, default=None, help="")
    parser.add_argument("--num_parallel_calls", type=int, default=4, help="")
    parser.add_argument("--num_buckets", type=int, default=5, help="")
    parser.add_argument(
        "--output_buffer_size", type=int, default=None, help="")
    parser.add_argument("--disable_shuffle", type=bool, default=False, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")

    parser.add_argument("--time_major", type=bool, default=False, help="")
    parser.add_argument("--dropout", type=float, default=0., help="")
    parser.add_argument("--unit_type", type=str, default="lstm", help="")
    parser.add_argument("--num_units", type=int, default=512, help="")
    parser.add_argument("--forget_bias", type=int, default=1., help="")
    parser.add_argument("--embedding_dim", type=int, default=512, help="")
    parser.add_argument("--encoder_type", type=str, default="bi", help="")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="")
    parser.add_argument("--optimizer", type=str, default="adam", help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--num_keep_ckpts", type=int, default=5, help="")
    parser.add_argument("--max_gradient_norm", type=float, default=5., help="")

    parser.add_argument(
        "--variable_update", type=str, default="parameter_server", help="")
    parser.add_argument(
        "--param_server_device", type=str, default="gpu", help="")
    parser.add_argument(
        "--local_parameter_device", type=str, default="gpu", help="")

    parser.add_argument(
        "--variable_consistency", type=str, default="strong", help="")
    parser.add_argument("--gradient_repacking", type=int, default=4, help="")
    parser.add_argument("--all_reduce_spec", type=str, default="nccl", help="")
    parser.add_argument(
        "--agg_small_grads_max_bytes", type=int, default=0, help="")
    parser.add_argument(
        "--agg_small_grads_max_group", type=int, default=10, help="")

    parser.add_argument("--enable_profile", type=bool, default=False, help="")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        src_file_name=flags.src_file_name,
        tgt_file_name=flags.tgt_file_name,
        src_vocab_file=flags.src_vocab_file,
        tgt_vocab_file=flags.tgt_vocab_file,
        src_vocab_size=flags.src_vocab_size,
        tgt_vocab_size=flags.tgt_vocab_size,
        bos=flags.bos,
        eos=flags.eos,
        unk_id=flags.unk_id,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        num_parallel_calls=flags.num_parallel_calls,
        num_buckets=flags.num_buckets,
        output_buffer_size=flags.output_buffer_size,
        disable_shuffle=flags.disable_shuffle,
        # when using multi-gpu cards, this means bath size per card.
        batch_size=flags.batch_size,

        # hyper parameters for model topology
        time_major=flags.time_major,
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        num_units=flags.num_units,
        forget_bias=flags.forget_bias,
        embedding_dim=flags.embedding_dim,
        encoder_type=flags.encoder_type,
        num_encoder_layers=flags.num_encoder_layers,
        # TODO: The current implementation requries encoder and decoder has
        # the same number RNN cells.
        num_decoder_layers=flags.num_decoder_layers,
        optimizer=flags.optimizer,
        learning_rate=flags.learning_rate,
        num_keep_ckpts=flags.num_keep_ckpts,
        max_gradient_norm=flags.max_gradient_norm,

        # parameter server places
        variable_update=flags.variable_update,
        param_server_device=flags.param_server_device,
        local_parameter_device=flags.local_parameter_device,

        # used for all reduce algorithm
        variable_consistency=flags.variable_consistency,
        gradient_repacking=flags.gradient_repacking,
        all_reduce_spec=flags.all_reduce_spec,
        agg_small_grads_max_bytes=flags.agg_small_grads_max_bytes,
        agg_small_grads_max_group=flags.agg_small_grads_max_group,
        enable_profile=flags.enable_profile, )
