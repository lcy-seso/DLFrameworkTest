#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf

__all__ = [
    "get_initializer",
    "gradient_clip",
]


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    return clipped_gradients, gradient_norm
