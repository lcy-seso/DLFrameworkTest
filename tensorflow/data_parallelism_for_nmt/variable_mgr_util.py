from __future__ import print_function

import collections as pycoll
import operator

import pdb
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops

AutoLossScaleParams = pycoll.namedtuple(
    "AutoLossScaleParams",
    [
        # If true, enable automatic loss scaling.
        "enable_auto_loss_scale",
        # The value to scale the loss before computing gradients.
        "loss_scale",
        # Number of normal steps with the current `loss_scale`.
        "loss_scale_normal_steps",
        # Increase loss scale every n steps.
        "inc_loss_scale_every_n",
        # If true, the current worker is chief. The current implementation
        # relies on the chief to update loss_scale value, but in future, we
        # might change this to ask the parameter server to update loss_scales
        # for better performance.
        "is_chief",
    ])


def append_gradients_with_loss_scale(training_ops,
                                     get_apply_gradients_ops_func,
                                     loss_scale_params, grad_has_inf_nan):
    """Selectively appends gradients update ops with loss scaling.

    Args:
      training_ops: a list of training ops to be executed.
      get_apply_gradients_ops_func: a function that returns a list of ops for
        applying gradients. Here, we must pass a function instead of the actual
        list of ops; otherwise, those ops would be executed unconditionally due to
        the semantics of tf.cond.
      loss_scale_params: An AutoLossScaleParams tuple.
      grad_has_inf_nan: Boolean tensor indicating whether the gradients have infs
        or nans.
    """
    is_chief = loss_scale_params.is_chief
    loss_scale = loss_scale_params.loss_scale
    loss_scale_normal_steps = loss_scale_params.loss_scale_normal_steps
    inc_loss_scale_every_n = loss_scale_params.inc_loss_scale_every_n
    enable_auto_loss_scale = loss_scale_params.enable_auto_loss_scale

    if loss_scale is None or not enable_auto_loss_scale or not is_chief:
        update_ops = get_apply_gradients_ops_func()
        training_ops.extend(update_ops)
    else:
        # If nans/infs occurred, skip applying gradients and instead update
        # loss_scale (halve loss_scale and reset loss_scale_normal_steps to zero).
        def update_op_if_nan_or_inf():
            """Update loss_scale and discard gradients if nans/infs occurred."""
            return tf.group(
                tf.assign(loss_scale, loss_scale / 2.),
                tf.assign(loss_scale_normal_steps, 0))

        # Otherwise, apply gradients, and update loss_scale and
        # loss_scale_normal_steps.
        def update_op_if_no_nan_or_inf():
            """Apply gradients, and update loss scaling."""
            return tf.group(
                get_loss_scale_update_op(loss_scale, loss_scale_normal_steps,
                                         inc_loss_scale_every_n),
                *get_apply_gradients_ops_func())

        assert grad_has_inf_nan is not None
        update_op = tf.cond(
            grad_has_inf_nan,
            update_op_if_nan_or_inf,
            update_op_if_no_nan_or_inf, )
        training_ops.append(update_op)


class ParamServerDeviceSetter(object):
    """Helper class to assign variables on the least loaded ps-device."""

    def __init__(self, worker_device, ps_devices):
        """Initializer for ParamServerDevicSetter.

        Args:
          worker_device: the device to use for computer ops.
          ps_devices: a list of device to use for Variable ops. Each variable is
          assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device

        if op.type not in ["Variable", "VariableV2"]:
            return self.worker_device

        device_index, _ = min(enumerate(self.ps_sizes),
                              key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]

        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


def aggregate_single_gradient_using_copy(grad_and_vars, use_mean,
                                         check_inf_nan):
    """Calculate the average gradient for a shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
        (gradient, variable) pair within the outer list represents the gradient
        of the variable calculated for a single tower, and the number of pairs
        equals the number of towers.
      use_mean: if True, mean is taken, else sum of gradients is taken.
      check_inf_nan: check grads for nans and infs.

    Returns:
      The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
        gradient has been averaged across all towers. The variable is chosen from
        the first tower. The has_nan_or_inf indicates the grads has nan or inf.
    """

    grads = []

    is_indexed_slices = None

    indices = []
    values = []

    lookup_table_grad = None
    for g, _ in grad_and_vars:
        if isinstance(g, ops.IndexedSlices):
            is_indexed_slices = True

            if len(grad_and_vars) == 1:
                # no need to merge gradiens when there is only one GPU device
                lookup_table_grad = g
            else:
                values.append(g.values)
                indices.append(g.indices)
        else:
            grads.append(g)

    with tf.name_scope("merge_gradients") as name_scope:
        if is_indexed_slices is not None:
            if lookup_table_grad is None:
                grad = tf.IndexedSlices(
                    values=tf.concat(values, axis=0),
                    indices=tf.concat(indices, axis=0))
            else:
                grad = lookup_table_grad
        else:
            grad = tf.add_n(grads)

        if use_mean and len(grads) > 1:
            grad = tf.multiply(grad, 1.0 / len(grads))

    v = grad_and_vars[0][1]
    if check_inf_nan:
        has_nan_or_inf = tf.logical_not(tf.reduce_all(tf.is_finite(grads)))
        return (grad, v), has_nan_or_inf
    else:
        return (grad, v), None


def aggregate_gradients_using_copy_with_variable_colocation(
        tower_grads, use_mean, check_inf_nan):
    """Aggregate gradients, colocating computation with the gradient"s variable.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over towers. The inner list is over individual gradients. All variables
        of the same gradient across towers must be the same (that is,
        tower_grads[x][a][1] == tower_grads[y][a][1] for all indices x, y, and a)
      use_mean: if True, mean is taken, else sum of gradients is taken.
      check_inf_nan: If true, check grads for nans and infs.

    Returns:
      The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
        gradient has been averaged across all towers. The variable is chosen from
        the first tower. The has_nan_or_inf indicates the grads has nan or inf.
    """
    agg_grads = []
    has_nan_or_inf_list = []

    for single_grads in zip(*tower_grads):
        # Note that each single_grads looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        var = single_grads[0][1]

        for _, v in single_grads:
            assert v == var

        with tf.device(var.device):
            (grad_and_var,
             has_nan_or_inf) = aggregate_single_gradient_using_copy(
                 single_grads, use_mean, check_inf_nan)
            agg_grads.append(grad_and_var)
            has_nan_or_inf_list.append(has_nan_or_inf)

    if check_inf_nan:
        return agg_grads, tf.reduce_any(has_nan_or_inf_list)
    else:
        return agg_grads, None
