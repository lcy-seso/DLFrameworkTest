from __future__ import print_function

import pdb
import tensorflow as tf

import variable_mgr_util


class VariableMgr(object):
    """Abstract superclass for class used by BenchmarkCNN to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
    """

    def __init__(self, model_helper):
        self.model_helper = model_helper
        self.staging_delta_ops = []
        self.use_resource_vars = False

        # A variable for automatic loss scaling.
        self.grad_has_inf_nan = None

    def each_tower_has_variables(self):
        """Returns True if each GPU tower of the model has separate variables."""
        assert False, "Must be implemented in subclass"

    def supports_staged_vars(self):
        """Whether staged variable management is supported."""
        return False

    def create_outer_variable_scope(self, device_num):
        """Create the tf.variable_scope around all model graph operations."""
        del device_num  # unused by this implementation
        assert False, "Must be implemented in subclass"

    def preprocess_device_grads(self, device_grads):
        """Preprocess the device gradients prior to applying them.

        Args:
          device_grads: List of lists of (gradient, variable) tuples.
            device_grads[t][g] = (gradient, variable), where t is the index of the
            tower and g is the index of the gradient-variable pair.

        Returns: a tuple of (apply_gradients_devices, gradient_state).
          gradient_state is an opaque structure that should be passed to
          get_gradients_to_apply() and append_apply_gradients_ops() (in that order).
          apply_gradients_devices is a list of devices where the gradients will be
          applied with get_gradients_to_apply() and append_apply_gradients_ops().
        """
        del device_grads  # unused by this implementation
        assert False, "Must be implemented in subclass"

    def get_gradients_to_apply(self, device_num, gradient_state):
        """Returns the [(gradient, variable)] list to apply for device_num.

        Args:
          device_num: indexes into apply_gradients_devices, which was returned by an
            earlier call to preprocess_device_grads.
          gradient_state: from previous call to apply_gradients_devices.
        """
        del device_num, gradient_state  # unused by this implementation
        assert False, "Must be implemented in subclass"

    def append_apply_gradients_ops(self, gradient_state, opt, grads,
                                   training_ops, loss_scale_params):
        """Adds training ops for grads to "training_ops".

        Args:
          gradient_state: from previous call to apply_gradients_devices.
          opt: the underlying optimizer
          grads: [(grad, var)] to apply
          training_ops: list to which to add ops
        """
        del gradient_state  # unused by this implementation

        def get_apply_gradients_ops_func():
            """Returns the apply_gradients op."""
            return [opt.apply_gradients(grads)]

        variable_mgr_util.append_gradients_with_loss_scale(
            training_ops, get_apply_gradients_ops_func, loss_scale_params,
            self.grad_has_inf_nan)

    def get_post_init_ops(self):
        """Returns ops that should run post-initialization."""
        return []

    def get_devices(self):
        """Returns devices to use for computation; includes replica selection."""
        assert False, "Must be implemented in subclass"

    def savable_variables(self):
        """Returns a list/dict of savable variables to pass to tf.train.Saver."""
        return tf.global_variables()

    def trainable_variables_on_device(self,
                                      rel_device_num,
                                      abs_device_num,
                                      writable=False):
        """Return the set of trainable variables on device.

        Args:
          rel_device_num: local worker device index.
          abs_device_num: global graph device index.
          writable: whether to get a reference to the underlying variable.

        Returns:
          The set of trainable variables on the specified device.
        """
        del rel_device_num, writable
        if self.each_tower_has_variables():
            params = [
                v for v in tf.trainable_variables()
                if v.name.startswith("v%s/" % abs_device_num)
            ]
        else:
            params = tf.trainable_variables()
        return params


class VariableMgrLocalFetchFromPS(VariableMgr):
    """VariableMgr that implements the --parameter_server mode for local jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
    """

    def each_tower_has_variables(self):
        return False

    def create_outer_variable_scope(self, device_num):
        return tf.variable_scope(
            "v", reuse=bool(device_num), use_resource=self.use_resource_vars)

    def preprocess_device_grads(self, device_grads):
        return ([self.model_helper.param_server_device], device_grads)

    def get_gradients_to_apply(self, gradient_state):
        device_grads = gradient_state
        agg_grads, self.grad_has_inf_nan = (
            variable_mgr_util.
            aggregate_gradients_using_copy_with_variable_colocation(
                device_grads, use_mean=True, check_inf_nan=False))
        return agg_grads

    def get_devices(self):
        raw_devices = self.model_helper.raw_devices
        if self.model_helper.local_parameter_device == "gpu":
            return [
                variable_mgr_util.ParamServerDeviceSetter(
                    d,
                    raw_devices, ) for d in raw_devices
            ]
        else:
            return [
                tf.train.replica_device_setter(
                    worker_device=d,
                    ps_device=self.model_helper.param_server_device,
                    ps_tasks=1) for d in raw_devices
            ]
