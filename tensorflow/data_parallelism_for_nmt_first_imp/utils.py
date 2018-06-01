#!/usr/bin/env python
#coding=utf-8
from tensorflow.python.client import device_lib


def get_available_gpus():
    """Returns a list of available GPU devices names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
