import copy
import numpy as np

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class DummyObjWrapper:
    def __init__(self, _obj):
        # DummyObjWrapper is used when passing a list of objects as inputs into ddp model's forward function
        # This wrapper is necessary because otherwise ddp will apply _recursive_to to the whole list of inputs,
        # resulting in a very high overhead (1ms * list length)
        self.obj = _obj
