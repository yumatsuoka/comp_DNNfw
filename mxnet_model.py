# coding: utf-8
#!/usr/bin/env python

from __future__ import print_function

import logging
logging.basicConfig(level=logging.DEBUG)
#logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
import mxnet as mx

import cifar10
import cifar100
from mxnet_model import AllConvNetBN, ResNet


def allConvNetBN():
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=32)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
