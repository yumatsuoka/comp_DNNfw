# coding: utf-8
#!/usr/bin/env python

# python2, python3の互換性のためのおまじない
from __future__ import print_function

import six
import numpy
import tensorflow as tf
from sklearn.model_selection import KFold

import cifar10
import cifar100
from tf_pure_model import AllConvNetBN, ResNet
from tf_pure_trainer import Trainer

# ### Hyper parameters
NUM_GPU = 1
EPOCH = 10 
BATCH_SIZE = 100
MODEL_TYPE = "allconvnet"
DATASET_TYPE = "cifar10"
print("epoch:{}, batch:{}, dataset:{}".format(EPOCH, BATCH_SIZE, DATASET_TYPE))

# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()
train_size, image_size, _, num_channels = dataset['train']['data'].shape
num_labels  = len(list(set(dataset['train']['target'])))
print("# train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))

## training 
model = ResNet if MODEL_TYPE == 'resnet' else AllConvNetBN
trainer = Trainer(model(n_class=num_labels), dataset=dataset,\
        batchsize=BATCH_SIZE, epoch=EPOCH, num_gpu=NUM_GPU)
trainer.fit()
