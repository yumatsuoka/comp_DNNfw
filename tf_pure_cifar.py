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
from tf_pure_model import AllConvNetBN
from tf_pure_trainer import Trainer

# ### Hyper parameters

EPOCH = 20 
BATCH_SIZE = 100
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
model = AllConvNetBN(n_class=num_labels)
trainer = Trainer(model, dataset=dataset, batchsize=BATCH_SIZE, epoch=EPOCH)
trainer.fit()
