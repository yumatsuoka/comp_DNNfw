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
from tf_pure_trainer import StandardTrainer, Multigpu_trainer 

# ### Hyper parameters
NUM_GPU = 1
EPOCH = 3
BATCH_SIZE = 100
MODEL_TYPE = "allconvnet"
DATASET_TYPE = "cifar10"
print("epoch:{}, batch:{}, dataset:{}".format(EPOCH, BATCH_SIZE, DATASET_TYPE))

# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()
train_data = {'data':dataset['train']['data']/255., 'target':dataset['train']['target']}
val_data = {'data':dataset['test']['data']/255., 'target':dataset['test']['target']}
train_size, image_size, _, num_channels = train_data['data'].shape
num_labels  = len(list(set(dataset['train']['target'])))
print("# train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))

## training
model = ResNet if MODEL_TYPE == 'resnet' else AllConvNetBN
if NUM_GPU <= 1:
    trainer = StandardTrainer(model(n_class=num_labels), train_data, val_data, BATCH_SIZE)
else:
    trainer = Multigpu_trainer(model(n_class=num_labels), train_data, val_data, BATCH_SIZE, num_gpus=NUM_GPU)
trainer.fit(epoch=EPOCH)
# trainer.test(test_data)
