# coding: utf-8
#!/usr/bin/env python

from __future__ import print_function

import logging
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
import mxnet as mx

import cifar10
import cifar100
from mxnet_model import resnet, allconvnet, shallownet


# ### Hyper parameters
GPUS = [mx.gpu(0)]
EPOCH = 10 
BATCH_SIZE = 100
MODEL_TYPE = "allconvnet"
DATASET_TYPE = "cifar10"
print("# epoch:{}, batch:{}, dataset:{}, gpus:{}".format(\
        EPOCH, BATCH_SIZE, DATASET_TYPE, GPUS))


# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()
train_data = dataset['train']['data'].astype(np.float32)/255.
train_size, image_size, _, num_channels = train_data.shape
num_labels  = len(list(set(dataset['train']['target'])))

train_lbl = dataset['train']['target'].astype(np.int8)
val_data = dataset['test']['data'].astype(np.float32)/255.
val_lbl = dataset['test']['target'].astype(np.int8)
print("# train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))
train_iter = mx.io.NDArrayIter(train_data, train_lbl, BATCH_SIZE, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_lbl, BATCH_SIZE)


# ### define net and trainer
net = allconvnet if MODEL_TYPE=='allconvnet' else\
        resnet if MODEL_TYPE=='resnet' else shallownet

#### Load trainer
model = mx.mod.Module(
        context=GPUS,
        symbol = net(num_labels),
        )

model.fit(
        train_data=train_iter,
        eval_data=val_iter,
        num_epoch = EPOCH,
        optimizer_params={'learning_rate':0.001},
        optimizer='adam',
        batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 500)
        )


prob = model.predict(val_iter)[0].asnumpy()
print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))
