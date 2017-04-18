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
#from mxnet_model import AllConvNetBN, ResNet

def dense_to_one_hot(labels_dense, n_values):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.eye(n_values)[labels_dense].astype(np.float32)
    return labels_one_hot

# ### Hyper parameters
NUM_GPU = 0
GPUS = [mx.gpu(n) for n in range(NUM_GPU)] if NUM_GPU > 1 else\
        mx.gpu(0) if NUM_GPU==1 else mx.cpu()
EPOCH = 2 
BATCH_SIZE = 100
MODEL_TYPE = "allconvnet"
DATASET_TYPE = "cifar10"
print("epoch:{}, batch:{}, dataset:{}".format(EPOCH, BATCH_SIZE, DATASET_TYPE))


# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()
train_data = dataset['train']['data'].astype(np.float32)/255.
train_size, image_size, _, num_channels = train_data.shape
num_labels  = len(list(set(dataset['train']['target'])))

train_lbl = mx.nd.one_hot(mx.nd.array(dataset['train']['target']), num_labels)
val_data = dataset['test']['data'].astype(np.float32)/255.
val_lbl = mx.nd.one_hot(mx.nd.array(dataset['test']['target']), num_labels)
print("# train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))

train_iter = mx.io.NDArrayIter(train_data, train_lbl, BATCH_SIZE, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_lbl, BATCH_SIZE)

####
# define neural network by debug
#data = mx.symbol.Variable('data')
data = mx.sym.Variable('data')
h = mx.sym.Convolution(data=data, pad=(1,1), kernel=(3,3), stride=(1,1), dilate=(1,1), num_filter=96)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Pooling(data=h, pool_type="max", kernel=(2,2), stride=(1,1))
#h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192)
h = mx.sym.Convolution(data=h, pad=(1,1), kernel=(3,3), stride=(1,1), dilate=(1,1), num_filter=192)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Pooling(data=h, pool_type="max", kernel=(2,2), stride=(1,1))
h = mx.sym.Flatten(data=h)
#h = mx.sym.FullyConnected(data=h, num_hidden=1000)
#h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.FullyConnected(data=h, num_hidden=10)
h = mx.sym.Activation(data=h, act_type="relu")
cnn = mx.sym.SoftmaxOutput(data=h, name='softmax')
"""

h = mx.sym.Variable('data')
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96, stride=(2,2))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192, stride=(2,2))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(1, 1), num_filter=192)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(1, 1), num_filter=num_labels)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Pooling(data=h, pool_type="avg", kernel=(8, 8))
h = mx.sym.Flatten(data=h)
cnn = mx.sym.SoftmaxOutput(data=h, name='softmax')
#return cnn
"""

_model = cnn
#### 
#_model = shallow if MODEL_TYPE == 'resnet' else allconvnet

model = mx.mod.Module(
        context=GPUS,
        symbol = _model,
        )

model.fit(
        train_data=train_iter,
        eval_data=val_iter,
        num_epoch = EPOCH,
        #optimizer='adam',
        #eval_metric=['acc', 'ce'],
        batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 50)
        )

prob = model.predict(val_iter)[0].asnumpy()
print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))

