# coding: utf-8
#!/usr/bin/env python

#from __future__ import print_function

import logging
#logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
import mxnet as mx

import cifar10
#import cifar100


# ### Hyper parameters
NUM_GPU = 1
EPOCH = 100 
BATCH_SIZE = 100
MODEL_TYPE = "allconvnet"
DATASET_TYPE = "cifar10"
GPUS = [mx.gpu(n) for n in range(NUM_GPU)] if NUM_GPU > 1 else\
        mx.gpu(0) if NUM_GPU==1 else mx.cpu()
print("epoch:{}, batch:{}, dataset:{}, gpus:{}".format(\
        EPOCH, BATCH_SIZE, DATASET_TYPE, GPUS))


# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()
train_data = dataset['train']['data'].astype(np.float32)/255.
train_size, image_size, _, num_channels = train_data.shape
num_labels  = len(list(set(dataset['train']['target'])))

#train_lbl = mx.nd.one_hot(mx.nd.array(dataset['train']['target']), num_labels)
train_lbl = dataset['train']['target'].astype(np.int8)
val_data = dataset['test']['data'].astype(np.float32)/255.
#val_lbl = mx.nd.one_hot(mx.nd.array(dataset['test']['target']), num_labels)
val_lbl = dataset['test']['target'].astype(np.int8)
print("# train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))

train_iter = mx.io.NDArrayIter(train_data, train_lbl, BATCH_SIZE, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_lbl, BATCH_SIZE)

####
# define neural network by debug
data = mx.symbol.Variable('data')
#data = mx.sym.Variable('data')
h1 = mx.sym.Convolution(data=data, pad=(1,1), kernel=(3,3), stride=(1,1), num_filter=32)
h2 = mx.sym.Activation(data=h1, act_type="relu")
h3 = mx.sym.Pooling(data=h2, pool_type="max", kernel=(3,3), stride=(2,2))
#h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192)
h4 = mx.sym.Convolution(data=h3, pad=(1,1), kernel=(3,3), stride=(1,1), num_filter=64)
h5 = mx.sym.Activation(data=h4, act_type="relu")
h6 = mx.sym.Pooling(data=h5, pool_type="max", kernel=(3,3), stride=(2,2))
h7 = mx.sym.Flatten(data=h6)
#h8 = mx.sym.FullyConnected(data=h7, num_hidden=1000)
h9 = mx.sym.Activation(data=h7, act_type="relu")
h10 = mx.symbol.FullyConnected(data=h9, num_hidden=num_labels)
#h10 = mx.sym.FullyConnected(data=h7, num_hidden=num_labels)
#h = mx.sym.Activation(data=h, act_type="relu")
cnn = mx.sym.SoftmaxOutput(data=h10, name='softmax')
"""

h = mx.sym.Variable('data')
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96, stride=(1,1), pad=(1,1))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96, stride=(1,1), pad=(1,1))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=96, stride=(2,2))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192, stride=(1,1), pad=(1,1))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192, stride=(1,1), pad=(1,1))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192, stride=(2,2))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(3, 3), num_filter=192, stride=(1,1), pad=(1,1))
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(1, 1), num_filter=192)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Convolution(data=h, kernel=(1, 1), num_filter=num_labels)
h = mx.sym.BatchNorm(h)
h = mx.sym.Activation(data=h, act_type="relu")
h = mx.sym.Pooling(data=h, global_pool=True, pool_type="avg", kernel=(8, 8))
h = mx.sym.Flatten(data=h)
cnn = mx.sym.SoftmaxOutput(data=h, name='softmax')
#return cnn
"""

#### 
model = mx.mod.Module(
        context=GPUS,
        symbol = cnn,
        )

model.fit(
        train_data=train_iter,
        eval_data=val_iter,
        num_epoch = EPOCH,
        optimizer_params={'learning_rate':0.001},
        optimizer='adam',
        #eval_metric=['acc', 'ce'],
        batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 500)
        )


prob = model.predict(val_iter)[0].asnumpy()
print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))

