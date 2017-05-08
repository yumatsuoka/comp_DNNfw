# coding: utf-8

import mxnet as mx


### shallow cnn
def shallownet(num_labels=10):
    h = mx.symbol.Variable('data')
    h = mx.sym.Convolution(data=h, pad=(1,1), kernel=(3,3), stride=(1,1), num_filter=32)
    h = mx.sym.Activation(data=h, act_type="relu")
    h = mx.sym.Pooling(data=h, pool_type="max", kernel=(3,3), stride=(2,2))
    h = mx.sym.Convolution(data=h, pad=(1,1), kernel=(3,3), stride=(1,1), num_filter=64)
    h = mx.sym.Activation(data=h, act_type="relu")
    h = mx.sym.Pooling(data=h, pool_type="max", kernel=(3,3), stride=(2,2))
    h = mx.sym.Flatten(data=h)
    h = mx.sym.Activation(data=h, act_type="relu")
    h = mx.symbol.FullyConnected(data=h, num_hidden=num_labels)
    loss = mx.sym.SoftmaxOutput(data=h, name='softmax')
    return loss 


### AllConvolutionalNet
def allconvnet(num_labels=10):
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
    loss = mx.sym.SoftmaxOutput(data=h, name='softmax')
    return loss


### ResNet
def resnet_block_fast(input_data, num_filter, name):
    conv1 = mx.symbol.Convolution(data=input_data, 
                                     kernel=(3, 3), 
                                     stride=(2, 2), 
                                     pad=(0, 0), 
                                     num_filter=num_filter, 
                                     name='conv'+name)
    conv1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu'+name)
    conv2 = mx.symbol.Convolution(data=conv1, 
                                     kernel=(3, 3), 
                                     stride=(1, 1), 
                                     pad=(1, 1), 
                                     num_filter=num_filter, 
                                     name='conv'+name+'_1')
    conv2 = mx.symbol.Activation(data=conv2, act_type='relu', name='relu'+name+'_1')
    conv3 = mx.symbol.Convolution(data=conv2, 
                                     kernel=(3, 3), 
                                     stride=(1, 1), 
                                     pad=(1, 1), 
                                     num_filter=num_filter, 
                                     name='conv'+name+'_2')
    conv3 = mx.symbol.Activation(data=conv3, act_type='relu', name='relu'+name+'_2')
    return conv3 + conv1

def resnet(num_labels=10):
    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), 
                                     stride=(2, 2), pad=(0, 0), 
                                     num_filter=32, name='conv1')
    conv1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
    res2 = resnet_block_fast(conv1, 32, '2')
    res3 = resnet_block_fast(res2, 32, '3')
    res4 = resnet_block_fast(res3, 64, '4')
    res5 = resnet_block_fast(res4, 128, '5')
    flatten = mx.symbol.Flatten(data=res5)
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=256, name='fc6')
    fc7 = mx.symbol.FullyConnected(data=fc6, num_hidden=num_labels, name='fc7')

    loss = mx.symbol.SoftmaxOutput(data=fc7, name='softmax')
    return loss
