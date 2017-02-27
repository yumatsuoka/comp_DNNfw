# coding: utf-8
#!/usr/bin/env python


# python2, python3の互換性のためのおまじない
from __future__ import print_function

import numpy
import tensorflow as tf
learn = tf.contrib.learn
slim = tf.contrib.slim


### AllConvolutionalNetwork ###
n_kernel = [96, 192]

def allconvnet(x, y):
    y = slim.one_hot_encoding(y, n_class)
	
    with slim.arg_scope([slim.conv2d, slim.batch_norm], 
              activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        dp1 = slim.dropout(x, 0.2)
        conv1 = slim.conv2d(dp1, n_kernel[0], kernel_size=3, activation_fn=tf.nn.relu)
        conv2 = slim.conv2d(conv1, n_kernel[0], kernel_size=3, activation_fn=tf.nn.relu)
        # global average pooling
        conv3 = slim.conv2d(conv2, n_kernel[0], kernel_size=3, stride=2, padding='VALID', activation_fn=tf.nn.relu)
        bn3 = slim.batch_norm(conv3)
        conv4 = slim.conv2d(bn3, n_kernel[1], kernel_size=3, activation_fn=tf.nn.relu)
        conv5 = slim.conv2d(conv4, n_kernel[1], kernel_size=3, activation_fn=tf.nn.relu)
        # global average pooling
        conv6 = slim.conv2d(conv5, n_kernel[1], kernel_size=3, stride=2, padding='VALID', activation_fn=tf.nn.relu)
        bn6 = slim.batch_norm(conv6)
        conv7 = slim.conv2d(bn6, n_kernel[1], kernel_size=3, activation_fn=tf.nn.relu)
        conv8 = slim.conv2d(conv7, n_kernel[1], kernel_size=1, padding='VALID', activation_fn=tf.nn.relu)
        conv9 = slim.conv2d(conv8, n_class, kernel_size=1, padding='VALID', activation_fn=tf.nn.relu)
        gap = slim.avg_pool2d(conv9, [6, 6], scope='logits') 
        logits = slim.flatten(gap)
        loss = slim.losses.softmax_cross_entropy(logits, y, scope='loss')
        train_op = slim.optimize_loss(loss, slim.get_global_step(), learning_rate=0.01, optimizer='Adam')
    return {'class': tf.argmax(logits, 1), 'prob': slim.softmax(logits), loss, train_op}



#### resnet #####
n_reputations = [3, 8, 36, 3]
n_outputs = [64 * 2 ** (i + 1) for i in range(len(n_reputations))]
mode = 'bottleneck'

def bn_actv_conv(inputs, stride, *args, **kwargs):
    net = slim.batch_norm(inputs)

    return slim.conv2d(net, stride=stride, activation_fn=None, *args, **kwargs)

def shortcut(identity, residual, stride, channel):
    if stride > 1:
        identity = slim.max_pool2d(identity, 1, stride, scope='downsample')
    if channel:
        identity = tf.pad(identity, 
                          [[0, 0], [0, 0], [0, 0], [channel, channel]], 
                          name='projection')

    return identity + residual

def basic_unit(inputs, stride, num_outputs, *args, **kwargs):
    residual = slim.stack(inputs, bn_actv_conv, [stride, 1], 
                          num_outputs=num_outputs, kernel_size=[3, 3], *args, 
                          **kwargs)

    return shortcut(inputs, residual, stride, 
                    (num_outputs - inputs.get_shape().as_list()[-1]) // 2)

def bottleneck(inputs, stride, num_outputs, *args, **kwargs):
    kwargs.pop('scope')
    residual = bn_actv_conv(inputs, stride, num_outputs=num_outputs, 
                            kernel_size=[1, 1], *args, **kwargs)
    residual = bn_actv_conv(residual, 1, num_outputs=num_outputs, 
                            kernel_size=[3, 3], *args, **kwargs)
    residual = bn_actv_conv(residual, 1, num_outputs=num_outputs * 4, 
                            kernel_size=[1, 1], *args, **kwargs)

    return shortcut(inputs, residual, stride, 
                    (num_outputs * 4 - inputs.get_shape().as_list()[-1]) // 2)

def residual_block(inputs, id_block, *args, **kwargs):
    n_reputation = n_reputations[id_block]
    num_outputs = n_outputs[id_block]
    downsample = False if id_block == 0 else True
    strides = [2 if downsample and i == 0 else 1 for i in range(n_reputation)]
    channel_diff = num_outputs - inputs.get_shape().as_list()[-1]
    scope = 'conv%d' % (id_block + 2)

    if mode == 'basic':
        net = slim.stack(inputs, basic_unit, strides, num_outputs=num_outputs, 
                         scope=scope)
    else:
        net = slim.stack(inputs, bottleneck, strides, num_outputs=num_outputs, 
                         scope=scope)

    return net

def resnet(x, y):
    net = x
    y = slim.one_hot_encoding(y, n_class)

    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.elu):
        net = slim.conv2d(net, n_outputs[0] // 2, [7, 7], stride=2, 
                          activation_fn=None, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], padding='SAME', scope='pool1')

        net = slim.stack(net, residual_block, 
                         [i for i in range(len(n_reputations))])

        net = slim.batch_norm(net)
        net = slim.avg_pool2d(net, net.get_shape()[1:3])
        net = slim.flatten(net)
        logits = slim.fully_connected(net, n_class, activation_fn=None, 
                                      scope='logits')

    loss = slim.losses.softmax_cross_entropy(logits, y, scope='loss')
    train_op = slim.optimize_loss(loss, slim.get_global_step(),
                                  learning_rate=0.01, optimizer='Adam')

    return {'class': tf.argmax(logits, 1), 'prob': slim.softmax(logits)}, loss, train_op

