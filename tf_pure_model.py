# -*-coding: utf-8-*-
#!/usr/bin/env python


from __future__ import print_function

import numpy as np
import tensorflow as tf

class CNN:
    def __init__(self, dim_img=32, channel_img=3, n_class=10):
        self.dim_img = dim_img
        self.channel_img = channel_img
        self.n_class = n_class

    def build_network(self):
        pass

    def classify(self):
        print("tower_loss, before  self.model.classify()")
        logits = tf.nn.softmax(self.output)
        print("tower_loss, after  self.model.classify()")
        return logits
    
    def inference_loss(self):
        print("tower_loss, before self.model.loss()")

        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
        _loss = tf.nn.softmax_cross_entropy_with_logits(\
                logits=self.pred, labels=self.t)
        print("tower_loss, after  self.model.loss()")
        loss = tf.reduce_mean(_loss)
        print("tower_loss, after  self.model.reducemean()")
        print(loss)
        return loss

    def logits_loss(self):
        logits = self.pred()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits, labels=self.t))
        return logits, loss
     


class ResNet(CNN):
    def __init__(self, dim_img=32, channel_img=3, n_class=10, n=56):
        CNN.__init__(self, dim_img, channel_img, n_class)
        n_dict = {20:1, 32:2, 44:3, 56:4}
        self.n = n
        self.x = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, channel_img])
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.bool)
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()

    def build_network(self):
        h = self.x
        if self.n < 20 or (self.n - 20) % 12 != 0:
            print("ResNet depth invalid.")
            return 0

        num_conv = int((self.n - 20) / 12 + 1)
        layers = []

        h = convnet_bn_relu(h, [3, 3, self.channel_img, 16], 1, self.phase_train, 'conv1')
        layers.append(h)

        for i in range(num_conv):
            #with tf.variable_scope('conv2_%d' %(i+1)):
            scope = 'conv2_' +str(i+1)
            h_x = residual_block(layers[-1], 16, False, phase_train=self.phase_train, scope=scope+"_x")
            h = residual_block(h_x, 16, False, phase_train=self.phase_train, scope=scope)
            layers.append(h_x)
            layers.append(h)
            assert h.get_shape().as_list()[1:] == [32, 32, 16]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            scope = 'conv3_' +str(i+1)
            h_x = residual_block(layers[-1], 32, down_sample, phase_train=self.phase_train, scope=scope+"_x")
            h = residual_block(h_x, 32, False, phase_train=self.phase_train, scope=scope)
            layers.append(h_x)
            layers.append(h)
            assert h.get_shape().as_list()[1:] == [16, 16, 32]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            scope = 'conv4_' +str(i+1)
            h_x = residual_block(layers[-1], 64, down_sample, phase_train=self.phase_train, scope=scope+"_x")
            h = residual_block(h_x, 64, False, phase_train=self.phase_train, scope=scope)
            layers.append(h_x)
            layers.append(h)
            assert h.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            h = tf.reduce_mean(layers[-1], [1, 2])
            assert h.get_shape().as_list()[1:] == [64]

            fc_shape = [64, 10]
            fc_w = tf.get_variable('weight', shape=fc_shape,\
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc_b = tf.get_variable('biases', shape=fc_shape[1],\
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            h = tf.matmul(h, fc_w) + fc_b
            layers.append(h)

        print("layers.len():", len(layers))
        return layers[-1]



### AllConvolutionalNet
channel_dense1 = 96
channel_dense2 = 192

class AllConvNetBN(CNN):
    def __init__(self, dim_img=32, channel_img=3, n_class=10):
        CNN.__init__(self, dim_img, channel_img, n_class)
        self.x = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, channel_img])
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.bool)
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()

    def build_network(self):
        h = self.x
        #h = tf.nn.dropout(h, keep_prob=self.keep_prob)
        h = convnet(h, [3, 3, self.channel_img, channel_dense1], 1, 'SAME', 'conv1')
        h = batch_norm(h, channel_dense1, self.phase_train, 'bn1')
        h = convnet(h, [3, 3, channel_dense1, channel_dense1], 1, 'SAME', 'conv2')
        h = batch_norm(h, channel_dense1, self.phase_train, 'bn2')
        # 2 stride convolution
        h = convnet(h, [3, 3, channel_dense1, channel_dense1], 2, 'VALID', 'conv3')
        h = batch_norm(h, channel_dense1, self.phase_train, 'bn3')
        h = convnet(h, [3, 3, channel_dense1, channel_dense2], 1, 'SAME', 'conv4')
        h = batch_norm(h, channel_dense2, self.phase_train, 'bn4')
        h = convnet(h, [3, 3, channel_dense2, channel_dense2], 1, 'SAME', 'conv5')
        h = batch_norm(h, channel_dense2, self.phase_train, 'bn5')
        # 2 stride convolution
        h = convnet(h, [3, 3, channel_dense2, channel_dense2], 2, 'VALID', 'conv6')
        h = batch_norm(h, channel_dense2, self.phase_train, 'bn6')
        h = convnet(h, [3, 3, channel_dense2, channel_dense2], 1, 'SAME', 'conv7')
        h = batch_norm(h, channel_dense2, self.phase_train, 'bn7')
        # replace fc with 1×1 conv
        h = convnet(h, [1, 1, channel_dense2, channel_dense2], 1, 'VALID', 'conv8')
        h = batch_norm(h, channel_dense2, self.phase_train, 'bn8')
        h = convnet(h, [1, 1, channel_dense2, self.n_class], 1, 'VALID', 'conv9')
        h = batch_norm(h, self.n_class, self.phase_train, 'bn9')
        # global average pooling
        h = avg_pool(h, 6)
        h = flatten_layer(h)
        return h
   

### define layer function ###

def convnet(x, filter_shape, stride=1, pad='SAME', scope='conv'):
    out_channels = filter_shape[3]
    with tf.variable_scope(scope):
        filter_ = tf.get_variable('weight', dtype=tf.float32,\
                    shape=filter_shape,\
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[out_channels],\
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding=pad)
        bias = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(bias)
    return out


def avg_pool(data, size=6, stride=2):
    avg = tf.nn.avg_pool(data, ksize=[1, size, size, 1],\
            strides=[1, stride, stride, 1], padding='VALID')
    return avg

def flatten_layer(x):
    x_shape = x.get_shape().as_list()
    dim = x_shape[1] * x_shape[2] * x_shape[3]
    reshape = tf.reshape(x, [-1, dim])
    return reshape

def batch_norm(x, out_channel, phase_train, scope):
    with tf.variable_scope(scope):
        beta = tf.get_variable('beta', dtype=tf.float32, shape=[out_channel],\
            initializer=tf.truncated_normal_initializer(stddev=0.001))
        gamma = tf.get_variable('gamma', dtype=tf.float32, shape=[out_channel],\
                initializer=tf.truncated_normal_initializer(stddev=0.001))
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        decay = 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
       
        mean, var = tf.cond(phase_train, mean_var_with_update,\
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #batch_norm = tf.nn.batch_normalization(x, mean, var, beta, gamma, 2e-5)

        batch_norm = tf.nn.batch_norm_with_global_normalization(
        x, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    return batch_norm


def convnet_bn_relu(x, filter_shape, stride, phase_train, scope):
    out_channel = filter_shape[3]
    with tf.variable_scope(scope):
        filter_ = tf.get_variable('weight', dtype=tf.float32,\
                    shape=filter_shape,\
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
        beta = tf.get_variable('beta', dtype=tf.float32, shape=[out_channel],\
                initializer=tf.truncated_normal_initializer(stddev=0.001))
        gamma = tf.get_variable('gamma', dtype=tf.float32, shape=[out_channel],\
                initializer=tf.truncated_normal_initializer(stddev=0.001))
        batch_mean, batch_var = tf.nn.moments(conv, axes=[0, 1, 2])
        decay = 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,\
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #batch_norm = tf.nn.batch_normalization(conv, mean, var, beta, gamma, 2e-5)

        batch_norm = tf.nn.batch_norm_with_global_normalization(
                conv, mean, var, beta, gamma, 0.001,
                scale_after_normalization=True)


        out = tf.nn.relu(batch_norm)

    return out


def residual_block(x, output_depth, down_sample, projection=True, phase_train=True, scope=''):
    input_depth = x.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 2, 2, 1]
        x = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')

    for j in range(2):
        h = convnet_bn_relu(x, [3, 3, input_depth, output_depth], 1, phase_train, scope+"_"+str(j+1))

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = convnet_bn_relu(x, [1, 1, input_depth, output_depth], 2, phase_train)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = x

    res = h + input_layer
    return res
