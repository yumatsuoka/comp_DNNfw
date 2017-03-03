# coding: utf-8
#!/usr/bin/env python


import numpy as np
import tensorflow as tf

class ResNet:
    def __init__(self, dim_img=32, channel_img=3, n_class=10):
        self.dim_img = dim_img 
        self.channel_img = channel_img
        self.n_class = n_class
        self.channel_dense1 = 96
        self.channel_dense2 = 192
        self.x = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, channel_img])
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.bool) 
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()
        n_dict = {20:1, 32:2, 44:3, 56:4}

    def build_network(self):
        h = self.x
        if n < 20 or (n - 20) % 12 != 0:
            print "ResNet depth invalid."
            return 0

        num_conv = (n - 20) / 12 + 1
        layers = []

        with tf.variable_scope('conv1'):
            conv1 = conv_layer(h, [3, 3, self.channel_img, 16], 1)
            layers.append(conv1)

        for i in range(num_conv):
            with tf.variable_scope('conv2_%d' %(i+1)):
                conv2_x = residual_block(layers[-1], 16, False)
                cov2 = residual_block(conv2_x, 16, False)
                layers.append(conv2_x)
                layers.append(conv2)
            assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

        for i in range(num_conv):
            down_sample = True if i== 0 else False
            with tf.variable_scope('conv3_%d' %(i+1)):
                conv3_x = residual_block(layers[-1], 32, down_sample)
                conv3 = residual_block(conv3_x, 32, False)
                layers.append(conv3_x)
                layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

        for i in range (num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = residual_block(layers[-1], 64, down_sample)
                conv4 = residual_block(conv4_x, 64, False)
                layers.append(conv4_x)
                layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])
            assert global_pool.get_shape().as_list()[1:] == [64]

            out = softmax_layer(global_pool, [64, 10])
            layers.append(out)

        return layers[-1]



### AllConvolutionalNet

class CNN:
    def __init__(self, dim_img=32, channel_img=3, n_class=10):

    def build_network(self):
        pass

    def classify(self):
        logits = tf.nn.softmax(self.output)
        return logits
    
    def inference_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=self.pred, labels=self.t))
        return loss


class AllConvNetBN:
    def __init__(self, dim_img=32, channel_img=3, n_class=10):
        self.dim_img = dim_img 
        self.channel_img = channel_img
        self.n_class = n_class
        self.channel_dense1 = 96
        self.channel_dense2 = 192
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
        h = convnet(h, [3, 3, self.channel_img, self.channel_dense1], 1, 'SAME', 'conv1')
        h = batch_norm(h, self.channel_dense1, self.phase_train, 'bn1')
        h = convnet(h, [3, 3, self.channel_dense1, self.channel_dense1], 1, 'SAME', 'conv2')
        h = batch_norm(h, self.channel_dense1, self.phase_train, 'bn2')
        # 2 stride convolution
        h = convnet(h, [3, 3, self.channel_dense1, self.channel_dense1], 2, 'VALID', 'conv3')
        h = batch_norm(h, self.channel_dense1, self.phase_train, 'bn3')
        h = convnet(h, [3, 3, self.channel_dense1, self.channel_dense2], 1, 'SAME', 'conv4')
        h = batch_norm(h, self.channel_dense2, self.phase_train, 'bn4')
        h = convnet(h, [3, 3, self.channel_dense2, self.channel_dense2], 1, 'SAME', 'conv5')
        h = batch_norm(h, self.channel_dense2, self.phase_train, 'bn5')
        # 2 stride convolution
        h = convnet(h, [3, 3, self.channel_dense2, self.channel_dense2], 2, 'VALID', 'conv6')
        h = batch_norm(h, self.channel_dense2, self.phase_train, 'bn6')
        h = convnet(h, [3, 3, self.channel_dense2, self.channel_dense2], 1, 'SAME', 'conv7')
        h = batch_norm(h, self.channel_dense2, self.phase_train, 'bn7')
        # replace fc with 1×1 conv
        h = convnet(h, [1, 1, self.channel_dense2, self.channel_dense2], 1, 'VALID', 'conv8')
        h = batch_norm(h, self.channel_dense2, self.phase_train, 'bn8')
        h = convnet(h, [1, 1, self.channel_dense2, self.n_class], 1, 'VALID', 'conv9')
        h = batch_norm(h, self.n_class, self.phase_train, 'bn9')
        # global average pooling
        h = avg_pool(h, 6) 
        h = flatten_layer(h)
        return h
    
    def classify(self):
        logits = tf.nn.softmax(self.output)
        return logits
    
    def inference_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=self.pred, labels=self.t))
        return loss

def convnet(x, filter_shape, stride=1, pad='SAME', scope='conv'):
    out_channels = filter_shape[3]
    with tf.variable_scope(scope):
        filter_ = tf.get_variable('weight', dtype=tf.float32,\
                    shape=filter_shape,\
                    initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', shape=[out_channels],\
                initializer=tf.truncated_normal_initializer())
        conv = tf.nn.conv2d(x, filter=filter_ , strides=[1, stride, stride, 1], padding=pad)
        bias = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(bias)
    return out

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
    conv, mean, var, beta, gamma, 0.001,
    scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out


def avg_pool(data, size=6, stride=2):
    avg = tf.nn.avg_pool(data, ksize=[1, size, size, 1],\
            strides=[1, stride, stride, 1], padding='VALID')
    return avg

def flatten_layer(x):
    x_shape = x.get_shape().as_list()
    dim = x_shape[1] * x_shape[2] * x_shape[3]
    reshape = tf.reshape(x,[-1, dim])
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
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 2e-5) 
    return normed 

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res

