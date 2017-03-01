# coding: utf-8
#!/usr/bin/env python

import tensorflow as tf



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
