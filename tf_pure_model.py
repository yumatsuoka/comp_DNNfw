# coding: utf-8
#!/usr/bin/env python

import numpy
import tensorflow as tf


class AllConvNetBN:
    def __init__(self):
        self.dim_img = 32
        self.channel_img = 3
        self.batchsize = 32
        self.ndim = 128
        self.n_class = 10
        self.learning_rate = 1e-3
        self.x = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, channel_img])
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()

    def build_network(self):
        h = self.x
        h = tf.nn.dropout(h, keep_prob=self.keep_prob)
        h = convnet(h, conv1_weights, conv1_biases, 1, 'SAME')
        h = convnet(h, conv2_weights, conv2_biases, 1, 'SAME')
        # global average pooling
        h = stride_convnet(h, conv3_weights, conv3_biases, 2, 'VALID')
        h = batch_norm(h, CHANNEL_dense1, scope)
        h = convnet(h, conv4_weights, conv4_biases, 1, 'SAME')
        h = convnet(h, conv5_weights, conv5_biases, 1, 'SAME')
        # global average pooling
        h = stride_convnet(h, conv6_weights, conv6_biases, 2, 'VALID')
        h = batch_norm(h, CHANNEL_dense2, scope) 
        h = convnet(h, conv7_weights, conv7_biases, 1, 'SAME')
        h = convnet(h, conv8_weights, conv8_biases, 1, 'VALID')
        h = convnet(h, conv9_weights, conv9_biases, 1, 'VALID')
        h = avg_pool(h, 6) 
        h = flatten_layer(h)
        return h
    
    def classify(self):
        logits = tf.nn.softmax(self.output)
        return logits
    
    def inference_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.t)
        return loss

    def train(self):
        train_data 
        return 0

class Trainer:
    def __init__(self, dataset, ):
        _dataset = separete_data(datset)
        self.train_data = dataset['train']
        self.train_data = dataset['train']

    def separete_data():
        pass

    def fit(x, ):
        pass

    def validate():
        pass
        


    
if __name__ == "__main__":
    model = VGG16()
    model.train() 

        




#########################

def conv_weight_def(f_size=3, input_c=4, output_c=96, SEED=10):
    conv_weights = tf.Variable(tf.truncated_normal([f_size, f_size, input_c, output_c], stddev=0.1, seed=SEED))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[output_c]))
    return conv_weights, conv_biases


conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, CHANNEL_dense1], stddev=0.1, seed=SEED))
conv1_biases = tf.Variable(tf.zeros([CHANNEL_dense1]))
conv2_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense1, CHANNEL_dense1], stddev=0.1, seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense1]))
conv3_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense1, CHANNEL_dense1], stddev=0.1, seed=SEED))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense1]))
conv4_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense1, CHANNEL_dense2], stddev=0.1, seed=SEED))
conv4_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense2]))
conv5_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense2, CHANNEL_dense2], stddev=0.1, seed=SEED))
conv5_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense2]))
conv6_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense2, CHANNEL_dense2], stddev=0.1, seed=SEED))
conv6_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense2]))
conv7_weights = tf.Variable(tf.truncated_normal([3, 3, CHANNEL_dense2, CHANNEL_dense2], stddev=0.1, seed=SEED))
conv7_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense2]))
conv8_weights = tf.Variable(tf.truncated_normal([1, 1, CHANNEL_dense2, CHANNEL_dense2], stddev=0.1, seed=SEED))
conv8_biases = tf.Variable(tf.constant(0.1, shape=[CHANNEL_dense2]))
conv9_weights = tf.Variable(tf.truncated_normal([1, 1, CHANNEL_dense2, num_labels], stddev=0.1, seed=SEED))
conv9_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))


def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    def conv_stpad_def(data, weight, bias, stride=1, pad='SAME'):
        conv = tf.nn.conv2d(data, weight, strides=[1, stride, stride, 1],padding=pad)
        relu_conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return relu_conv


    # 1_batch_normalization
    def batch_norm(x, n_out, phase_train):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
    ####

    # 2_batch_normalization
    def batch_norm_wrapper(inputs, phase_train=None, decay=0.99):
        epsilon = 1e-5
        out_dim = inputs.get_shape()[-1]
        scale = tf.Variable(tf.ones([out_dim]))
        beta = tf.Variable(tf.zeros([out_dim]))
        pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
        pop_var = tf.Variable(tf.ones([out_dim]), trainable=False)

        if phase_train == None:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        rank = len(inputs.get_shape()) - 1
        # python3 ではrange()がlistを返さないことに注意

        def update():  # Update ema.
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale, epsilon),
        def average():  # Use avarage of ema.
            train_mean = pop_mean.assign(ema.average(batch_mean))
            train_var = pop_var.assign(ema.average(batch_var))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon),
        #return update if phase_train else average 
        return tf.cond(phase_train, update, average) 
    
    # ###comp graph
    phase_train = tf.placeholder(tf.bool, (BATCH_SIZE, 1), name='phase_train') if train else None
  
    return reshape

def convnet():
    return conv

def stride_convnet():
    return conv

def avg_pool(data, size=8, stride=2):
    avg = tf.nn.avg_pool(data, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')
    return avg

def flatten_layer(x):
    # Reshape the feature map cuboid into a 2D matrix 
    x_shape = x.get_shape().as_list()
    reshape = tf.reshape(x,[x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]])
    return reshape

def batch_norm(x, out_channel, scope):
    with tf.variable_scope(scope):
        beta = tf.getvariable('beta', dtype=tf.float32, shape=[out_channel], initializer=tf.truncated_normal_initializer(stddev=0.1))
        gamma = tf.getvariable('gamma', dtype=tf.float32, shape=[out_channel], initializer=tf.truncated_normal_initializer(stddev=0.1))
        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        bn = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 0.001, scale_after_normalization=True)
    return bn
