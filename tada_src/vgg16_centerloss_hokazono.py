from tqdm import tqdm
import random
import os
import sys
import glob
import cv2 
import numpy as np
import tensorflow as tf

class VGG16:
    
    def __init__(self):
        self.dimImg = 112 
        self.batch_size = 128 
        self.ndims = 128
        self.n_class = 5484
        self.learning_rate = 1e-3
        self.centers = tf.get_variable("ctrs", [self.n_class, self.ndims], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        self.x = tf.placeholder(tf.float32, [None, self.dimImg, self.dimImg, 3]) 
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.e = tf.placeholder(tf.float32, [self.n_class])
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()


    def build_network(self):

        def PReLU(x):
            dim = x.get_shape()[-1]
            alpha = tf.get_variable('alpha', dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
            return out

        def conv_layer(x, filter_shape, stride, scope):
            out_channels = filter_shape[3]
            with tf.variable_scope(scope):
                filter_ = tf.get_variable('weight', dtype=tf.float32, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                beta = tf.get_variable('beta', dtype=tf.float32, shape=[out_channels], initializer=tf.truncated_normal_initializer(stddev=0.0))
                gamma = tf.get_variable('gamma', dtype=tf.float32, shape=[out_channels], initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
                mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
                batch_norm = tf.nn.batch_norm_with_global_normalization(conv, mean, var, beta, gamma, 0.001, scale_after_normalization=True)
                out = PReLU(batch_norm)
            return out

        def max_pooling_layer(x, size, stride):
            out = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
            return out

        def flatten_layer(x):
            input_shape = x.get_shape().as_list()
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            input_transposed = tf.transpose(x, (0, 3, 1, 2))
            input_processed = tf.reshape(input_transposed, [-1, dim])
            return input_processed
    
        def full_connection_layer(x, hiddens, scope):
            input_shape = x.get_shape().as_list()
            dim = input_shape[1]
            with tf.variable_scope(scope):
                fc_W = tf.get_variable('fc_W', dtype=tf.float32, shape=[dim, hiddens], initializer=tf.truncated_normal_initializer(stddev=0.1))
                fc_b = tf.get_variable('fc_b', dtype=tf.float32, shape=[hiddens], initializer=tf.constant_initializer(0.0))
                fc = tf.nn.bias_add(tf.matmul(x, fc_W), fc_b)
                out = PReLU(fc)
            return out

        x = self.x
        x = conv_layer(x, [3, 3, 3, 64], 1, 'conv1a')
        x = conv_layer(x, [3, 3, 64, 64], 1, 'conv1b')
        x = max_pooling_layer(x, 2, 2)
        x = conv_layer(x, [3, 3, 64, 128], 1, 'conv2a')
        x = conv_layer(x, [3, 3, 128, 128], 1, 'conv2b')
        x = max_pooling_layer(x, 2, 2)
        x = conv_layer(x, [3, 3, 128, 256], 1, 'conv3a')
        x = conv_layer(x, [3, 3, 256, 256], 1, 'conv3b')
        x = conv_layer(x, [3, 3, 256, 512], 1, 'conv3c')
        x = max_pooling_layer(x, 2, 2)
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv4a')
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv4b')
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv4c')
        x = max_pooling_layer(x, 2, 2)
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv5a')
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv5b')
        x = conv_layer(x, [3, 3, 512, 512], 1, 'conv5c')
        x = max_pooling_layer(x, 2, 2)
        x = flatten_layer(x)
        x = full_connection_layer(x, 1024, 'fc1')
        x = full_connection_layer(x, self.ndims, 'fc2')
        return x

    
    def classify(self):

        def softmax_layer(x, hiddens, scope):
            input_shape = x.get_shape().as_list()
            dim = input_shape[1]
            with tf.variable_scope(scope):
                fc_W = tf.get_variable('W', dtype=tf.float32, shape=[dim, hiddens], initializer=tf.truncated_normal_initializer(stddev=0.1))
                fc_b = tf.get_variable('b', dtype=tf.float32, shape=[hiddens], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(tf.matmul(x, fc_W), fc_b)
            return out 

        x = softmax_layer(self.output, self.n_class, 'softmax')
        return x


    def inference_loss(self):
        loss = tf.add(self.softmax_loss(), self.center_loss())
        return loss

    def softmax_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.t))
        return loss

    def center_loss(self):
        alpha = 1e-3
        result = tf.zeros([self.n_class, self.ndims], dtype=tf.float32)
        for i, emb in zip(tf.split(0, self.batch_size, self.t), tf.split(0, self.batch_size, self.output)):
            i = tf.reshape(i, [-1, 1]) 
            rval = tf.mul(i, emb)
            result = tf.add(result, rval)
        denom = tf.reshape(self.e, [-1, 1]) 
        result = tf.div(result, denom)
        loss = tf.reduce_sum((result - self.centers) ** 2)
        return loss * alpha

    def get_centers(self):
        decay = 0.99
        result = np.zeros([self.n_class, self.ndims], dtype=np.float32)
        inputs = tf.convert_to_tensor(result)
        #result = tf.zeros([self.n_class, self.ndims], dtype=tf.float32)
        for i, emb in zip(tf.split(0, self.batch_size, self.t), tf.split(0, self.batch_size, self.output)):
            i = tf.reshape(i, [-1, 1]) 
            rval = tf.mul(i, emb)
            result = tf.add(result, rval)
        denom = tf.reshape(self.e, [-1, 1]) 
        diff = tf.div(result, denom)
        self.centers += (1 - decay) * diff
        return self.centers

    def train(self):
        def get_elements(t):
            tmp = list(np.argmax(t, 1)) 
            elements = [tmp.count(i) for i in range(self.n_class)]
            nonezero_elements = [1 if i == 0 else i for i in elements]
            return np.array(nonezero_elements)

        sys.path.insert(0, '../data/')
        from load_data import load_data
        from augment_images import augment_images

        print('... loading')
        x_train, t_train = load_data('../data/train')
        n_train = x_train.shape[0]
        print(x_train.shape)
        print(t_train.shape)

        print('... building')
        sess = tf.Session()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=global_step)

        print('... initializing')
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state('backup/'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('backup/')
            last_model = ckpt.model_checkpoint_path
            print('... restoring the model: {}'.format(last_model))
            saver.restore(sess, last_model)

        print('... training')
        batch_index = len(x_train) // self.batch_size
        while True:
            epoch = int(sess.run(global_step) / batch_index)
            sum_softmax_loss = 0
            sum_center_loss = 0
            perm = np.random.permutation(n_train)
            x_train = x_train[perm]
            t_train = t_train[perm]
            for i in tqdm(range(batch_index)):
                x_batch = x_train[i*self.batch_size:(i+1)*self.batch_size]
                t_batch = t_train[i*self.batch_size:(i+1)*self.batch_size]
                e_batch = get_elements(t_batch)
                x_batch = augment_images(x_batch)
                self.get_centers().eval(feed_dict={self.x: x_batch.astype(np.float32), self.t: t_batch.astype(np.float32), self.e: e_batch.astype(np.float32)}, session=sess)
                optimizer.run(feed_dict={self.x: x_batch.astype(np.float32), self.t: t_batch.astype(np.float32), self.e: e_batch.astype(np.float32)}, session=sess)
                softmax_loss_value = self.softmax_loss().eval(feed_dict={self.x: x_batch.astype(np.float32), self.t: t_batch.astype(np.float32)}, session=sess)
                center_loss_value = self.center_loss().eval(feed_dict={self.x: x_batch.astype(np.float32), self.t: t_batch.astype(np.float32), self.e: e_batch.astype(np.float32)}, session=sess)
                sum_softmax_loss += softmax_loss_value
                sum_center_loss += center_loss_value
            print('----- epoch {} -----'.format(epoch+1))
            print('softmax loss = {}'.format(sum_softmax_loss))
            print('center loss = {}'.format(sum_center_loss))
            print('... validating')
            prediction = np.array([])
            answer = np.array([])
            for i in range(0, n_train, self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                t_batch = t_train[i:i+self.batch_size]
                output = self.pred.eval(feed_dict={self.x: x_batch.astype(np.float32)}, session=sess)
                prediction = np.concatenate([prediction, np.argmax(output, 1)])
                answer = np.concatenate([answer, np.argmax(t_batch, 1)])
            correct_prediction = np.equal(prediction, answer)
            accuracy = np.mean(correct_prediction)
            print('training accuracy = {} %'.format(accuracy*100))
 
            print('... saving the model')
            saver = tf.train.Saver()
            saver.save(sess, 'backup/model', write_meta_graph=False)


if __name__ == "__main__":
    model = VGG16()
    model.train() 


