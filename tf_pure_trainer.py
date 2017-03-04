# coding: utf-8
#!/usr/bin/env python

import time
import six
import numpy
import tensorflow as tf
from tqdm import tqdm

"""
# reference on making multi gpu processing
# tensorflow.optimizer - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L257
# multigpu example 1 - https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN/blob/master/VAE-GAN-multi-gpu-celebA.ipynb
# multigpu example 2 - https://github.com/asheshjain399/Tensormodels/blob/master/tensormodels/multigpu.py
# multigpu example 3 - https://github.com/distrect9/tensorflow_distribute_multi_gpus/blob/master/multigpu.py
# multigpu example 4 - https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

"""

class Trainer:
    def __init__(self, model, dataset, batchsize=100, epoch=100, num_gpu=1, lr=1e-3):
        self.model = model
        self.train_data = dataset['train']['data']/255.
        self.train_label = dataset['train']['target']
        self.test_data = dataset['test']['data']/255.
        self.test_label = dataset['test']['target']
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.batchsize = batchsize
        self.num_gpu = num_gpu
        self.batch_index = self.train_size // self.batchsize
        self.epoch = epoch
        self.learning_rate = lr
        self.data_augment = False
        self.dump_log = {'trainloss':numpy.zeros(self.epoch), \
                'trainacc':numpy.zeros(self.epoch),\
                'testloss':numpy.zeros(self.epoch),\
                'testacc':numpy.zeros(self.epoch), \
                'epoch':numpy.zeros(self.epoch),\
                'iteration':numpy.zeros(self.epoch*self.batch_index),\
                'e_time':numpy.zeros(self.epoch)\
                }

        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(\
                learning_rate=self.learning_rate).minimize(\
                self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

    def fit(self):
        for e in six.moves.range(self.epoch):
            sum_loss = 0
            start_time = time.time() 
            perm = numpy.random.permutation(self.train_size)
            x_train = self.train_data[perm]
            t_train = self.train_label[perm]
            for itr in tqdm(six.moves.range(self.batch_index)):
                x_batch = x_train[itr*self.batchsize:(itr+1)*self.batchsize] 
                t_batch = dense_to_one_hot(t_train[\
                        itr*self.batchsize:(itr+1)*self.batchsize],\
                        self.model.n_class)
                if self.data_augment:
                    # future work
                    x_batch = self.datafeeder.augument_images(x_batch)
                self.optimizer.run(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32), 
                        self.model.t: t_batch.astype(numpy.float32),
                        self.model.phase_train: True,\
                        self.model.keep_prob: 0.8,}, session=self.sess)
                logit_value, loss_value = self.model.loss_logits.eval(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32),\
                        self.model.t: t_batch.astype(numpy.float32),\
                        self.model.phase_train: True,\
                        self.model.keep_prob: 1.0,}, session=self.sess)
                sum
                sum_loss += loss_value
            print('# epoch: {}'.format(e+1))
            self.dump_log['epoch'][e] = e+1
            self.dump_log['iteration'][e] = (e+1) * self.batch_index
            self.dump_log['trainloss'][e] = sum_loss/self.batch_index
            print('# average loss[epoch]: {}'.format(\
                    sum_loss/self.batch_index))
            print('# validation')
            sum_loss = 0
            prediction = numpy.array([])
            answer = numpy.array([])
            for itr in six.moves.range(0, self.test_size, self.batchsize):
                x_batch = self.test_data[itr:itr+self.batchsize]
                t_batch = dense_to_one_hot(self.test_label[itr:itr+self.batchsize],\
                        self.model.n_class)
                output = self.model.pred.eval(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32),\
                        self.model.phase_train: False,\
                        self.model.keep_prob: 1.0,}, session=self.sess)
                prediction = numpy.concatenate([prediction, numpy.argmax(output, 1)])
                answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
            correct_prediction = numpy.equal(prediction, answer)
            accuracy = numpy.mean(correct_prediction)
            self.dump_log['testacc'][e] = accuracy*100
            print('# training acuracy: {} %'.format(accuracy*100))
            elapsed_time = time.time() - start_time
            self.dump_log['e_time'][e] = elapsed_time
            print('# elapsed time on this epoch: ', elapsed_time)
        with open('tf_train_log.pkl', mode='wb') as f:
            six.moves.cPickle.dump(self.dump_log, f, protocol=2)

    def multigpu_train(self):
        tower_grads = []
        for i in range(self.num_gpu):
            with tf.device('/gpu:%d' %i):
                with tf.namescope('Tower_%d' %i) as scope:
                    next_batch = all_input[i*self.batchsize:(i+1)*self.batchsize, :]
                    logits = self.model.pred(next_batch)
                    loss = tf.reduce_mean(tf.softmax_cross_entropy_with_logits(logits=logits, labels=self.t))
		    loss = self.model.inference_loss(feed_dict={\
	            self.model.x: x_batch.astype(numpy.float32), 
		    self.model.t: t_batch.astype(numpy.float32),
		    self.model.phase_train: True,\
		    self.model.keep_prob: 0.8,}, session=self.sess)
                    # make <list of variables>
                    params = tf.trainable_variables()
                    var_list = [j for j in params]
                    tf.get_variable_scope().reuse_variables()
                    one_grads = self.optimizer.compute_gradients(loss, var_list=var_list)
                    tower_grads.append(one_grads)
                       
        overall_loss = culc_overall_loss(tower_grads, culc_way='sum')
        self.optimizer.apply_gradients(overall_loss, self.global_step) 
        return logits, loss
 
    def test(self, x):
        # future work
        y = self.model.classify(x)
        return 0


def culc_overall_loss(tower_grads, culc_way='sum'):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0) if culc_way == 'average' else tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def dense_to_one_hot(labels_dense, n_values):
    ### convert train and test label to one-hot vector
    num_labels = labels_dense.shape[0]
    #n_values = numpy.max(labels_dense) + 1
    labels_one_hot = numpy.eye(n_values)[labels_dense].astype(numpy.float32)
    return labels_one_hot
