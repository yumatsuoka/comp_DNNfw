# coding: utf-8
#!/usr/bin/env python

import time
import six
import numpy
import tensorflow as tf
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, batchsize=100, epoch=100, lr=1e-3):
        self.model = model
        self.train_data = dataset['train']['data']
        self.train_label = dataset['train']['target']
        self.test_data = dataset['test']['data']
        self.test_label = dataset['test']['target']
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.batchsize = batchsize
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
                loss_value = self.model.loss.eval(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32),\
                        self.model.t: t_batch.astype(numpy.float32),\
                        self.model.phase_train: True,\
                        self.model.keep_prob: 1.0,}, session=self.sess)
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

    
    def test(self, x):
        # future work
        y = self.model.classify(x)
        return 0

class DataFeeder:
    def __init__(self, feed_dict, batchsize=100):
        self.feed_dict = feed_dict
        self.batchsize = batchsize
    
    def augument_images(self, batch):
        # future work
        return batch

def separete_data():
    # bool型でcross_validationを使って
    # validation dataを作るメソッド 
    # ついでにpermutate 
    return 0

def dense_to_one_hot(labels_dense, n_values):
    ### convert train and test label to one-hot vector
    num_labels = labels_dense.shape[0]
    #n_values = numpy.max(labels_dense) + 1
    labels_one_hot = numpy.eye(n_values)[labels_dense].astype(numpy.float32)
    return labels_one_hot
