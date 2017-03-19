# coding: utf-8
#!/usr/bin/env python

import re
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
    def __init__(self, model, train_data, val_data, batchsize=100, lr=1e-3):
        self.model = model
        self.train_data = train_data['data']
        self.train_label = train_data['target']
        self.val_data = val_data['data']
        self.val_label = val_data['target']
        self.train_size = len(self.train_label)
        self.batchsize = batchsize
        self.val_size = len(self.val_label)
        self.batch_index = self.train_size // self.batchsize
        self.learning_rate = lr
        self.initialize_optimizer()

    def initialize_optimizer(self):
        pass

    def fit(self, epoch):
        pass

    def test(self, test_data, test_label):
        pass


class StandardTrainer(Trainer):
    def __init__(self, model, train_data, val_data, batchsize=100, lr=1e-3):
        Trainer.__init__(self, model, train_data, val_data, batchsize)

    def initialize_optimizer(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(\
                learning_rate=self.learning_rate).minimize(\
                self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, epoch=100):
        dump_log = {'trainloss':numpy.zeros(epoch), \
		'trainacc':numpy.zeros(epoch),\
		'valloss':numpy.zeros(epoch),\
		'valacc':numpy.zeros(epoch), \
		'epoch':numpy.zeros(epoch),\
		'iteration':numpy.zeros(epoch*self.batch_index),\
		'e_time':numpy.zeros(epoch)\
		}
        for e in six.moves.range(epoch):
            sum_loss = 0
            #prediction = numpy.array([])
            #answer = numpy.array([])
            start_time = time.time()
            perm = numpy.random.permutation(self.train_size)
            x_train = self.train_data[perm]
            t_train = self.train_label[perm]
            for itr in tqdm(six.moves.range(self.batch_index)):
                x_batch = x_train[itr*self.batchsize:(itr+1)*self.batchsize]
                t_batch = dense_to_one_hot(t_train[\
                        itr*self.batchsize:(itr+1)*self.batchsize],\
                        self.model.n_class)
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
                #prediction = numpy.concatenate([prediction, numpy.argmax(logit_value, 1)])
                #answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
                sum_loss += loss_value
            print('# epoch: {}'.format(e+1))
            dump_log['epoch'][e] = e+1
            dump_log['iteration'][e] = (e+1) * self.batch_index
            dump_log['trainloss'][e] = sum_loss/self.batch_index
            print('# average train_loss[epoch]: {}'.format(\
                    sum_loss/self.batch_index))
            #correct_prediction = numpy.equal(prediction, answer)
            #accuracy = numpy.mean(correct_prediction)
            #dump_log['trainacc'][e] = accuracy*100
            #print('# training acuracy: {} %'.format(accuracy*100))
            elapsed_time = time.time() - start_time
            print('# validation')
            #sum_loss = 0
            prediction = numpy.array([])
            answer = numpy.array([])
            for itr in six.moves.range(0, self.val_size, self.batchsize):
                x_batch = self.val_data[itr:itr+self.batchsize]
                t_batch = dense_to_one_hot(self.val_label[itr:itr+self.batchsize],\
                        self.model.n_class)
                logit_value = self.model.pred.eval(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32),\
                        self.model.phase_train: False,\
                        self.model.keep_prob: 1.0,}, session=self.sess)
                #sum_loss += loss_value
                prediction = numpy.concatenate([prediction, numpy.argmax(logit_value, 1)])
                answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
	    #dump_log['valloss'][e] = sum_loss/self.batch_index
	    #print('# average valloss[epoch]: {}'.format(\
	    #	     sum_loss/self.batch_index))
            correct_prediction = numpy.equal(prediction, answer)
            accuracy = numpy.mean(correct_prediction)
            dump_log['valacc'][e] = accuracy*100
            print('# validation acuracy: {} %'.format(accuracy*100))
            elapsed_time = time.time() - start_time
            dump_log['e_time'][e] = elapsed_time
            print('# elapsed time on this epoch: ', elapsed_time)
        with open('tf_train_log.pkl', mode='wb') as f:
            six.moves.cPickle.dump(dump_log, f, protocol=2)

 
    def test(self, test_data, test_label):
        print('# test')
        test_size = len(test_label)
        sum_loss = 0
        prediction = numpy.array([])
        answer = numpy.array([])
        for itr in six.moves.range(0, test_size, self.batchsize):
            x_batch = test_data[itr:itr+self.batchsize]
            t_batch = dense_to_one_hot(test_label[itr:itr+self.batchsize],\
                    self.model.n_class)
            logit_value, loss_value = self.model.logits_loss.eval(feed_dict={\
                    self.model.x: x_batch.astype(numpy.float32),\
                    self.model.phase_train: False,\
                    self.model.keep_prob: 1.0,}, session=self.sess)
            sum_loss += loss_value
            prediction = numpy.concatenate([prediction, numpy.argmax(logit_value, 1)])
            answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
        print('# average testloss[epoch]: {}'.format(\
                sum_loss/self.batch_index))
        correct_prediction = numpy.equal(prediction, answer)
        accuracy = numpy.mean(correct_prediction)
        print('# test acuracy: {} %'.format(accuracy*100))
        return 0


class Multigpu_trainer(Trainer):
    def __init__(self, model, train_data, val_data, batchsize=100, lr=1e-3, num_gpus=1):
        self.num_gpus = num_gpus
        Trainer.__init__(self, model, train_data, val_data, batchsize, lr)

    def initialize_optimizer(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in six.moves.range(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('Tower_%d' %i) as scope:
                        loss = self.tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()
                        grads = self.optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = culc_overall_loss(tower_grads, culc_way='sum')
        apply_gradient_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())


    def fit(self, epoch=100):
        dump_log = {'trainloss':numpy.zeros(epoch), \
                'trainacc':numpy.zeros(epoch),\
                'valloss':numpy.zeros(epoch),\
                'valacc':numpy.zeros(epoch), \
                'epoch':numpy.zeros(epoch),\
                'iteration':numpy.zeros(epoch*self.batch_index),\
                'e_time':numpy.zeros(epoch)\
                }
        for e in six.moves.range(epoch):
            sum_loss = 0
            #prediction = numpy.array([])
            #answer = numpy.array([])
            start_time = time.time()
            perm = numpy.random.permutation(self.train_size)
            x_train = self.train_data[perm]
            t_train = self.train_label[perm]
            for itr in tqdm(six.moves.range(self.batch_index)):
                x_batch = x_train[itr*self.batchsize:(itr+1)*self.batchsize]
                t_batch = dense_to_one_hot(t_train[\
                        itr*self.batchsize:(itr+1)*self.batchsize],\
                        self.model.n_class)
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
                #prediction = numpy.concatenate([prediction, numpy.argmax(logit_value, 1)])
                #answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
                sum_loss += loss_value
            print('# epoch: {}'.format(e+1))
            dump_log['epoch'][e] = e+1
            dump_log['iteration'][e] = (e+1) * self.batch_index
            dump_log['trainloss'][e] = sum_loss/self.batch_index
            print('# average train_loss[epoch]: {}'.format(\
                    sum_loss/self.batch_index))
            #correct_prediction = numpy.equal(prediction, answer)
            #accuracy = numpy.mean(correct_prediction)
            #dump_log['trainacc'][e] = accuracy*100
            #print('# training acuracy: {} %'.format(accuracy*100))
            elapsed_time = time.time() - start_time
            print('# validation')
            #sum_loss = 0
            prediction = numpy.array([])
            answer = numpy.array([])
            for itr in six.moves.range(0, self.val_size, self.batchsize):
                x_batch = self.val_data[itr:itr+self.batchsize]
                t_batch = dense_to_one_hot(self.val_label[itr:itr+self.batchsize],\
                        self.model.n_class)
                logit_value = self.model.pred.eval(feed_dict={\
                        self.model.x: x_batch.astype(numpy.float32),\
                        self.model.phase_train: False,\
                        self.model.keep_prob: 1.0,}, session=self.sess)
                #sum_loss += loss_value
                prediction = numpy.concatenate([prediction, numpy.argmax(logit_value, 1)])
                answer = numpy.concatenate([answer, numpy.argmax(t_batch, 1)])
            #dump_log['valloss'][e] = sum_loss/self.batch_index
            #print('# average valloss[epoch]: {}'.format(\
                    #	     sum_loss/self.batch_index))
            correct_prediction = numpy.equal(prediction, answer)
            accuracy = numpy.mean(correct_prediction)
            dump_log['valacc'][e] = accuracy*100
            print('# validation acuracy: {} %'.format(accuracy*100))
            elapsed_time = time.time() - start_time
            dump_log['e_time'][e] = elapsed_time
            print('# elapsed time on this epoch: ', elapsed_time)
        with open('tf_train_log.pkl', mode='wb') as f:
            six.moves.cPickle.dump(dump_log, f, protocol=2)

    def tower_loss(self, scope):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
           scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """
        # Get images and labels for CIFAR-10.
        #images, labels = cifar10.distorted_inputs()

        # Build inference Graph.
        #logits = cifar10.inference(images)
        # ここがうまくいかない
        self.model.loss()

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        #_ = cifar10.loss(logits, labels)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')
        return total_loss


def bu_tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
       scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build inference Graph.
    logits = cifar10.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss



    def backup_last_run(self, epoch):
        for e in six.moves.range(epoch):
            start_time = time.time()
            perm = numpy.random.permutation(self.train_size)
            x_train = self.train_data[perm]
            t_train = self.train_label[perm]
            for itr in tqdm(six.moves.range(self.batch_index//self.num_gpu)):
                tower_grads = []
                tower_logits = []
                tower_loss = []
                nb = self.num_gpu * itr
                for i in six.moves.range(self.num_gpu):
                    with tf.device('/gpu:%d' %i):
                        with tf.name_scope('Tower_%d' %i) as scope:
                            next_x =x_train[(nb+ i)*self.batchsize:(nb+ i+ 1)*self.batchsize]
                            next_y =dense_to_one_hot(t_train[(nb+ i)*self.batchsize:\
                                    (nb+ i+ 1)*self.batchsize], self.model.n_class)
                            logits, loss = self.model.logits_loss()
                            #loss = self.model.inference_loss(feed_dict={\
                                    #	 self.model.x: x_batch.astype(numpy.float32),
                            #	 self.model.t: t_batch.astype(numpy.float32),
                            #	 self.model.phase_train: True,\
                                    #		 self.model.keep_prob: 0.8,}, session=self.sess)
                            # make <list of variables>
                            params = tf.trainable_variables()
                            var_list = [j for j in params]
                            tf.get_variable_scope().reuse_variables()
                            one_grads = self.optimizer.compute_gradients(loss, var_list=var_list)
                            tower_grads.append(one_grads)
                            tower_logits.append(logits)
                            tower_loss.append(loss)

                overall_loss = culc_overall_loss(tower_grads, culc_way='sum')
                self.optimizer.apply_gradients(overall_loss, self.global_step)
        return tower_logits, tower_loss



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
