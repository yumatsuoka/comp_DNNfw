# coding: utf-8
#!/usr/bin/env python

import time
import six
import numpy
import tensorflow as tf

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
