#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import six  
from six.moves.urllib import request
import tarfile

fname = 'cifar-10-python.tar.gz'
batches = 5
batchsize = 10000
category_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def download():
    url = 'https://www.cs.toronto.edu/~kriz'
    request.urlretrieve(url+'/'+fname, fname)

def convert_train_image():
    data = np.zeros((batches*batchsize, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((batches*batchsize), dtype=np.uint8)
    with tarfile.open(fname, 'r:gz') as f:
        dir_name = u'cifar-10-batches-py'
        for i in six.moves.range(batches):
            batch_name = dir_name + u'/data_batch_' + str(i+1)
            r_data = f.extractfile(batch_name)
            batch = six.moves.cPickle.load(r_data, encoding='latin1')
            data[i*batchsize:(i+1)*batchsize] = batch['data'].reshape(batchsize, 32, 32, 3)
            labels[i*batchsize:(i+1)*batchsize] = batch['labels']
        # if use chainer, replace above line to below one
        #data = data.transpose(0, 2, 3, 1) 
    return data, labels

def convert_test_image():
    with tarfile.open(fname, 'r:gz') as f:
        dir_name = 'cifar-10-batches-py'
        batch_name = dir_name + '/test_batch'
        r_data = f.extractfile(batch_name)
        batch = six.moves.cPickle.load(r_data, encoding='latin1')
        data = batch['data'].reshape(batchsize, 32, 32, 3)
        # if use chainer, remove commentout
        #data = data.transpose(0, 2, 3, 1)
        labels = np.asarray(batch['labels']).astype(np.uint8)
    return data, labels


def load(name='cifar10.pkl'):
    with open(name, 'rb') as data:
        cifar10 = six.moves.cPickle.load(data)
        #cifar10 = six.moves.cPickle.load(data, encoding='latin1')
    return cifar10


if __name__ == '__main__':
    download()

    train_data, train_labels = convert_train_image()
    train = {'data': train_data, 
             'target': train_labels, 
             'size': len(train_labels), 
             'categories': len(category_names),
             'category_names': category_names}

    print(train['data'].shape)
    test_data, test_labels = convert_test_image()
    test = {'data': test_data, 
            'target': test_labels, 
            'size': len(test_labels), 
            'categories': len(category_names),
            'category_names': category_names}

    data = {'train': train, 'test': test}

    out_name = 'cifar10.pkl'
    with open(out_name, 'wb') as out_data:
        six.moves.cPickle.dump(data, out_data, -1)
