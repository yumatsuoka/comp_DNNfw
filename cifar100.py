#!/usr/bin/env python
# coding: utf-8
# python2

import numpy as np
import six  
from six.moves.urllib import request
import tarfile

fname = 'cifar-100-python.tar.gz'
category_names = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers', 'clock', 'computer_keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion','tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
category_names.sort()


def download():
    url = 'https://www.cs.toronto.edu/~kriz'
    request.urlretrieve(url+'/'+fname, fname)  


def convert_image(data_type='train'):
    dir_name = {'train':u'cifar-100-python/train', 'test':u'cifar-100-python/test'}
    with tarfile.open(fname, 'r:gz') as f:
        r_data = f.extractfile(dir_name[data_type])
        train_dic = six.moves.cPickle.load(r_data, encoding='latin1')
        data = np.asarray([d.reshape(3, 32, 32) for d in train_dic['data']], dtype=np.uint8)
        # if visualize image or use tensorflow, remove bellow commentout
        data = np.asarray([d.transpose(1,2,0) for d in data])
        labels = np.asarray(train_dic['fine_labels'], dtype=np.uint8)
    return data, labels


def load(name='cifar100.pkl'):
    with open(name, 'rb') as data:
        cifar100 = six.moves.cPickle.load(data, encoding='latin1')
    return cifar100


if __name__ == '__main__':
    download()

    which_data = 'train'
    train_data, train_labels = convert_image(which_data)
    train = {'data': train_data, 
                 'target': train_labels, 
                 'size': len(train_labels), 
                 'categories': len(category_names),
                 'category_names': category_names}

    which_data = 'test'
    test_data, test_labels = convert_image(which_data)
    test = {'data': test_data, 
                'target': test_labels, 
                'size': len(test_labels), 
                'categories': len(category_names),
                'category_names': category_names}

    data = {'train': train, 'test': test}

    out_name = 'cifar100.pkl'
    with open(out_name, 'wb') as out_data:
        six.moves.cPickle.dump(data, out_data, -1)
