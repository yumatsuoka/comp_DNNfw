# coding:utf-8

from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]
            
        
def to_multi_gpu(model, n_gpus=2):
    if n_gpus == 1:
        return model
    else:
        with tf.device('/cpu:0'):
            x = Input(model.input_shape[1:], name="test")
            
        towers = []
        # Assign a part of batch to all GPU
        for g in range(n_gpus):
            with tf.device('/gpu:' + str(g)):
                slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
                towers.append(model(slice_g))
                
        with tf.device('/cpu:0'):
            merged = Concatenate(axis=0)(towers)
            
        return Model(inputs=[x], outputs=[merged])
        
