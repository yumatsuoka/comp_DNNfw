# coding: utf-8
#!/usr/bin/env python


# python2, python3の互換性のためのおまじない
from __future__ import print_function

import numpy
import tensorflow as tf
from sklearn.model_selection import KFold
learn = tf.contrib.learn
slim = tf.contrib.slim

import cifar10
import cifar100

# ### Hyper parameters

EPOCH = 2
BATCH_SIZE = 100
DATASET_TYPE = "cifar10"
#CHANNEL_dense1 = 96
#CHANNEL_dense2 = 192
val_kf = 3
SEED = 100

# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()

train_size = len(dataset['train']['target'])
num_labels  = len(list(set(dataset['train']['target'])))
image_size = len(dataset['train']['data'][0][0])
num_channels = len(dataset['train']['data'][0][0][0])
print("train_size:{}, image_size:{}, num_channels:{}, num_labels:{}".format(
                  train_size, image_size, num_channels, num_labels))

count = 0
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for idxes in kf.split(dataset['train']['data']):
	train_idx, val_idx = idxes
	count += 1
	if count == val_kf:
		break

train_data = dataset['train']['data'][train_idx].astype(numpy.float32)
train_labels = dataset['train']['target'][train_idx]
validation_data = dataset['train']['data'][val_idx].astype(numpy.float32)
validation_labels = dataset['train']['target'][val_idx]
test_data = dataset['test']['data'].astype(numpy.float32)
test_labels = dataset['test']['target']

print("train_data:{}, train_labels:{}, validation_data:{}, validation_labels:{}, test_data:{}, test_labels:{}".format(train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape, test_data.shape, test_labels.shape))


from tf_model import allconvnet

model = allconvnet

###

class CustomMonitor(learn.monitors.EveryN):
    def begin(self, max_steps):
        super(CustomMonitor, self).begin(max_steps)
        print('Start training')

    def end(self):
        super(CustomMonitor, self).end()
        print('Completed')

    def every_n_step_begin(self, step):
        return ['loss/value:0']

    def every_n_step_end(self, step, outputs):
        print('Step %d - loss: %s' % (step, outputs['loss/value:0']))

classifier = learn.Estimator(model_fn=model, model_dir='/tmp/my_model')
classifier.fit(x=train_data, y=train_labels, steps=EPOCH*train_size/BATCH_SIZE, 
               batch_size=BATCH_SIZE)#, 
               #monitors=learn.monitors.get_default_monitors(save_summary_steps=1000)+\
               #         [CustomMonitor(every_n_steps=10, first_n_steps=0)])

y_predicted = [p['class'] for p in classifier.predict(test_data, as_iterable=True)]
score = metrics.accuracy_score(test_data, y_predicted)
print('Accuracy: {0:f}'.format(score))
