# coding: utf-8
#!/usr/bin/env python


# python2, python3の互換性のためのおまじない
from __future__ import print_function

import numpy
import tensorflow as tf
#import matplotlib.pyplot as plt

import cifar10
import cifar100

# ### Hyper parameters

EPOCH = 10
BATCH_SIZE = 100
DATASET_TYPE = "cifar10"
CHANNEL_dense1 = 96
CHANNEL_dense2 = 192

# ### Load dataset and make target labels
dataset_load = cifar10 if DATASET_TYPE == "cifar10" else cifar100
dataset = dataset_load.load()

train_size = len(dataset['train']['target'])
num_labels  = len(list(set(dataset['train']['target'])))
image_size = len(dataset['train']['data'][0][0])
num_channels = len(dataset['train']['data'][0][0][0])

print(train_size, image_size, num_channels, num_labels)

train_data = dataset['train']['data'].astype(numpy.float32)
test_data = dataset['test']['data'].astype(numpy.float32)

### convert train and test label to one-hot vector
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = [numpy.arange(num_labels) for i in range(num_labels)]
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

train_target = dense_to_one_hot(dataset['train']['target'], num_labels)
test_target = dense_to_one_hot(dataset['test']['target'], num_labels)


print("train_data:{}, train_target:{}, test_data:{}, test_target:{}".format(
        train_data.shape, train_target.shape, test_data.shape, test_target.shape))
print(train_target[0])



# The random seed that defines initialization.
SEED = 100

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step, which we'll write once we define the graph structure.
train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, image_size, image_size, num_channels))
train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, num_labels))

# For the validation and test data, we'll just hold the entire dataset in
# one constant node.
test_data_node = tf.constant(test_data)

# The variables below hold all the trainable weights. For each, the
# parameter defines how the variables will be initialized.
def conv_weight_def(f_size=3, input_c=4, output_c=96, SEED=10):
    conv_weights = tf.Variable(
      tf.truncated_normal([f_size, f_size, input_c, output_c],  
                      stddev=0.1,
                      seed=SEED))
    conv_biases = tf.Variable(tf.zeros([output_c]))
    return conv_weights, conv_biases


conv1_weights, conv1_biases = conv_weight_def(3, num_channels, CHANNEL_dense1, SEED)
conv2_weights, conv2_biases = conv_weight_def(3, CHANNEL_dense1, CHANNEL_dense1, SEED)
conv3_weights, conv3_biases = conv_weight_def(3, CHANNEL_dense1, CHANNEL_dense1, SEED)
conv4_weights, conv4_biases = conv_weight_def(3, CHANNEL_dense1, CHANNEL_dense2, SEED)
conv5_weights, conv5_biases = conv_weight_def(3, CHANNEL_dense2, CHANNEL_dense2, SEED)
conv6_weights, conv6_biases = conv_weight_def(3, CHANNEL_dense2, CHANNEL_dense2, SEED)
conv7_weights, conv7_biases = conv_weight_def(3, CHANNEL_dense2, CHANNEL_dense2, SEED)
conv8_weights, conv8_biases = conv_weight_def(1, CHANNEL_dense2, CHANNEL_dense2, SEED)
conv9_weights, conv9_biases = conv_weight_def(1, CHANNEL_dense2, num_labels, SEED)

print('Done network weight define')

def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    def conv_stpad_def(data, weight, bias, stride=1, pad='SAME'):
        conv = tf.nn.conv2d(data,
                        weight,
                        strides=[1, stride, stride, 1],
                        padding=pad)
        relu_conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return relu_conv

    def avg_pool(data, size=8):
        return tf.nn.avg_pool(data, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='VALID')

    # batch_normalization
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
        
        return tf.cond(phase_train, update, average) 
    
    # comp graph
    phase_train = tf.placeholder(tf.bool, (BATCH_SIZE, 1), name='phase_train') if train else None
    conv1 = tf.nn.dropout(conv_stpad_def(data, conv1_weights, conv1_biases, 1, 'VALID'), 0.2, seed=SEED)
    conv2 = conv_stpad_def(conv1, conv2_weights, conv2_biases, 1, 'SAME')
    # global average pooling
    conv3 = conv_stpad_def(conv2, conv3_weights, conv3_biases, 2, 'VALID')
    bn1 = batch_norm(conv3, CHANNEL_dense1, phase_train)
    #bn1 = batch_norm_wrapper(conv3, phase_train)
    conv4 = conv_stpad_def(bn1, conv4_weights, conv4_biases, 1, 'SAME')
    conv5 = conv_stpad_def(conv4, conv5_weights, conv5_biases, 1, 'SAME')
    # global average pooling
    conv6 = conv_stpad_def(conv5, conv6_weights, conv6_biases, 2, 'VALID')
    bn2 = batch_norm(conv6, CHANNEL_dense2, phase_train) 
    #bn2 = batch_norm_wrapper(conv6, phase_train) 
    conv7 = conv_stpad_def(bn2, conv7_weights, conv7_biases, 1, 'SAME')
    conv8 = conv_stpad_def(conv7, conv8_weights, conv8_biases, 1, 'VALID')
    conv9 = conv_stpad_def(conv8, conv9_weights, conv9_biases, 1, 'VALID')
    gap = avg_pool(conv9, 6) 
    # Reshape the feature map cuboid into a 2D matrix 
    gap_shape = gap.get_shape().as_list()
    reshape = tf.reshape(
        gap,
        [gap_shape[0], gap_shape[1] * gap_shape[2] * gap_shape[3]])
  
    return reshape
print('Done AllConvNet define')


# Tining computation: logits + cross-entropy loss.
logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  logits=logits, labels=train_labels_node))

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per EPOCH, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=batch)
#optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                       0.9).minimize(loss,
#                                                     global_step=batch)

# Predictions for the minibatch, validation set and test set.
train_prediction = tf.nn.softmax(logits)
# We'll compute them only once in a while by calling their {eval()} method.
#validation_prediction = tf.nn.softmax(model(validation_data_node))
test_prediction = tf.nn.softmax(model(test_data_node))

print('Done optimizer define')

# Create a new interactive session that we'll use in
# subsequent code cells.
s = tf.InteractiveSession()

# Use our newly created session as the default for 
# subsequent operations.
s.as_default()

# Initialize all the variables we defined above.
#tf.initialize_all_variables().run()
tf.global_variables_initializer().run()

# Grab the first BATCH_SIZE examples and labels.
batch_data = train_data[:BATCH_SIZE, :, :, :]
batch_labels = train_target[:BATCH_SIZE]

# This dictionary maps the batch data (as a numpy array) to the
# node in the graph it should be fed to.
feed_dict = {train_data_node: batch_data,
             train_labels_node: batch_labels}

# Run the graph and fetch some of the nodes.
_, l, lr, predictions = s.run(
  [optimizer, loss, learning_rate, train_prediction],
  feed_dict=feed_dict)

print('Done datafeeder define')



def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.float32)
    bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions

print('Done error calculation define')


# Train over the first 1/4th of our training set.
steps = train_size // BATCH_SIZE
for k in range(EPOCH):
    for step in range(steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across EPOCHs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_target[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)

        # Print out the loss periodically.
        if step % 100 == 0:
            error, _ = error_rate(predictions, batch_labels)
            print('Step %d of %d' % (step, steps))
            print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
            #print('Validation error: %.1f%%' % error_rate(
            #      validation_prediction.eval(), validation_labels)[0])



    test_error, confusions = error_rate(test_prediction.eval(), test_target)
    print('Test error: %.1f%%' % test_error)

"""
### optional visualize
plt.clf()
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(False)
plt.xticks(numpy.arange(num_labels))
plt.yticks(numpy.arange(num_labels))
plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

plt.xlim([-0.5, 1.5])
plt.ylim([1.5, -0.5])

for i, cas in enumerate(confusions):
    for j, count in enumerate(cas):
        if count > 0:
            xoff = .07 * len(str(count))
            plt.text(j-xoff, i+.2, int(count), fontsize=48, color='white')

plt.savefig()


## plot a number of classified label
plt.clf()
plt.xticks(numpy.arange(num_labels))
plt.xlim(range(num_labels))
plt.hist(numpy.argmax(test_labels, 1));
plt.savefig("hist.png")

"""
