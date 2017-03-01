# coding: utf-8
#!/usr/bin/env python

import six
import numpy
import tensorflow as tf


class AllConvNetBN:
    def __init__(self, dim_img=32, channel_img=3, batchsize=100, ndim=100, n_class=10):
        self.dim_img = dim_img 
        self.channel_img = channel_img
        self.batchsize = batchsize
        self.ndim = ndim
        self.n_class = n_class
        self.x = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, channel_img])
        self.t = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.float32)
        self.output = self.build_network()
        self.pred = self.classify()
        self.loss = self.inference_loss()
		self.CHANNEL_dense1 = 96
		self.CHANNEL_dense2 = 192

    def build_network(self):
        h = self.x
        h = tf.nn.dropout(h, keep_prob=self.keep_prob)
        h = convnet(h, [3, 3, self.channel_img, self.CHANNEL_dense1], 1, 'SAME')
        h = convnet(h, [3, 3, self.CHANNEL_dense1, self.CHANNEL_dense1], 1, 'SAME')
        # 2 stride convolution
        h = convnet(h, [3, 3, self.CHANNEL_dense1, self.CHANNEL_dense1], 2, 'VALID')
        h = batch_norm(h, self.CHANNEL_dense1, scope)
        h = convnet(h, [3, 3, self.CHANNEL_dense1, self.CHANNEL_dense2], 1, 'SAME')
        h = convnet(h, [3, 3, self.CHANNEL_dense2, self.CHANNEL_dense2], 1, 'SAME')
        # 2 stride convolution
        h = convnet(h, [3, 3, self.CHANNEL_dense2, self.CHANNEL_dense2], 2, 'VALID')
        h = batch_norm(h, self.CHANNEL_dense2, scope) 
        h = convnet(h, [3, 3, self.CHANNEL_dense2, self.CHANNEL_dense2], 1, 'SAME')
        # replace fc with 1×1 conv
        h = convnet(h, [1, 1, self.CHANNEL_dense2, self.CHANNEL_dense2], 1, 'VALID')
        h = convnet(h, [1, 1, self.CHANNEL_dense2, self.n_class], 1, 'VALID')
        # global average pooling
        h = avg_pool(h, 6) 
        h = flatten_layer(h)
        return h
    
    def classify(self):
        logits = tf.nn.softmax(self.output)
        return logits
    
    def inference_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.t)
        return loss

    def train(self):
        # おそらく使わない.
        train_data 
        return 0

class Trainer:
    def __init__(self, model, dataset, epoch=100, lr=1e-3):
        # future work
        #if validate == True
        #_dataset = separete_data(datset)
        self.model = model
        self.train_data = dataset['train']['data']
        self.train_label = dataset['train']['target']
        self.test_data = dataset['test']['data']
        self.test_label = dataset['test']['target']

        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.batchsize = self.model.batchsize
        self.batch_index = self.train_size // self.batchsize
        self.epoch = epoch
        self.learning_rate = lr

        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(\
                learning_rate=self.learning_rate).minimize(\
                self.loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        if tf.train/get_checkpoint_state('allconvnet'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('allconvnet/')
            last_model = ckpt.model_checkpoint_path
            print('# restoring the model: {}'.format(last_model))
            saver.restore(sess, last_model)

    def fit(self):
        for i in six.moves.range(self.epoch):
            sum_loss = 0
            perm = numpy.random.permutation(self.train_size)
            x_train = self.train_data[perm]
            t_train = self.train_label[perm]
            for i in tqdm(six.moves.range(self.batch_index)):
                x_batch = x_train[i*self.batchsize:(i+1)*self.batchsize] 
                t_batch = t_train[i*self.batchsize:(i+1)*self.batchsize] 
                if self.data_augment:
                    # future work
                    x_batch = self.datafeeder.augument_images(x_batch)
                self.optimizer.run(feed_dict={\
                        self.x: x_batch.astype(np.float32), 
                        self.t: t_batch.astype(np.float32)}, session=sess)
                loss_value = self.loss.eval(\
                        self.x: x_batch.astype(np.float32),\
                        self.t: t_batch.astype(np.float32)}, session=sess)
                sum_loss += loss_value
            print('# epoch: {}'.format(epoch+1))
            print('# average loss[epoch]: {}'.format(\
                    sum_loss/self.batch_index))
            print('# validation')
            prediction = np.array([])
            answer = np.array([])
            for i in six.moves.range(0, self.test_size, self.batchsize):
                x_batch = self.test_data[i:i+self.batchsize]
                y_batch = self.test_label[i:i+self.batchsize]
                output = self.pred.eval(feed_dict={\
                        self.x: x_batch.astype(np.float32)}, session=sess)
                prediction = np.concatenate([prediction, np.argmax(output, 1)])
                answer = np.concatenate([answer, np.argmax(t_batch, 1)])
            correct_prediction = np.equal(prediction, answer)
            accuracy = np.mean(correct_prediction)
            print('training acuracy: {} %',.format(acuracy*100))

    
    def test(self, x):
        # future work
        y = self.model.classify()
        return 0

class DataFeeder:
    def __init__(self, feed_dict, batchsize=100):
        self.feed_dict = feed_dict
        self.batchsize = batchsize
    
    def augument_images(self, batch):
        # future work
        return batch

#num_labels = len(list(set(self.train_label)))
@static
def separete_data():
    # bool型でcross_validationを使って
    # validation dataを作るメソッド 
    # ついでにpermutate 
    phase_train = tf.placeholder(tf.bool, (BATCH_SIZE, 1), name='phase_train') if train else None
	pass
    
def convnet(x, filter_shape, stride=1, pad='SAME'):#, scope):
	out_channels = filter_shape[3]
	filter_ = tf.get_variable('weight', dtype=tf.float32,\
			shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
	conv = tf.nn.conv2d(x, filter=filter_, stride=[1, stride, stride, 1], padding=pad)
    return conv

def avg_pool(data, size=6, stride=2):
    avg = tf.nn.avg_pool(data, ksize=[1, size, size, 1],\
			strides=[1, stride, stride, 1], padding='VALID')
    return avg

def flatten_layer(x):
    x_shape = x.get_shape().as_list()
    reshape = tf.reshape(x,[x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]])
    return reshape

def batch_norm(x, out_channel, scope):
    with tf.variable_scope(scope):
        beta = tf.get_variable('beta', dtype=tf.float32, shape=[out_channel],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))
        gamma = tf.get_variable('gamma', dtype=tf.float32, shape=[out_channel],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))
        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        bn = tf.nn.batch_norm_with_global_normalization(\
				x, mean, var, beta, gamma, 0.001, scale_after_normalization=True)
    return bn
    
if __name__ == "__main__":
    detaset = [1,]
    epoch = 10
    model = AllConvNetBN()
    trainer = Trainer(model, dataset, epoch)
    trainer.fit()
    #trainer.test()

        
#####





# 1_batch_normalization
def batch_norma(x, n_out, phase_train):
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

# 2_batch_normalization
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
	#return update if phase_train else average 
	return tf.cond(phase_train, update, average) 
