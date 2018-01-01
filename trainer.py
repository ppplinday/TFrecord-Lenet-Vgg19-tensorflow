from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow.contrib.slim as slim

import os
import sys
import numpy as np
import tensorflow as tf
from load_data import load_CIFAR10
from model import Model
from cifar10_model import Model_cifar10
import config
from data_preprocess import _preprocess, transform

class Trainer:

	def __init__(self, network, sess, saver, dataset_xtrain, dataset_ytrain, X_test, y_test):
		self.learning_rate = config.learning_rate
		self.batch_size = config.batch_size
		self.num_epoch = config.num_epoch
		self.num_sample = config.num_sample
		self.model = network
		self.sess = sess
		self.dataset_xtrain = dataset_xtrain
		self.dataset_ytrain = dataset_ytrain
		self.X_test = X_test
		self.y_test = y_test

		self.train()

	def train(self):

		for epoch in range(self.num_epoch):
			for iter in range(self.num_sample // self.batch_size):
				start = iter * self.batch_size
				batch = self.dataset_xtrain[start:start + self.batch_size]
				label = self.dataset_ytrain[start:start + self.batch_size]

				self.sess.run(self.model.train_op, feed_dict={self.model.input_image: batch, self.model.input_label: label})

				if iter % 100 == 0 or iter == 390:
					loss, accurary, step, lr = self.sess.run([self.model.loss, self.model.train_accuracy, 
						self.model.global_step, self.model.lr],
						feed_dict={self.model.input_image: batch, self.model.input_label: label})

					print('[Epoch {}] Iter: {} Loss: {} Accurary: {} step: {} lr: {}'.format(epoch, iter, loss, accurary,step, lr))

			sum = 0.0;
			for i in range(10000):
				accurary = self.sess.run([self.model.train_accuracy], 
					feed_dict={self.model.input_image: self.X_test[i:i + 1], self.model.input_label: self.y_test[i: i + 1]})
				sum += accurary[0]
			print('Accurary: {}'.format(sum / 10000.0))

		print('Done! End of training!')

def main(model_name):
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	# y_train = tf.one_hot(y_train, 10)
	label = np.zeros((50000, 10))
	for i in range(50000):
		label[i, y_train[i]] = 1.0
	y_train = label

	label = np.zeros((10000, 10))
	for i in range(10000):
		label[i][y_test[i]] = 1

	x_mean = np.mean([x for x in X_train], axis=(0,1,2))
	x_std = np.std([x for x in X_train], axis=(0,1,2))
	x_res = []
	for x in X_train:
		img = transform(x, x_mean, x_std, expand_ratio=1.2, crop_size=(28,28))
		x_res.append(img)
	x_res = np.array(x_res)
	X_train = x_res

	y_mean = np.mean([y for y in X_test], axis=(0,1,2))
	y_std = np.std([y for y in X_test], axis=(0,1,2))
	y_res = []
	for y in X_test:
		img = transform(y, y_mean, y_std, expand_ratio=1.2, crop_size=(28,28))
		y_res.append(img)
	y_res = np.array(y_res)
	X_test = y_res
	print(X_train.shape)
	print(X_test.shape)
	#return ;

	sess = tf.Session()
	parameter_path = "checkpoint_" + model_name + "/variable.ckpt"
	path_exists = "checkpoint_" + model_name

	if model_name == "lenet":
		print('begin to train lenet model')
		model = Model()
	elif model_name == "vgg19":
		print('begin to train vgg19 model')
		model = Model_cifar10()
		X_train = np.array(X_train)
		X_train = np.reshape(X_train, (50000, 3072))
		X_train = np.array(_preprocess(X_train))
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (10000, 3072))
		X_test = np.array(_preprocess(X_test))
	else:
		print('we do not have this model')
		return ;

	saver = tf.train.Saver()
	if os.path.exists(path_exists):
		saver.restore(sess, parameter_path)
		print('loaded the weight')
	else:
		sess.run(tf.global_variables_initializer())
		print('init all the weight')

	train = Trainer(model, sess, saver, X_train, y_train, X_test, label)
	save_path = saver.save(sess, parameter_path)


if __name__ == '__main__':
	model_name = sys.argv[1]
	main(model_name)