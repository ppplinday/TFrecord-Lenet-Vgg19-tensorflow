from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow.contrib.slim as slim

import os
import time
import numpy as np
import tensorflow as tf
from load_data import load_CIFAR10
from model import Model

class Trainer:

	def __init__(self, network, sess, saver, dataset_xtrain, dataset_ytrain):
		self.learning_rate = 1e-3
		self.batch_size = 32
		self.num_epoch = 5
		self.num_sample = 50000
		self.model = network
		self.sess = sess
		self.dataset_xtrain = dataset_xtrain
		self.dataset_ytrain = dataset_ytrain

		self.train()

	def train(self):

		for epoch in range(self.num_epoch):
			for iter in range(self.num_sample // self.batch_size):
				start = iter * 32
				batch = self.dataset_xtrain[start:start + 32]
				temp = self.dataset_ytrain[start:start + 32]
				label = np.zeros((32, 10))
				for i in range(32):
					label[i][temp[i]] = 1
				self.sess.run(self.model.train_op, feed_dict={self.model.input_image: batch, self.model.input_label: label})

			if epoch % 2 == 0:
				loss, accurary = self.sess.run([self.model.loss, self.model.train_accuracy],
					feed_dict={self.model.input_image: batch, self.model.input_label: label})
				print('[Epoch {}] Loss: {} Accurary: {}'.format(epoch, loss, accurary))

		print('Done! End of training!')

def main():
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	sess = tf.Session()
	lenet = Model()
	parameter_path = "checkpoint/variable.ckpt"
	path_exists = "checkpoint"

	saver = tf.train.Saver()
	if os.path.exists(path_exists):
		saver.restore(sess, parameter_path)
		print('loaded the weight')
	else:
		sess.run(tf.global_variables_initializer())
		print('init all the weight')

	train = Trainer(lenet, sess, saver, X_train, y_train)
	save_path = saver.save(sess, parameter_path)


if __name__ == '__main__':
	main()