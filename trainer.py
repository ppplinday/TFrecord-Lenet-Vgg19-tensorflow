from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow.contrib.slim as slim

import os
import time
import tensorflow as tf
from load_data import load_CIFAR10
from model import Model

class Trainer(object):

	def init(self, network, dataset_xtrain, dataset_ytrain):
		self.learning_rate = 1e-3
		self.batch_size = 32
		self.num_epoch = 50
		self.num_sample = 50000
		self.model = Model()

		train()

	def train():

		for epoch in range(num_epoch):
			for iter in range(num_sample // batch_size):
				start = iter * 32
				batch = dataset_xtrain[start, start + 32]
				label = dataset_ytrain[start, start + 32]
				loss, accurary = network.run_single_step(batch, label)

			if epoch % 5 == 0:
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
	parameter_path = 'checkpoint/variable.ckpt'

	saver = tf.tf.train.Saver()
	if os.path.exists(parameter_path):
		saver.restore(parameter_path)
	else:
		sess.run(tf.initialize_all_variables())

	train = Trainer(lenet, X_train, y_train)


if __name__ == '__main__':
	main()