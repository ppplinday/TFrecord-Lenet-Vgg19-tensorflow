import os
import sys
import config
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from load_data import load_CIFAR10
from model_lenet import Model_Lenet
from model_vgg19 import Model_Vgg19
from tfrecord import input
from data_preprocess import _preprocess, transform, transform_test, data_preprocess, label_one_hot

class Trainer:

	def __init__(self, network, sess, X_train, Y_train, X_test, Y_test, name):
		self.batch_size = config.batch_size
		self.num_epoch = config.num_epoch
		self.num_sample = config.num_sample
		self.model = network
		self.sess = sess
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		self.name = name

		self.train()

	def train(self):

		for epoch in range(self.num_epoch):
			for iter in range(self.num_sample // self.batch_size):
				start = iter * self.batch_size
				# batch_x = self.X_train[start:start + self.batch_size]
				# batch_y = self.Y_train[start:start + self.batch_size]
				print('xxxxx')
				batch_x, batch_y = self.sess.run([self.X_train, self.Y_train])
				batch_x = data_preprocess(batch_x, model=self.name)
				batch_y = label_one_hot(batch_y, 10)
				print('yyyyy')

				self.sess.run(self.model.train_op, feed_dict={self.model.input_image: batch_x, self.model.input_label: batch_y})

				if iter % 100 == 0:
					loss, accurary, step, lr = self.sess.run([self.model.loss, self.model.train_accuracy, 
						self.model.global_step, self.model.lr],
						feed_dict={self.model.input_image: batch_x, self.model.input_label: batch_y})

					print('[Epoch {}] Iter: {} Loss: {} Accurary: {} step: {} lr: {}'.format(epoch, iter, loss, accurary,step, lr))

			sum = 0.0;
			for i in range(self.X_test.shape[0]):
				test_accurary = self.sess.run([self.model.train_accuracy], 
					feed_dict={self.model.input_image: self.X_test[i:i + 1], self.model.input_label: self.Y_test[i: i + 1]})
				sum += test_accurary[0]
			print('Accurary: {}'.format(sum / self.X_test.shape[0]))

		print('Done! End of training!')

def main(model_name):
	cifar10_dir = 'cifar-10-batches-py'
	X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

	X_test = data_preprocess(X_test, train=False, model=model_name)
	print(X_train.shape)
	print(X_test.shape)
	#return ;
	
	X_train, Y_train = input('train', 128)
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		parameter_path = "checkpoint_" + model_name + "/variable.ckpt"
		path_exists = "checkpoint_" + model_name

		if model_name == "lenet":
			print('begin to train lenet model')
			model = Model_Lenet()
		elif model_name == "vgg19":
			print('begin to train vgg19 model')
			model = Model_Vgg19()
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

		train = Trainer(model, sess, X_train, Y_train, X_test, Y_test, model_name)
		save_path = saver.save(sess, parameter_path)


if __name__ == '__main__':
	model_name = sys.argv[1]
	main(model_name)