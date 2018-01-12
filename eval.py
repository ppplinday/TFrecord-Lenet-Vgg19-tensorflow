import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from load_data import load_CIFAR10
from model_lenet import Model_Lenet
from model_vgg19 import Model_Vgg19
from tfrecord import input
from data_preprocess import _preprocess, transform, transform_test, data_preprocess, label_one_hot
import config

def main(model_name):
	cifar10_dir = 'cifar-10-batches-py'
	X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

	#sess = tf.Session()
	parameter_path = "checkpoint_" + model_name + "/variable.ckpt"
	if model_name == "lenet":
		print('loaded the lenet model')
		X_test = data_preprocess(X_test, train=False, model=model_name)
		model = Model_Lenet()
	elif model_name == "vgg19":
		print('loaded the vgg19 model')
		model = Model_Vgg19()
		X_test = data_preprocess(X_test, train=False, model=model_name)
	else:
		print('cannot find the checkpoint!')
		return ;

	# saver = tf.train.Saver()
	# saver.restore(sess, parameter_path)
	# print('loaded the weight')
	# sum = 0.0;
	# for i in range(X_test.shape[0]):
	# 	accurary = sess.run([model.train_accuracy], 
	# 		feed_dict={model.input_image: X_test[i:i + 1], model.input_label: Y_test[i: i + 1]})
	# 	sum += accurary[0]
	# print('Accurary: {}'.format(sum / X_test.shape[0]))
	
	images, labels = input('test', 1)
	sum = 0.0;
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, parameter_path)
		print('loaded the weight')
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		for i in range(X_test.shape[0]):
			X_test, Y_test = sess.run([images, labels])
			print(X_test)
			print(Y_test)
			Y_test = label_one_hot(Y_test, 10)
			accurary = sess.run([model.train_accuracy], 
				feed_dict={model.input_image: X_test, model.input_label: Y_test})
			sum += accurary[0]
		print('Accurary: {}'.format(sum / X_test.shape[0]))

		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	model_name = sys.argv[1]
	main(model_name)