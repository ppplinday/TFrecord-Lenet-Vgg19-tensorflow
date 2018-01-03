import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from load_data import load_CIFAR10
from model_lenet import Model_Lenet
from cifar10_model import Model_cifar10
from data_preprocess import _preprocess, transform, transform_test, data_preprocess
import config

def main(model_name):
	cifar10_dir = 'cifar-10-batches-py'
	X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

	sess = tf.Session()
	parameter_path = "checkpoint_" + model_name + "/variable.ckpt"
	if model_name == "lenet":
		print('loaded the lenet model')
		X_test = data_preprocess(X_test, train=False)
		model = Model_Lenet()
	elif model_name == "vgg19":
		print('loaded the vgg19 model')
		model = Model_cifar10()
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (10000, 3072))
		X_test = np.array(_preprocess(X_test))
	else:
		print('cannot find the checkpoint!')
		return ;

	saver = tf.train.Saver()
	saver.restore(sess, parameter_path)
	print('loaded the weight')
	sum = 0.0;
	for i in range(X_test.shape[0]):
		accurary = sess.run([model.train_accuracy], 
			feed_dict={model.input_image: X_test[i:i + 1], model.input_label: Y_test[i: i + 1]})
		sum += accurary[0]
	print('Accurary: {}'.format(sum / X_test.shape[0]))

if __name__ == "__main__":
	model_name = sys.argv[1]
	main(model_name)