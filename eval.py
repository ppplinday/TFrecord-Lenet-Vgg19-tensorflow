import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from load_data import load_CIFAR10
from model import Model
from cifar10_model import Model_cifar10
from data_preprocess import _preprocess, transform, transform_test
import config

def pro(X_train, train=True):
	x_mean = np.mean([x for x in X_train], axis=(0,1,2))
	x_std = np.std([x for x in X_train], axis=(0,1,2))
	x_res = []
	for x in X_train:
		img = transform(x, x_mean, x_std, expand_ratio=1.2, crop_size=(28,28), train=train)
		x_res.append(img)
	x_res = np.array(x_res)
	return x_res

def main(model_name):
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	label = np.zeros((10000, 10))
	for i in range(10000):
		label[i][y_test[i]] = 1

	sess = tf.Session()
	parameter_path = "checkpoint_" + model_name + "/variable.ckpt"
	if model_name == "lenet":
		print('loaded the lenet model')
		X_test = pro(X_test, train=False)
		print(X_test.shape)
		model = Model()
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
	for i in range(10000):
		accurary = sess.run([model.train_accuracy], 
			feed_dict={model.input_image: X_test[i:i + 1], model.input_label: label[i: i + 1]})
		sum += accurary[0]
	print('Accurary: {}'.format(sum / 10000.0))

if __name__ == "__main__":
	model_name = sys.argv[1]
	main(model_name)