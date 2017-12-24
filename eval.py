import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from load_data import load_CIFAR10
from model import Model

def main():
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	label = np.zeros((10000, 10))
	for i in range(10000):
		label[i][y_test[i]] = 1

	sess = tf.Session()
	model = Model()
	parameter_path = "checkpoint/variable.ckpt"

	saver = tf.train.Saver()
	saver.restore(sess, parameter_path)
	print('loaded the weight')

	accurary = sess.run([model.train_accuracy], 
		feed_dict={model.input_image: X_test, model.input_label: label})
	print('Accurary: {}'.format(accurary))

if __name__ == "__main__":
	main()