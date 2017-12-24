import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def main():
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	sess = tf.Session()
	lenet = Model()
	parameter_path = "checkpoint/variable.ckpt"

	saver = tf.train.Saver()
	saver.restore(sess, parameter_path)
	print('loaded the weight')

	accurary = sess.run([self.train_accuracy], 
		feed_dict={self.model.input_image: X_test, self.model.input_label: y_test})
	print('Accurary: {}'.format(accurary))

if __name__ == "__main__":
	main()