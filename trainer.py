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
from cifar10_model import Model_cifar10
import config
from scipy.misc import imresize

class Trainer:

	def __init__(self, network, sess, saver, dataset_xtrain, dataset_ytrain):
		self.learning_rate = config.learning_rate
		self.batch_size = config.batch_size
		self.num_epoch = config.num_epoch
		self.num_sample = config.num_sample
		self.model = network
		self.sess = sess
		self.dataset_xtrain = dataset_xtrain
		self.dataset_ytrain = dataset_ytrain

		self.train()

	def train(self):

		for epoch in range(self.num_epoch):
			for iter in range(self.num_sample // self.batch_size):
				start = iter * self.batch_size
				batch = self.dataset_xtrain[start:start + self.batch_size]
				temp = self.dataset_ytrain[start:start + self.batch_size]
				label = np.zeros((self.batch_size, 10))
				for i in range(self.batch_size):
					label[i][temp[i]] = 1
				self.sess.run(self.model.train_op, feed_dict={self.model.input_image: batch, self.model.input_label: label})

			if epoch % 1 == 0:
				loss, accurary = self.sess.run([self.model.loss, self.model.train_accuracy],
					feed_dict={self.model.input_image: batch, self.model.input_label: label})
				print('[Epoch {}] Loss: {} Accurary: {}'.format(epoch, loss, accurary))

		print('Done! End of training!')

def rotate_reshape(images, output_shape):
    """ Rotate and reshape n images"""
    # def r_r(img):
    #    """ Rotate and reshape one image """
    #    img = np.reshape(img, output_shape, order="F")
    #    img = np.rot90(img, k=3)
    # new_images = list(map(r_r, images))
    new_images = []
    for img in images:
        img = np.reshape(img, output_shape, order="F")
        img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images


def rescale(images, new_size):
    """ Rescale image to new size"""
    return list(map(lambda img: imresize(img, new_size), images))


def subtract_mean_rgb(images):
    """ Normalize by subtracting from the mean RGB value of all images"""
    return images - np.round(np.mean(images))

def _preprocess(images_1d, n_labels=10, dshape=(32, 32, 3),
                reshape=[224, 224, 3]):
    """ Preprocesses CIFAR10 images
    images_1d: np.ndarray
        Unprocessed images
    labels_1d: np.ndarray
        1d vector of labels
    n_labels: int, 10
        Images are split into 10 classes
    dshape: array, [32, 32, 3]
        Images are 32 by 32 RGB
    """
    # Reshape and rotate 1d vector into image
    images_raw = rotate_reshape(images_1d, dshape)
    # Rescale images to 224,244
    images = rescale(images_raw, reshape)
    # Subtract mean RGB value from every pixel
    #images = subtract_mean_rgb(images_rescaled)
    return images

def main():
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	sess = tf.Session()
	#lenet = Model()
	lenet = Model_cifar10()
	parameter_path = "checkpoint/variable.ckpt"
	path_exists = "checkpoint"

	X_train = np.array(X_train)
	X_train = np.reshape(X_train, (50000, 3072))
	print(X_train.shape)
	X_train = np.array(_preprocess(X_train))
	print(X_train.shape)
	#print(X_train)

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