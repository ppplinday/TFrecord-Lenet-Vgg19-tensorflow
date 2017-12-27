import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from load_data import load_CIFAR10
from model import Model
from cifar10_model import Model_cifar10
import config
from scipy.misc import imresize


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
	label = np.zeros((10000, 10))
	for i in range(10000):
		label[i][y_test[i]] = 1

	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (10000, 3072))
	print(X_test.shape)
	X_test = np.array(_preprocess(X_test))

	sess = tf.Session()
	#model = Model()
	model = Model_cifar10()
	parameter_path = "checkpoint/variable.ckpt"

	saver = tf.train.Saver()
	saver.restore(sess, parameter_path)
	print('loaded the weight')

	accurary = sess.run([model.train_accuracy], 
		feed_dict={model.input_image: X_test, model.input_label: label})
	print('Accurary: {}'.format(accurary))

if __name__ == "__main__":
	main()