from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from util import log
from pprint import pprint

import tensorflow.contrib.slim as slim

import os
import time
import tensorflow as tf
import h5py

def main();
	cifar10_dir = 'cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test)
	print(y_test)

if __name__ == '__main__':
	main()