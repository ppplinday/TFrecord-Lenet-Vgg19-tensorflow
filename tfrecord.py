import os
import numpy as np
import pickle as p
import tensorflow as tf
from load_data import load_CIFAR_batch

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images, labels, name):
	num = images.shape[0]
	filename = name + '.tfrecords'
	writer = tf.python_io.TFRecordWriter(filename)
	for i in range(num):
		img = images[i].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'label':_int64_feature(int(labels[i])),
			'image':_bytes_feature(img)
			}))
		writer.write(example.SerializeToString())
	writer.close()

def main():
	cifar10_dir = 'cifar-10-batches-py'
	# for i in range(1, 6):
	# 	f = os.path.join(cifar10_dir, 'data_batch_%d' % (i,))
	# 	print('open the file: {}'.format(f))
	# 	x, y = load_CIFAR_batch(f)
	# 	convert(x, y, 'train')

	xt, yt = load_CIFAR_batch(os.path.join(cifar10_dir, 'test_batch'))
	convert(xt, yt, 'test')
	print('finish tfrecord!')

if __name__ == '__main__':
	main()