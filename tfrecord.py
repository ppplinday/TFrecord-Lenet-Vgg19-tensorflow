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

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features={
		'label':tf.FixedLenFeature([],tf.int64),
		'image':tf.FixedLenFeature([],tf.string)
		})

	image = tf.decode_raw(features['image'],tf.uint8)
	label = tf.cast(features['label'],tf.int32)

	image.set_shape([32*32*3])
	image = tf.reshape(image, [32, 32, 3])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	return image, label

def inputs(data_set, batch_size):
	print('welcome to tfrecords')
	if data_set == 'train':
		file = 'train.tfrecords'
	else:
		file = 'validation.tfrecords'
	print('tttttt')
	with tf.name_scope('input') as scope:
		filename_queue = tf.train.string_input_producer([file], num_epochs=None)
	print('tttttt11111')
	image, label = read_and_decode(filename_queue)
	print('tttttt22222')
	images, labels = tf.train.batch([image, label],
		batch_size=batch_size,
		num_threads = 1,
		capacity = 10 * batch_size,
		)

	return images, labels

def main():
	cifar10_dir = 'cifar-10-batches-py'
	for i in range(1, 6):
		f = os.path.join(cifar10_dir, 'data_batch_%d' % (i,))
		print('open the file: {}'.format(f))
		x, y = load_CIFAR_batch(f)
		convert(x, y, 'train')

	xt, yt = load_CIFAR_batch(os.path.join(cifar10_dir, 'test_batch'))
	convert(xt, yt, 'test')
	print('finish tfrecord!')

def test_tfrecords():
	images, labels = inputs('train', 128)
	print(images.shape)
	print(labels.shape)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		print(images.eval())
		print(labels.eval())
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	#main()
	test_tfrecords()