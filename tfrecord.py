import os
import numpy as np
import pickle as p
import tensorflow as tf
from load_data import load_CIFAR_batch, load_CIFAR10
from data_preprocess import _preprocess, transform, transform_test, data_preprocess

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
	image_placeholder = tf.placeholder(tf.uint8)
	encoded_image = tf.image.encode_png(image_placeholder)

	cifar10_dir = 'cifar-10-batches-py'
	
	with tf.Session() as sess:
		writer = tf.python_io.TFRecordWriter('train.tfrecords')
		for i in range(1, 6):
			f = os.path.join(cifar10_dir, 'data_batch_%d' % (i,))
			print('open the file: {}'.format(f))
			images, labels = load_CIFAR_batch(f)
			for i in range(10000):
				png_string = sess.run(encoded_image, feed_dict={image_placeholder: images[i]})
				#img = images[i].tostring()
				example = tf.train.Example(features=tf.train.Features(feature={
					'label':_int64_feature(int(labels[i])),
					'image':_bytes_feature(png_string)
					}))
				writer.write(example.SerializeToString())
		writer.close()

		writer = tf.python_io.TFRecordWriter('test.tfrecords')
		print('open the test file')
		images, labels = load_CIFAR_batch(os.path.join(cifar10_dir, 'test_batch'))
		for i in range(10000):
			png_string = sess.run(encoded_image, feed_dict={image_placeholder: images[i]})
			#img = images[i].tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'label':_int64_feature(int(labels[i])),
				'image':_bytes_feature(png_string)
				}))
			writer.write(example.SerializeToString())
		writer.close()

	print('finish tfrecord!')

def test_tfrecords():
	reader = tf.TFRecordReader()
	file = 'train.tfrecords'
	filename_queue = tf.train.string_input_producer([file], num_epochs=None)
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features={
		'label':tf.FixedLenFeature([],tf.int64),
		'image':tf.FixedLenFeature([],tf.string)
		})

	image = tf.image.decode_png(features['image'], channels=3)
	image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
	#image = tf.decode_raw(features['image'],tf.uint8)
	label = tf.cast(features['label'],tf.int32)

	images, labels = tf.train.batch([image, label],
		batch_size=128,
		num_threads = 1,
		capacity = 10 * 128,
		)

	images = tf.cast(images, tf.float32)
	
	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		a,b = sess.run([images, labels])
		print('rrrrrr')
		print(a.shape)
		print(b.shape)
		print(a)
		print(b)
		c = data_preprocess(a)
		print(c.shape)
		coord.request_stop()
		coord.join(threads)

	cifar10_dir = 'cifar-10-batches-py'
	X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
	print(X_train[0])
	print(Y_train[0])



def input(data_set, batch_size):
	reader = tf.TFRecordReader()
	if data_set == 'train':
		file = 'train.tfrecords'
	else:
		file = 'test.tfrecords'

	filename_queue = tf.train.string_input_producer([file], num_epochs=None)
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features={
		'label':tf.FixedLenFeature([],tf.int64),
		'image':tf.FixedLenFeature([],tf.string)
		})

	image = tf.image.decode_png(features['image'], channels=3)
	image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
	label = tf.cast(features['label'],tf.int32)

	images, labels = tf.train.batch([image, label],
		batch_size=batch_size,
		num_threads = 1,
		capacity = batch_size,
		)

	images = tf.cast(images, tf.float32)

	return images, labels

if __name__ == '__main__':
	main()
	test_tfrecords()