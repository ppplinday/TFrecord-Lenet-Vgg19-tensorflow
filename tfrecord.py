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
	#image = tf.decode_raw(features['image'],tf.uint8)
	label = tf.cast(features['label'],tf.int32)

	print(image)

	images, labels = tf.train.batch([image, label],
		batch_size=128,
		num_threads = 1,
		capacity = 10 * 128,
		)

	images_batch = tf.cast(images_batch, tf.float32)
	print(image.shape)
	print(label.shape)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		a,b = sess.run([image, label])
		print(image)
		print(label)
		print('rrrrrr')
		print(a.shape)
		print(b.shape)
		print(a)
		print(b)
		coord.request_stop()
		coord.join(threads)


if __name__ == '__main__':
	#main()
	test_tfrecords()