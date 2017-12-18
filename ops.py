import tensorflow as tf
import tensorflow.contrib.layers as layers

def conv2d(input, output_shape, k_h=5, k_w=5, stddev=0.02, padding='VALID', name='conv2d'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape], 
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=padding)

		biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
		conv = tf.nn.bias_add(conv, biases)

	return conv

def max_pool(input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def fc(input, output_shape, stddev=0.02, name='fc'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [input.shape[-1], output_shape], 
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
		f = tf.matmul(input, w)
		f = tf.nn.bias_add(f, biases)

	return f